# -*- coding: utf-8 -*-
"""
akshare适配器 - 老王说：内部多数据源自动切换！
东财挂了切同花顺，同花顺挂了切新浪，新浪挂了切腾讯...
"""
import akshare as ak
import pandas as pd
import inspect
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from .base_adapter import BaseAdapter


logger = logging.getLogger(__name__)


class AkshareAdapter(BaseAdapter):
    """akshare数据源适配器，支持内部多数据源冗余"""

    _SPOT_CACHE_TTL_SECONDS = 6 * 60 * 60
    _SPOT_CACHE_EMPTY_TTL_SECONDS = 10 * 60
    _YAHOO_NAME_TTL_SECONDS = 7 * 24 * 60 * 60
    _HK_HIST_FAIL_WINDOW_SECONDS = 120
    _HK_HIST_FAIL_LIMIT = 3
    _HK_HIST_BLOCK_SECONDS = 60  # Reduced from 600s to 60s
    _HK_HIST_TIMEOUT_SECONDS = 8

    # 字段映射：统一不同数据源的返回格式
    FIELD_MAPPING = {
        'stock_zh_a_hist': {
            '日期': 'date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount'
        },
        'stock_zh_a_hist_tx': {},  # 腾讯接口字段已是英文
        'stock_hk_hist': {
            '日期': 'date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount'
        },
        'stock_us_hist': {
            '日期': 'date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount'
        },
    }

    @property
    def name(self) -> str:
        return "akshare"

    def _format_code_for_tx(self, code: str) -> str:
        """转换股票代码为腾讯格式：000001 -> sz000001"""
        code = code.replace('.SH', '').replace('.SZ', '').replace('sh', '').replace('sz', '')
        prefix = 'sh' if code.startswith('6') else 'sz'
        return f"{prefix}{code}"

    def _format_code_for_hk(self, code: str) -> str:
        code = (code or '').strip()
        code = code.replace('.HK', '').replace('.hk', '')
        digits = ''.join([c for c in code if c.isdigit()])
        if digits:
            return digits.zfill(5)
        return code

    def _normalize_us_symbol(self, code: str) -> str:
        c = (code or '').strip().upper()
        if not c:
            return c
        if ':' in c:
            c = c.split(':')[-1]
        c = re.sub(r'\s+', '', c)
        if '.' in c:
            c = c.split('.')[0]
        c = re.sub(r'[^A-Z0-9\-]', '', c)
        return c

    def _safe_call_ak(self, func, **kwargs):
        try:
            sig = inspect.signature(func)
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                return func(**kwargs)
            call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            if len(sig.parameters) == 0:
                return func()
            return func(**call_kwargs)
        except Exception:
            return None

    def _format_date_dash(self, date_str: str) -> str:
        if not date_str:
            return ''
        s = str(date_str).strip()
        if not s:
            return ''
        try:
            if len(s) == 8 and s.isdigit():
                return f"{s[:4]}-{s[4:6]}-{s[6:]}"
            dt = pd.to_datetime(s, errors='coerce')
            if pd.isna(dt):
                return s
            if isinstance(dt, pd.Timestamp):
                dt = dt.to_pydatetime()
            return dt.strftime('%Y-%m-%d')
        except Exception:
            return s

    def _fetch_hk_history_tencent(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        hk_code = self._format_code_for_hk(code)
        if not hk_code:
            return pd.DataFrame()

        symbol = f"hk{hk_code}"
        start_s = self._format_date_dash(start_date)
        end_s = self._format_date_dash(end_date)
        count = 640
        try:
            start_dt = pd.to_datetime(start_s, errors='coerce')
            end_dt = pd.to_datetime(end_s, errors='coerce')
            if pd.notna(start_dt) and pd.notna(end_dt):
                days = int((end_dt - start_dt).days) + 5
                count = max(120, min(max(days, 120), 1000))
        except Exception:
            pass

        params = f"{symbol},day,{start_s},{end_s},{count},qfq"
        url = (
            'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param='
            f"{urllib.parse.quote(params)}"
        )
        try:
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Accept': 'application/json,text/plain,*/*',
                },
                method='GET',
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode('utf-8', errors='ignore')
            logger.debug(f'Tencent HK API response length: {len(raw)} for {symbol}')
        except Exception as e:
            logger.warning(f'Tencent HK API request failed: {e}')
            return pd.DataFrame()

        raw = (raw or '').strip()
        if not raw:
            return pd.DataFrame()
        if not raw.startswith('{'):
            match = re.search(r'(\{.*\})', raw, re.S)
            if not match:
                return pd.DataFrame()
            raw = match.group(1)
        try:
            payload = json.loads(raw)
        except Exception:
            return pd.DataFrame()

        data = payload.get('data') if isinstance(payload, dict) else None
        if not isinstance(data, dict) or not data:
            return pd.DataFrame()

        entry = data.get(symbol) or data.get(symbol.lower()) or data.get(symbol.upper())
        if not isinstance(entry, dict) and data:
            entry = next(iter(data.values()))
        if not isinstance(entry, dict):
            return pd.DataFrame()

        rows = entry.get('qfqday') or entry.get('day') or []
        if not rows:
            return pd.DataFrame()

        out_rows = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                continue
            out_rows.append({
                'date': row[0],
                'open': row[1],
                'close': row[2],
                'high': row[3],
                'low': row[4],
                'volume': row[5],
                'amount': row[6] if len(row) > 6 else None,
            })
        if not out_rows:
            return pd.DataFrame()

        df = pd.DataFrame(out_rows)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['date', 'close'])
        if start_s:
            start_dt = pd.to_datetime(start_s, errors='coerce')
            if pd.notna(start_dt):
                df = df[df['date'] >= start_dt]
        if end_s:
            end_dt = pd.to_datetime(end_s, errors='coerce')
            if pd.notna(end_dt):
                df = df[df['date'] <= end_dt]
        return df

    def _hk_hist_is_blocked(self) -> bool:
        now = time.time()
        return now < float(getattr(self, '_hk_hist_block_until', 0.0) or 0.0)

    def _record_hk_hist_failure(self, err: Exception) -> None:
        now = time.time()
        last_ts = float(getattr(self, '_hk_hist_last_fail_ts', 0.0) or 0.0)
        count = int(getattr(self, '_hk_hist_fail_count', 0))
        if now - last_ts > self._HK_HIST_FAIL_WINDOW_SECONDS:
            count = 0
        count += 1
        self._hk_hist_last_fail_ts = now
        self._hk_hist_fail_count = count
        if count >= self._HK_HIST_FAIL_LIMIT:
            self._hk_hist_block_until = now + float(self._HK_HIST_BLOCK_SECONDS)
            logger.warning(
                "HK history blocked for %ss after %d failures; last_error=%s",
                int(self._HK_HIST_BLOCK_SECONDS),
                count,
                err,
            )

    def _reset_hk_hist_failures(self) -> None:
        self._hk_hist_fail_count = 0
        self._hk_hist_last_fail_ts = 0.0
        self._hk_hist_block_until = 0.0

    def _to_epoch_seconds(self, date_str: str, end_of_day: bool = False) -> Optional[int]:
        if not date_str:
            return None
        s = str(date_str).strip()
        if not s:
            return None
        try:
            if len(s) == 8 and s.isdigit():
                dt = datetime.strptime(s, '%Y%m%d')
            else:
                dt = pd.to_datetime(s, errors='coerce')
                if pd.isna(dt):
                    return None
                if isinstance(dt, pd.Timestamp):
                    dt = dt.to_pydatetime()
            if end_of_day:
                dt = dt + timedelta(days=1)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception:
            return None

    def _fetch_hk_history_yahoo(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        hk_code = self._format_code_for_hk(code)
        if not hk_code:
            return pd.DataFrame()

        start_ts = self._to_epoch_seconds(start_date, end_of_day=False)
        end_ts = self._to_epoch_seconds(end_date, end_of_day=True)
        if not start_ts or not end_ts:
            return pd.DataFrame()

        symbol = f"{hk_code}.HK"
        params = {
            'period1': str(start_ts),
            'period2': str(end_ts),
            'interval': '1d',
            'events': 'history',
            'includeAdjustedClose': 'true',
        }
        url = (
            'https://query1.finance.yahoo.com/v8/finance/chart/'
            f"{urllib.parse.quote(symbol)}?{urllib.parse.urlencode(params)}"
        )

        try:
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Accept': 'application/json,text/plain,*/*',
                },
                method='GET',
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode('utf-8', errors='ignore')
            payload = json.loads(raw) if raw else {}
        except Exception:
            return pd.DataFrame()

        result_list = (payload.get('chart') or {}).get('result') or []
        if not result_list:
            return pd.DataFrame()
        result = result_list[0] if isinstance(result_list[0], dict) else {}
        timestamps = result.get('timestamp') or []
        if not timestamps:
            return pd.DataFrame()

        indicators = result.get('indicators') or {}
        quote = (indicators.get('quote') or [{}])
        quote = quote[0] if quote else {}
        adj = (indicators.get('adjclose') or [{}])
        adj = adj[0] if adj else {}

        n = len(timestamps)

        def _slice(vals):
            if not isinstance(vals, list):
                return [None] * n
            if len(vals) < n:
                return vals + [None] * (n - len(vals))
            return vals[:n]

        close_vals = _slice(adj.get('adjclose'))
        if not any(v is not None for v in close_vals):
            close_vals = _slice(quote.get('close'))

        df = pd.DataFrame({
            'date': pd.to_datetime(timestamps, unit='s', utc=True).tz_convert(None),
            'open': _slice(quote.get('open')),
            'high': _slice(quote.get('high')),
            'low': _slice(quote.get('low')),
            'close': close_vals,
            'volume': _slice(quote.get('volume')),
        })
        if df.empty:
            return df
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
        df['amount'] = None
        return df

    def _pick_first_value(self, info: Dict, keys: List[str]) -> str:
        for k in keys:
            v = info.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if s and s.lower() != 'nan':
                return s
        return ''

    def _ensure_yahoo_name_cache(self):
        if not hasattr(self, '_yahoo_name_cache'):
            self._yahoo_name_cache = {}

    def _fetch_us_name_from_yahoo(self, symbol: str) -> str:
        s = self._normalize_us_symbol(symbol)
        if not s:
            return ''

        self._ensure_yahoo_name_cache()
        now = time.time()
        cached = self._yahoo_name_cache.get(s)
        if cached and (now - float(cached.get('ts') or 0.0)) < self._YAHOO_NAME_TTL_SECONDS:
            return str(cached.get('name') or '').strip()

        try:
            q = urllib.parse.quote(s)
            url = (
                'https://query1.finance.yahoo.com/v1/finance/search'
                f'?q={q}&quotesCount=10&newsCount=0&listsCount=0&enableFuzzyQuery=false'
            )
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Accept': 'application/json,text/plain,*/*',
                },
                method='GET',
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw = resp.read().decode('utf-8', errors='ignore')
            data = json.loads(raw) if raw else {}
            quotes = data.get('quotes') or []

            best = None
            for item in quotes:
                sym = str(item.get('symbol') or '').upper().strip()
                if sym == s:
                    best = item
                    break
            if best is None and quotes:
                best = quotes[0]

            if isinstance(best, dict):
                name = (
                    str(best.get('longname') or '').strip()
                    or str(best.get('shortname') or '').strip()
                    or str(best.get('name') or '').strip()
                )
                if name:
                    self._yahoo_name_cache[s] = {'ts': now, 'name': name}
                    return name
        except Exception:
            pass

        self._yahoo_name_cache[s] = {'ts': now, 'name': ''}
        return ''

    def _extract_name_from_df(self, df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return ''

        if 'item' in df.columns and 'value' in df.columns:
            try:
                info = dict(zip(df['item'], df['value']))
                return self._pick_first_value(info, [
                    'org_short_name_cn',
                    'org_name_cn',
                    'org_short_name_en',
                    'org_name_en',
                    'comcnname',
                    'comenname',
                    'name',
                    '名称',
                    '中文名称',
                    '英文名称',
                    '股票简称',
                ])
            except Exception:
                return ''

        name_col = None
        for c in ['名称', '中文名称', '英文名称', 'name', 'Name', '公司名称', '证券简称']:
            if c in df.columns:
                name_col = c
                break
        if name_col is None:
            return ''

        try:
            v = df.iloc[0][name_col]
            s = str(v).strip()
            return '' if (not s or s.lower() == 'nan') else s
        except Exception:
            return ''

    def _extract_name_from_any(self, obj) -> str:
        if obj is None:
            return ''
        if isinstance(obj, pd.DataFrame):
            return self._extract_name_from_df(obj)
        if isinstance(obj, dict):
            return self._pick_first_value(obj, [
                'org_short_name_cn',
                'org_name_cn',
                'org_short_name_en',
                'org_name_en',
                'comcnname',
                'comenname',
                'name',
                '名称',
                '中文名称',
                '英文名称',
                '股票简称',
            ])
        return ''

    def _try_us_name_by_introspection(self, symbol: str) -> str:
        s = self._normalize_us_symbol(symbol)
        if not s:
            return ''

        preferred_tokens = [
            'profile',
            'fundamental',
            'basic',
            'info',
            'individual',
        ]

        candidates = []
        for name in dir(ak):
            n = name.lower()
            if not n.startswith('stock_us'):
                continue
            if any(t in n for t in preferred_tokens):
                candidates.append(name)

        tried = 0
        for fname in sorted(set(candidates)):
            func = getattr(ak, fname, None)
            if not callable(func):
                continue

            result = None
            for key in ['symbol', 'ticker', 'stock', 'code']:
                result = self._safe_call_ak(func, **{key: s})
                if result is not None:
                    break

            name = self._extract_name_from_any(result)
            if name:
                return name

            tried += 1
            if tried >= 25:
                break

        return ''

    def _ensure_spot_cache(self):
        if not hasattr(self, '_spot_cache'):
            self._spot_cache = {
                'HK': {'ts': 0.0, 'map': {}},
                'US': {'ts': 0.0, 'map': {}},
            }

    def _pick_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _refresh_hk_spot_cache(self):
        self._ensure_spot_cache()
        now = time.time()
        bucket = self._spot_cache['HK']
        age = now - float(bucket.get('ts') or 0.0)
        if bucket['map']:
            if age < self._SPOT_CACHE_TTL_SECONDS:
                return
        else:
            if age < self._SPOT_CACHE_EMPTY_TTL_SECONDS:
                return

        mapping: Dict[str, str] = {}
        df = None
        try:
            df = ak.stock_hk_spot_em()
        except Exception:
            df = None

        if df is None or df.empty:
            try:
                df = ak.stock_hk_spot()
            except Exception:
                df = None

        if df is None or df.empty:
            bucket['ts'] = now
            bucket['map'] = {}
            return

        code_col = self._pick_column(df, ['代码', 'symbol', 'Symbol'])
        name_col = self._pick_column(df, ['名称', '中文名称', 'name', 'Name'])
        if not code_col or not name_col:
            bucket['ts'] = now
            bucket['map'] = {}
            return

        for _, row in df.iterrows():
            c = row.get(code_col)
            n = row.get(name_col)
            if c is None or n is None:
                continue
            code = self._format_code_for_hk(str(c))
            name = str(n).strip()
            if code and name and name.lower() != 'nan':
                mapping[code] = name

        bucket['ts'] = now
        bucket['map'] = mapping

    def _refresh_us_spot_cache(self):
        self._ensure_spot_cache()
        now = time.time()
        bucket = self._spot_cache['US']
        age = now - float(bucket.get('ts') or 0.0)
        if bucket['map']:
            if age < self._SPOT_CACHE_TTL_SECONDS:
                return
        else:
            if age < self._SPOT_CACHE_EMPTY_TTL_SECONDS:
                return

        mapping: Dict[str, str] = {}
        df = None

        preferred = [
            'stock_us_spot_em',
            'stock_us_spot',
            'stock_us_famous_spot_em',
            'stock_us_famous_spot',
        ]
        discovered = [
            n for n in dir(ak)
            if n.lower().startswith('stock_us') and ('spot' in n.lower())
        ]
        func_names: List[str] = []
        for n in preferred:
            if n in discovered and n not in func_names:
                func_names.append(n)
        for n in sorted(discovered):
            if n not in func_names:
                func_names.append(n)

        for fname in func_names:
            func = getattr(ak, fname, None)
            if not callable(func):
                continue
            df = self._safe_call_ak(func)
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                break
            df = None

        if df is None or df.empty:
            bucket['ts'] = now
            bucket['map'] = {}
            if func_names:
                logger.info(f"US spot cache empty; tried: {func_names[:12]}")
            return

        code_col = self._pick_column(df, ['代码', 'symbol', 'Symbol', 'ticker', 'Ticker'])
        name_col = self._pick_column(df, ['名称', '中文名称', 'name', 'Name', '英文名称'])
        if not code_col or not name_col:
            bucket['ts'] = now
            bucket['map'] = {}
            logger.info(f"US spot cache columns not matched: {list(df.columns)[:30]}")
            return

        for _, row in df.iterrows():
            c = row.get(code_col)
            n = row.get(name_col)
            if c is None or n is None:
                continue
            raw_code = str(c).strip()
            name = str(n).strip()
            if not raw_code or not name or name.lower() == 'nan':
                continue

            k1 = raw_code.upper()
            k2 = self._normalize_us_symbol(raw_code)
            if k1:
                mapping[k1] = name
            if k2:
                mapping[k2] = name

        bucket['ts'] = now
        bucket['map'] = mapping

    def get_stock_history(self, code: str, start_date: str, end_date: str,
                          adjust: str = "qfq", market_type: str = "A") -> pd.DataFrame:
        """获取股票历史K线 - 多市场支持"""
        mt = (market_type or 'A').strip().upper()

        if mt == 'HK':
            hk_code = self._format_code_for_hk(code)
            if not self._hk_hist_is_blocked():
                try:
                    params = {
                        'symbol': hk_code,
                        'period': 'daily',
                        'start_date': start_date,
                        'end_date': end_date,
                        'adjust': adjust or "",
                    }
                    sig = inspect.signature(ak.stock_hk_hist)
                    if 'timeout' in sig.parameters:
                        params['timeout'] = int(self._HK_HIST_TIMEOUT_SECONDS)
                    df = ak.stock_hk_hist(**params)
                    if df is not None and not df.empty:
                        self._reset_hk_hist_failures()
                        df = df.rename(columns=self.FIELD_MAPPING['stock_hk_hist'])
                        return df
                except Exception as exc:
                    self._record_hk_hist_failure(exc)

            # Try sina HK (stock_hk_daily) as fallback
            try:
                df = ak.stock_hk_daily(symbol=hk_code, adjust="qfq")
                if df is not None and not df.empty:
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        start_dt = pd.to_datetime(start_date, errors='coerce')
                        end_dt = pd.to_datetime(end_date, errors='coerce')
                        if pd.notna(start_dt):
                            df = df[df['date'] >= start_dt]
                        if pd.notna(end_dt):
                            df = df[df['date'] <= end_dt]
                    if 'amount' not in df.columns:
                        df['amount'] = None
                    return df
            except Exception:
                pass

            df = self._fetch_hk_history_tencent(hk_code, start_date, end_date)
            if df is not None and not df.empty:
                return df

            df = self._fetch_hk_history_yahoo(hk_code, start_date, end_date)
            if df is not None and not df.empty:
                return df
            return pd.DataFrame()

        if mt == 'US':
            us_code = (code or '').strip()

            # 东财美股历史
            try:
                df = ak.stock_us_hist(
                    symbol=us_code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust or "",
                )
                if df is not None and not df.empty:
                    df = df.rename(columns=self.FIELD_MAPPING['stock_us_hist'])
                    return df
            except Exception:
                pass

            # 新浪美股日线（不支持 start/end 参数，需本地过滤）
            try:
                sina_adjust = 'qfq' if (adjust or '').lower() == 'qfq' else ''
                df = ak.stock_us_daily(symbol=us_code, adjust=sina_adjust)
                if df is not None and not df.empty:
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        start_dt = pd.to_datetime(start_date, errors='coerce')
                        end_dt = pd.to_datetime(end_date, errors='coerce')
                        if pd.notna(start_dt):
                            df = df[df['date'] >= start_dt]
                        if pd.notna(end_dt):
                            df = df[df['date'] <= end_dt]
                    if 'amount' not in df.columns:
                        df['amount'] = None
                    return df
            except Exception:
                pass

            return pd.DataFrame()

        # Default: A-share
        code = (code or '').replace('.SH', '').replace('.SZ', '').replace('sh', '').replace('sz', '')

        # 尝试东财接口
        try:
            df = ak.stock_zh_a_hist(symbol=code, start_date=start_date,
                                    end_date=end_date, adjust=adjust)
            if df is not None and not df.empty:
                df = df.rename(columns=self.FIELD_MAPPING['stock_zh_a_hist'])
                return df
        except Exception:
            pass

        # 东财挂了，切腾讯
        try:
            tx_code = self._format_code_for_tx(code)
            df = ak.stock_zh_a_hist_tx(symbol=tx_code, start_date=start_date,
                                       end_date=end_date, adjust=adjust)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

        return pd.DataFrame()

    def get_index_stocks(self, index_code: str) -> List[str]:
        """获取指数成分股"""
        try:
            df = ak.index_stock_cons_weight_csindex(symbol=index_code)
            if df is not None and not df.empty:
                col = '成分券代码' if '成分券代码' in df.columns else df.columns[0]
                return df[col].tolist()
        except Exception:
            pass
        return []

    def get_stock_info(self, code: str, market_type: str = "A") -> Dict:
        """获取股票基本信息 - 东财→雪球"""
        mt = (market_type or 'A').strip().upper()

        if mt == 'HK':
            hk_code = self._format_code_for_hk(code)
            try:
                df = ak.stock_individual_basic_info_hk_xq(symbol=hk_code)
                if df is not None and not df.empty:
                    return dict(zip(df['item'], df['value']))
            except Exception:
                pass
            try:
                self._refresh_hk_spot_cache()
                name = (self._spot_cache.get('HK') or {}).get('map', {}).get(hk_code, '')
                if name:
                    return {'comcnname': name}
            except Exception:
                pass
            return {'code_name': hk_code}

        if mt == 'US':
            us_code = (code or '').strip()
            try:
                df = ak.stock_individual_basic_info_us_xq(symbol=us_code)
                if df is not None and not df.empty:
                    return dict(zip(df['item'], df['value']))
            except Exception:
                pass
            try:
                self._refresh_us_spot_cache()
                us_key = self._normalize_us_symbol(us_code)
                spot_map = (self._spot_cache.get('US') or {}).get('map', {})
                name = spot_map.get(us_code.upper(), '') or spot_map.get(us_key, '')
                if name:
                    return {'org_short_name_en': name}
            except Exception:
                pass
            try:
                name = self._try_us_name_by_introspection(us_code)
                if name:
                    return {'org_short_name_en': name}
            except Exception:
                pass
            try:
                name = self._fetch_us_name_from_yahoo(us_code)
                if name:
                    return {'org_short_name_en': name}
            except Exception:
                pass
            return {'code_name': self._normalize_us_symbol(us_code) or us_code}

        code = (code or '').replace('.SH', '').replace('.SZ', '').replace('sh', '').replace('sz', '')

        # 东财
        try:
            df = ak.stock_individual_info_em(symbol=code)
            if df is not None and not df.empty:
                return dict(zip(df['item'], df['value']))
        except Exception:
            pass

        # 雪球
        try:
            df = ak.stock_individual_basic_info_xq(symbol=code)
            if df is not None and not df.empty:
                return df.to_dict('records')[0] if len(df) > 0 else {}
        except Exception:
            pass

        return {}

    def get_financial_data(self, code: str) -> Dict:
        """获取财务数据 - 东财→同花顺"""
        code = code.replace('.SH', '').replace('.SZ', '').replace('sh', '').replace('sz', '')

        # 东财财务分析指标
        try:
            df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2023")
            if df is not None and not df.empty:
                return {'indicator': df.to_dict('records')}
        except Exception:
            pass

        # 同花顺财务摘要
        try:
            df = ak.stock_financial_abstract_ths(symbol=code)
            if df is not None and not df.empty:
                return {'abstract': df.to_dict('records')}
        except Exception:
            pass

        return {}

    def get_board_stocks(self, board: str) -> List[str]:
        """获取板块股票列表"""
        board_key = (board or '').strip().lower()
        board_map = {
            'all': 'stock_zh_a_spot_em',
            'sh': 'stock_sh_a_spot_em',
            'sz': 'stock_sz_a_spot_em',
            'bj': 'stock_bj_a_spot_em',
            'cyb': 'stock_cy_a_spot_em',
            'kcb': 'stock_kc_a_spot_em',
        }
        func_name = board_map.get(board_key)
        if not func_name:
            return []

        candidates: List[str] = []

        def _add_candidate(name: str) -> None:
            if name and name not in candidates:
                candidates.append(name)

        _add_candidate(func_name)
        if func_name.endswith('_em'):
            _add_candidate(func_name[:-3])
        else:
            _add_candidate(f"{func_name}_em")

        prefix_map = {
            'all': 'stock_zh_a_spot',
            'sh': 'stock_sh_a_spot',
            'sz': 'stock_sz_a_spot',
            'bj': 'stock_bj_a_spot',
            'cyb': 'stock_cy_a_spot',
            'kcb': 'stock_kc_a_spot',
        }
        prefix = prefix_map.get(board_key)
        if prefix:
            for name in dir(ak):
                if name.lower().startswith(prefix.lower()):
                    _add_candidate(name)

        for fname in candidates:
            func = getattr(ak, fname, None)
            if not callable(func):
                continue
            # 重试机制：akshare API 不稳定时多次尝试
            df = None
            for retry in range(5):
                df = self._safe_call_ak(func)
                if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                    break
                if retry < 4:
                    import time
                    time.sleep(3)
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                continue
            col = '代码' if '代码' in df.columns else df.columns[0]
            codes = [str(x).strip() for x in df[col].tolist() if str(x).strip()]
            if codes:
                return codes

        if candidates:
            logger.warning("A board stocks empty for board=%s; tried=%s", board, candidates[:10])
        return []

    def get_industry_list(self) -> pd.DataFrame:
        """获取行业板块列表 - 东财→同花顺"""
        # 东财
        try:
            df = ak.stock_board_industry_name_em()
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

        # 同花顺
        try:
            df = ak.stock_board_industry_summary_ths()
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

        return pd.DataFrame()

    def get_industry_stocks(self, industry: str) -> List[str]:
        """获取行业成分股"""
        try:
            df = ak.stock_board_industry_cons_em(symbol=industry)
            if df is not None and not df.empty:
                col = '代码' if '代码' in df.columns else df.columns[0]
                return df[col].tolist()
        except Exception:
            pass
        return []

    def get_concept_stocks(self, concept: str) -> List[str]:
        """获取概念板块成分股代码列表"""
        # 先尝试概念板块
        try:
            df = ak.stock_board_concept_cons_em(symbol=concept)
            if df is not None and not df.empty:
                col = '代码' if '代码' in df.columns else df.columns[0]
                return df[col].tolist()
        except Exception:
            pass

        # 概念失败，尝试行业板块
        try:
            df = ak.stock_board_industry_cons_em(symbol=concept)
            if df is not None and not df.empty:
                col = '代码' if '代码' in df.columns else df.columns[0]
                return df[col].tolist()
        except Exception:
            pass

        return []

    def get_concept_stocks_detail(self, concept: str) -> List[Dict]:
        """获取概念板块成分股详细信息（含名称、价格等）"""
        # 先尝试概念板块
        try:
            df = ak.stock_board_concept_cons_em(symbol=concept)
            if df is not None and not df.empty:
                return self._parse_board_stocks_df(df)
        except Exception:
            pass

        # 概念失败，尝试行业板块
        try:
            df = ak.stock_board_industry_cons_em(symbol=concept)
            if df is not None and not df.empty:
                return self._parse_board_stocks_df(df)
        except Exception:
            pass

        return []

    def _parse_board_stocks_df(self, df) -> List[Dict]:
        """解析板块成分股DataFrame为字典列表"""
        import math
        result = []
        for _, row in df.iterrows():
            price = row.get("最新价", 0)
            change = row.get("涨跌幅", 0)
            # 处理NaN值
            price = 0 if (price is None or (isinstance(price, float) and math.isnan(price))) else float(price)
            change = 0 if (change is None or (isinstance(change, float) and math.isnan(change))) else float(change)
            item = {
                "code": str(row.get("代码", "")),
                "name": str(row.get("名称", "")),
                "price": price,
                "change_percent": change,
                "main_net_inflow": 0,
                "main_net_inflow_percent": 0
            }
            result.append(item)
        return result

    def get_capital_flow(self, code: str) -> Dict:
        """获取资金流向"""
        try:
            df = ak.stock_individual_fund_flow(stock=code, market="sh" if code.startswith('6') else "sz")
            if df is not None and not df.empty:
                return {'flow': df.to_dict('records')}
        except Exception:
            pass
        return {}

    def get_north_flow(self) -> pd.DataFrame:
        """获取北向资金"""
        try:
            df = ak.stock_hsgt_hist_em(symbol="沪股通")
            return df if df is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def health_check(self) -> bool:
        """健康检查"""
        try:
            df = ak.stock_zh_a_spot_em()
            return df is not None and len(df) > 0
        except Exception:
            return False
