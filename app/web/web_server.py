# -*- coding: utf-8 -*-
"""
基础Web服务器架构 - 基于Flask的股票预测管理系统设计与实现
版本：v1.0.0
"""
import os
import logging
import threading
import time
import json
import re
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

# Flask相关导入
from flask import Flask, render_template, request, jsonify

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 环境变量加载
from dotenv import load_dotenv

from ..core.data_provider import get_data_provider
from ..core.news_fetcher import news_fetcher, start_news_scheduler

load_dotenv()

# 创建Flask应用实例
app = Flask(__name__)

# 配置日志
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
log_file = os.getenv('LOG_FILE', 'data/logs/server.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"基础Web服务器已启动，日志级别: {log_level}")

start_news_scheduler()

class _WerkzeugProgressFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            if '/api/progress' in msg:
                return False
        except Exception:
            return True
        return True


logging.getLogger('werkzeug').addFilter(_WerkzeugProgressFilter())


def _allow_insecure_https() -> bool:
    v = (os.getenv('ALLOW_INSECURE_HTTPS') or '').strip().lower()
    return v in {'1', 'true', 'yes', 'y', 'on'}


def _urlopen_read_text(req: urllib.request.Request, timeout: int) -> str:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        msg = str(e)
        if 'CERTIFICATE_VERIFY_FAILED' not in msg:
            raise

        url = getattr(req, 'full_url', '')

        try:
            import certifi  # type: ignore

            ctx = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                return resp.read().decode('utf-8', errors='ignore')
        except Exception as e2:
            if not _allow_insecure_https():
                raise urllib.error.URLError(
                    f"{e} (certifi_retry_failed={type(e2).__name__}: {e2}; "
                    "set ALLOW_INSECURE_HTTPS=1 to bypass verification as a temporary workaround)"
                )

        logger.warning('ssl verify failed; retrying with insecure context url=%s', url)
        ctx2 = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx2) as resp:
            return resp.read().decode('utf-8', errors='ignore')


_progress_lock = threading.Lock()
_progress_store: Dict[str, Dict[str, Any]] = {}

_hotspot_lock = threading.Lock()
_HOTSPOT_CACHE_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'hotspot_cache.json'))

def _load_hotspot_cache() -> Dict[str, Any]:
    try:
        if os.path.exists(_HOTSPOT_CACHE_FILE):
            with open(_HOTSPOT_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {'ts': 0.0, 'fail_ts': 0.0, 'items': [], 'source': ''}

def _save_hotspot_cache(cache_data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(_HOTSPOT_CACHE_FILE), exist_ok=True)
        with open(_HOTSPOT_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False)
    except Exception as e:
        logger.warning('Failed to save hotspot cache: %s', e)

_gru_model_cache: Dict[str, Any] = {
    'model': None,
    'meta': None,
    'mtime': 0.0,
}

_lstm_model_cache: Dict[str, Any] = {
    'model': None,
    'meta': None,
    'mtime': 0.0,
}


def _progress_set(request_id: str, percent: int, stage: str) -> None:
    rid = (request_id or '').strip()
    if not rid:
        return
    p = int(max(0, min(100, percent)))
    s = (stage or '').strip()
    with _progress_lock:
        _progress_store[rid] = {
            'percent': p,
            'stage': s,
            'updated_at': time.time(),
        }


def _progress_get(request_id: str) -> Dict[str, Any]:
    rid = (request_id or '').strip()
    if not rid:
        return {}
    with _progress_lock:
        return dict(_progress_store.get(rid) or {})


def _progress_cleanup(max_age_seconds: int = 30 * 60) -> None:
    now = time.time()
    with _progress_lock:
        keys = list(_progress_store.keys())
        for k in keys:
            v = _progress_store.get(k) or {}
            ts = float(v.get('updated_at') or 0.0)
            if now - ts > max_age_seconds:
                _progress_store.pop(k, None)


def _trigger_news_fetch_async() -> None:
    def _run_fetch() -> None:
        try:
            news_fetcher.fetch_and_save()
        except Exception:
            logger.exception('api_fetch_news background fetch failed')

    threading.Thread(target=_run_fetch, daemon=True).start()


# 基础路由 - 页面路由
@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/analysis')
def analysis():
    """股票分析页面"""
    return render_template('analysis.html')


@app.route('/predict')
def predict():
    """股票预测页面"""
    return render_template('predict.html')


def _is_important_news(item: Dict[str, Any]) -> bool:
    title = str(item.get('title') or '')
    content = str(item.get('content') or '')
    text = (title + ' ' + content)
    keywords = ['重要', '重磅', '突发', '紧急', '警报', '大消息', '利好', '利空']
    return any(k in text for k in keywords)


def _normalize_hotspot_url(url: Any) -> str:
    u = '' if url is None else str(url)
    u = u.strip()
    if not u:
        return ''
    if u.startswith('//'):
        return 'https:' + u
    if u.startswith('/'):
        return 'https://tophub.today' + u
    if (not u.startswith('http://')) and (not u.startswith('https://')):
        if re.match(r'^[A-Za-z0-9.-]+\.[A-Za-z]{2,}(/|$)', u):
            return 'https://' + u
    return u


def _is_hotspot_image_url(url: str) -> bool:
    u = (url or '').strip().lower()
    if not u:
        return False
    u = re.split(r'[?#]', u)[0]
    return u.endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.svg'))


def _sanitize_hotspot_title(title: Any) -> str:
    t = '' if title is None else str(title)
    t = t.strip()
    if not t:
        return ''

    t = re.sub(r'^\s*!\[', '', t)
    t = re.sub(r'\]\s*$', '', t)
    t = re.sub(r'^\s*Image\s*\d+\s*[:：]\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'^\s*Image\s*\d+\s*$', '', t, flags=re.IGNORECASE)
    t = t.strip(' \t\r\n-—–|:：')
    return t.strip()


def _clean_hotspot_items(items: Any, limit: int) -> list[Dict[str, str]]:
    out: list[Dict[str, str]] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        title = _sanitize_hotspot_title(it.get('title'))
        link = _normalize_hotspot_url(it.get('url'))
        extra = it.get('extra')
        extra_s = '' if extra is None else str(extra).strip()
        if not title or not link:
            continue
        if _is_hotspot_image_url(link):
            continue
        out.append({'title': title, 'url': link, 'extra': extra_s})
        if len(out) >= int(limit):
            break
    return out


def _fetch_tophubdata_hotspots(limit: int) -> Dict[str, Any]:
    access_key = (os.getenv('TOPHUBDATA_ACCESS_KEY') or '').strip()
    if not access_key:
        return {'items': [], 'source': ''}

    url = 'https://api.tophubdata.com/nodes/1VdJkxkeLQ'
    req = urllib.request.Request(
        url,
        headers={
            'Authorization': access_key,
            'Accept': 'application/json,text/plain,*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'identity',
            'User-Agent': 'Mozilla/5.0',
        },
        method='GET',
    )

    try:
        raw = _urlopen_read_text(req, timeout=8)
        payload = json.loads(raw) if raw else {}

        data: Any = (payload.get('data') if isinstance(payload, dict) else None)
        items: Any = []
        if isinstance(data, dict):
            items = data.get('items') or data.get('data') or []
        elif isinstance(data, list):
            items = data
        else:
            items = payload.get('items') if isinstance(payload, dict) else []

        out = []
        for it in (items or []):
            if not isinstance(it, dict):
                continue
            title = _sanitize_hotspot_title(it.get('title'))
            link = _normalize_hotspot_url(it.get('url'))

            extra = it.get('extra')
            extra_s = '' if extra is None else str(extra).strip()
            if not title or not link:
                continue
            if _is_hotspot_image_url(link):
                continue
            out.append({'title': title, 'url': link, 'extra': extra_s})
            if len(out) >= limit:
                break

        return {'items': out, 'source': 'tophubdata'}
    except Exception as e:
        logger.warning('fetch tophubdata hotspots failed: %s', e)
        return {'items': [], 'source': ''}


def _fetch_tophub_today_hotspots(limit: int) -> Dict[str, Any]:
    try:
        urls = [
            'https://r.jina.ai/https://tophub.today/n/1VdJkxkeLQ',
            'https://r.jina.ai/http://tophub.today/n/1VdJkxkeLQ',
            'https://r.jina.ai/https://www.tophub.today/n/1VdJkxkeLQ',
            'https://r.jina.ai/http://www.tophub.today/n/1VdJkxkeLQ',
            'https://tophub.today/n/1VdJkxkeLQ',
        ]

        raw = ''
        last_err: str = ''
        for url in urls:
            req = urllib.request.Request(
                url,
                headers={
                    'Accept': 'text/plain,text/markdown,*/*',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'Accept-Encoding': 'identity',
                    'User-Agent': 'Mozilla/5.0',
                    'Referer': 'https://tophub.today/',
                },
                method='GET',
            )

            try:
                raw = _urlopen_read_text(req, timeout=10)
                if raw:
                    break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.warning('fetch tophub.today via jina failed url=%s err=%s', url, last_err)
                continue

        if not raw:
            if last_err:
                logger.warning('fetch tophub.today all urls failed last_err=%s', last_err)
            return {'items': [], 'source': ''}

        out: list[Dict[str, str]] = []
        seen = set()
        
        ignore_titles = {
            '今日热榜', '首页', '晚报', '动态', '追踪', '榜中榜', '热文库', 
            '话题', '日历', '综合', '搜索', '登录', '注册', '发现', '推荐'
        }

        # Markdown links
        for m in re.finditer(r'\[(?P<title>[^\]]+?)\]\((?P<link>[^\)\s]+)\)', raw):
            title = _sanitize_hotspot_title(m.group('title'))
            link = _normalize_hotspot_url(m.group('link'))
            if not title or not link or _is_hotspot_image_url(link):
                continue
            if title in ignore_titles or len(title) < 3:
                continue
            key = (title, link)
            if key in seen:
                continue
            seen.add(key)
            out.append({'title': title, 'url': link, 'extra': ''})
            if len(out) >= limit:
                break

        # HTML anchors (title can contain nested tags)
        if len(out) < limit:
            for m2 in re.finditer(r'<a\s+[^>]*href="(?P<link>[^"]+)"[^>]*>(?P<title>.*?)</a>', raw, flags=re.IGNORECASE | re.DOTALL):
                link = _normalize_hotspot_url(m2.group('link'))
                title_html = m2.group('title')
                title_text = re.sub(r'<[^>]+>', ' ', title_html)
                title = _sanitize_hotspot_title(title_text)
                if not title or not link or _is_hotspot_image_url(link):
                    continue
                if title in ignore_titles or len(title) < 3:
                    continue
                key = (title, link)
                if key in seen:
                    continue
                seen.add(key)
                out.append({'title': title, 'url': link, 'extra': ''})
                if len(out) >= limit:
                    break

        if not out:
            sample = raw[:600].replace('\r', ' ').replace('\n', ' ')
            logger.warning('tophub.today parse yielded 0 items. raw_sample=%s', sample)

        return {'items': out, 'source': 'tophub_today'}
    except Exception as e:
        logger.warning('fetch tophub.today hotspots failed: %s', e)
        return {'items': [], 'source': ''}


@app.route('/api/latest_news')
def api_latest_news():
    try:
        days = int(request.args.get('days') or 1)
    except Exception:
        days = 1

    try:
        limit = int(request.args.get('limit') or 50)
    except Exception:
        limit = 50
    try:
        important = int(request.args.get('important') or 0)
    except Exception:
        important = 0
    try:
        fast = int(request.args.get('fast') or 0)
    except Exception:
        fast = 0

    days = max(1, min(7, days))
    limit = max(1, min(500, limit))

    try:
        items = news_fetcher.get_latest_news(days=days, limit=limit)
        if not items:
            if fast:
                _trigger_news_fetch_async()
            else:
                try:
                    news_fetcher.fetch_and_save()
                except Exception:
                    pass
                items = news_fetcher.get_latest_news(days=days, limit=limit)

        if important:
            items = [x for x in items if _is_important_news(x)]

        return jsonify({'success': True, 'news': items})
    except Exception as e:
        logger.exception('api_latest_news failed')
        return jsonify({'success': False, 'error': str(e), 'news': []}), 500


@app.route('/api/fetch_news', methods=['POST'])
def api_fetch_news():
    try:
        _trigger_news_fetch_async()
        return jsonify({'success': True})
    except Exception as e:
        logger.exception('api_fetch_news failed')
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/hotspots')
def api_hotspots():
    try:
        limit = int(request.args.get('limit') or 10)
    except Exception:
        limit = 10
    limit = max(1, min(30, limit))

    now = time.time()
    with _hotspot_lock:
        cache_data = _load_hotspot_cache()
        cached_ts = float(cache_data.get('ts') or 0.0)
        cached_fail_ts = float(cache_data.get('fail_ts') or 0.0)
        cached_items = cache_data.get('items') or []
        cached_source = cache_data.get('source') or ''

        cleaned_cached = _clean_hotspot_items(cached_items, limit)
        if cleaned_cached and cleaned_cached != cached_items:
            cache_data['items'] = cleaned_cached
            _save_hotspot_cache(cache_data)
            cached_items = cleaned_cached

        if isinstance(cached_items, list) and len(cached_items) > 0 and (now - cached_ts < 300):
            return jsonify({
                'success': True,
                'items': cached_items,
                'source': cached_source,
            })

        # Avoid hammering upstream on repeated failures
        if (not cached_items) and (now - cached_fail_ts < 20):
            return jsonify({
                'success': True,
                'items': [],
                'source': cached_source,
            })

    data = _fetch_tophubdata_hotspots(limit)
    if not (data.get('items') or []):
        data = _fetch_tophub_today_hotspots(limit)

    items = data.get('items') or []
    source = data.get('source') or ''

    items = _clean_hotspot_items(items, limit)

    with _hotspot_lock:
        cache_data = _load_hotspot_cache()
        cached_items = cache_data.get('items') or []
        cached_source = cache_data.get('source') or ''

        # Only cache non-empty results; keep last non-empty as fallback.
        if items:
            cache_data['ts'] = now
            cache_data['items'] = items
            cache_data['source'] = source
            cache_data['fail_ts'] = 0.0
            _save_hotspot_cache(cache_data)
        else:
            cache_data['fail_ts'] = now
            _save_hotspot_cache(cache_data)
            if isinstance(cached_items, list) and cached_items:
                items = cached_items
                source = cached_source

    return jsonify({
        'success': True,
        'items': items,
        'source': source,
    })


def _period_to_days(period: str) -> int:
    period_map = {
        '1m': 31,
        '3m': 93,
        '6m': 186,
        '1y': 366,
    }
    return period_map.get(period or '', 366)


def _fmt_yyyymmdd(d: datetime) -> str:
    return d.strftime('%Y%m%d')


def _get_history_df(stock_code: str, market_type: str, period: str) -> pd.DataFrame:
    market_type = (market_type or 'A').strip().upper()
    if market_type not in {'A', 'HK', 'US'}:
        raise ValueError('不支持的市场类型')

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=_period_to_days(period))

    provider = get_data_provider()
    df = provider.get_stock_history(
        code=stock_code,
        start_date=_fmt_yyyymmdd(start_dt),
        end_date=_fmt_yyyymmdd(end_dt),
        adjust='qfq',
        market_type=market_type,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize columns
    if 'date' not in df.columns:
        # Best effort
        if '日期' in df.columns:
            df = df.rename(columns={'日期': 'date'})
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Normalize date type
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'open', 'high', 'low', 'close']).sort_values('date')
    return df


def _fetch_last_close_n(stock_code: str, market_type: str, days: int) -> pd.DataFrame:
    try:
        n = int(days)
    except Exception:
        n = 0
    n = max(1, n)

    # Fetch a sufficiently long window to ensure we have at least n trading days.
    # Using 1y is simple and stable across different data providers.
    df = _get_history_df(stock_code, market_type, period='1y')
    if df is None or df.empty:
        return pd.DataFrame()

    if 'close' not in df.columns or 'date' not in df.columns:
        return pd.DataFrame()

    out = df[['date', 'close']].copy()
    out['close'] = pd.to_numeric(out['close'], errors='coerce')
    out['date'] = pd.to_datetime(out['date'], errors='coerce')
    out = out.dropna(subset=['date', 'close']).sort_values('date')
    if len(out) > n:
        out = out.tail(n)
    out = out.reset_index(drop=True)
    return out


def _minmax_scale_1d(x: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {'scaled': np.asarray([], dtype=np.float32), 'x_min': 0.0, 'x_max': 0.0}
    x_min = float(np.nanmin(arr))
    x_max = float(np.nanmax(arr))
    denom = x_max - x_min
    if not np.isfinite(denom) or denom == 0.0:
        scaled = np.zeros_like(arr, dtype=np.float32)
    else:
        scaled = ((arr - x_min) / denom).astype(np.float32)
    return {'scaled': scaled, 'x_min': x_min, 'x_max': x_max}


def _minmax_inv(y: float, x_min: float, x_max: float) -> float:
    try:
        y0 = float(y)
        mn = float(x_min)
        mx = float(x_max)
    except Exception:
        return float('nan')
    denom = mx - mn
    if not np.isfinite(denom) or denom == 0.0:
        return mn
    return mn + y0 * denom


def _next_trading_days(last_dt: datetime, n: int) -> list[str]:
    try:
        k = int(n)
    except Exception:
        k = 0
    k = max(0, k)

    cur = last_dt
    out: list[str] = []
    while len(out) < k:
        cur = cur + timedelta(days=1)
        # 0=Mon ... 5=Sat 6=Sun
        if cur.weekday() >= 5:
            continue
        out.append(cur.strftime('%Y-%m-%d'))
    return out


def _normalize_predict_code(stock_code: str, market_type: str) -> str:
    code = (stock_code or '').strip()
    mt = (market_type or 'A').strip().upper()
    if mt == 'HK':
        digits = ''.join([ch for ch in code if ch.isdigit()])
        return digits.zfill(5) if digits else code
    if mt == 'US':
        return code.upper()
    return code


def _forecasting_root() -> Path:
    return Path(__file__).resolve().parents[2] / 'forecasting'


def _model_paths(model_type: str = 'gru') -> Dict[str, Path]:
    root = _forecasting_root() / 'models' / model_type.lower()
    ckpt = root / 'checkpoints'
    return {
        'root': root,
        'meta': root / 'meta.json',
        'weights': ckpt / 'latest.weights.h5',
    }

def _load_model_meta(model_type: str = 'gru') -> Dict[str, Any]:
    p = _model_paths(model_type)['meta']
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _build_gru_regression_model_named(
    lookback: int,
    units: int,
    layers: int,
    dropout: float,
    learning_rate: float,
    gru_layer_names: list[str],
    dense_layer_name: str,
):
    import tensorflow as tf

    if int(lookback) <= 0:
        raise ValueError('lookback must be > 0')
    if int(layers) <= 0:
        raise ValueError('layers must be > 0')

    model = tf.keras.Sequential()
    for i in range(int(layers)):
        return_sequences = i < int(layers) - 1
        name = gru_layer_names[i] if i < len(gru_layer_names) else None
        if i == 0:
            model.add(
                tf.keras.layers.GRU(
                    units=int(units),
                    return_sequences=return_sequences,
                    input_shape=(int(lookback), 1),
                    name=name,
                )
            )
        else:
            model.add(
                tf.keras.layers.GRU(
                    units=int(units),
                    return_sequences=return_sequences,
                    name=name,
                )
            )
        if dropout and float(dropout) > 0:
            model.add(tf.keras.layers.Dropout(float(dropout)))

    model.add(tf.keras.layers.Dense(1, activation='linear', name=(dense_layer_name or None)))
    opt = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))
    model.compile(optimizer=opt, loss='mse')
    return model


def _read_h5_layer_names(weights_path: Path) -> list[str]:
    try:
        import h5py  # type: ignore

        with h5py.File(str(weights_path), 'r') as f:
            try:
                raw_attr = f.attrs.get('layer_names')
                if raw_attr is not None:
                    out0: list[str] = []
                    for x in raw_attr:
                        if isinstance(x, bytes):
                            out0.append(x.decode('utf-8', errors='ignore'))
                        else:
                            out0.append(str(x))
                    out0 = [s for s in out0 if s]
                    if out0:
                        return out0
            except Exception:
                pass

            if 'layer_names' in f:
                raw = f['layer_names'][()]
                out: list[str] = []
                for x in raw:
                    if isinstance(x, bytes):
                        out.append(x.decode('utf-8', errors='ignore'))
                    else:
                        out.append(str(x))
                return [s for s in out if s]

            if 'model_weights' in f and 'layer_names' in f['model_weights']:
                raw = f['model_weights']['layer_names'][()]
                out2: list[str] = []
                for x in raw:
                    if isinstance(x, bytes):
                        out2.append(x.decode('utf-8', errors='ignore'))
                    else:
                        out2.append(str(x))
                return [s for s in out2 if s]

            if 'model_weights' in f:
                try:
                    mw = f['model_weights']
                    raw_attr2 = mw.attrs.get('layer_names')
                    if raw_attr2 is not None:
                        out3: list[str] = []
                        for x in raw_attr2:
                            if isinstance(x, bytes):
                                out3.append(x.decode('utf-8', errors='ignore'))
                            else:
                                out3.append(str(x))
                        out3 = [s for s in out3 if s]
                        if out3:
                            return out3
                except Exception:
                    pass

                try:
                    mw_keys = [str(k) for k in f['model_weights'].keys()]
                    return [s for s in mw_keys if s]
                except Exception:
                    pass
    except Exception:
        logger.warning('read h5 layer_names failed (missing h5py or invalid file): %s', weights_path)
        return []
    return []


def _manual_load_gru_weights_from_h5(model: Any, weights_path: Path) -> bool:
    try:
        import h5py  # type: ignore
        import tensorflow as tf
    except Exception:
        logger.warning('manual gru load unavailable: missing h5py/tensorflow (weights=%s)', weights_path)
        return False

    logger.info('manual gru load: attempting to assign weights from h5 (weights=%s)', weights_path)

    def _iter_datasets(g, prefix: str):
        out = []
        for k, v in g.items():
            p = f'{prefix}/{k}' if prefix else str(k)
            try:
                if hasattr(v, 'items'):
                    out.extend(_iter_datasets(v, p))
                else:
                    out.append((p, v[()]))
            except Exception:
                continue
        return out

    try:
        with h5py.File(str(weights_path), 'r') as f:
            try:
                logger.info('manual gru load: h5 top_keys=%s', list(f.keys())[:30])
            except Exception:
                pass
            pairs = _iter_datasets(f, '')
    except Exception:
        return False

    # More general: infer weights from dataset paths rather than top-level group names.
    # Supports multiple TF/Keras H5 layouts (some omit layer_names / use different nesting).
    gru_blocks: dict[str, dict[str, Any]] = {}
    dense_blocks: dict[str, dict[str, Any]] = {}
    for path, arr in pairs:
        p = '/' + str(path).lstrip('/')

        mg = re.match(
            r'^(.*/gru_cell[^/]*)/(kernel(?::0)?|recurrent_kernel(?::0)?|bias(?::0)?)$',
            p,
        )
        if mg:
            prefix = mg.group(1)
            kind = mg.group(2).split(':', 1)[0]
            d = gru_blocks.setdefault(prefix, {})
            d[kind] = arr
            continue

        md = re.match(r'^(.*/dense[^/]*)/(kernel(?::0)?|bias(?::0)?)$', p)
        if md:
            prefix = md.group(1)
            kind = md.group(2).split(':', 1)[0]
            d = dense_blocks.setdefault(prefix, {})
            d[kind] = arr

    if not gru_blocks and not dense_blocks:
        try:
            logger.warning('manual gru load: no matching datasets; total_datasets=%s', len(pairs))
        except Exception:
            pass
        try:
            sample = []
            for path, _arr in pairs:
                s = str(path)
                sl = s.lower()
                if any(k in sl for k in ('gru', 'dense', 'kernel', 'recurrent', 'bias', 'cell', 'weight')):
                    sample.append(s)
                if len(sample) >= 60:
                    break
            logger.warning('manual gru load: sample_dataset_paths=%s', sample)
        except Exception:
            pass

        # If still empty, try TF checkpoint-style H5 layout:
        # _layer_checkpoint_dependencies\\gru\\cell/vars/0..2
        # _layer_checkpoint_dependencies\\dense/vars/0..1
        gru_ckpt: dict[str, dict[int, Any]] = {}
        dense_ckpt: dict[int, Any] = {}
        for path, arr in pairs:
            s = str(path)

            md0 = re.match(r'^_layer_checkpoint_dependencies\\dense/vars/(\d+)$', s)
            if md0:
                try:
                    dense_ckpt[int(md0.group(1))] = arr
                except Exception:
                    pass
                continue

            mg0 = re.match(r'^_layer_checkpoint_dependencies\\(gru(?:_\d+)?)\\cell/vars/(\d+)$', s)
            if mg0:
                layer_name = mg0.group(1)
                try:
                    idx = int(mg0.group(2))
                except Exception:
                    continue
                d = gru_ckpt.setdefault(layer_name, {})
                d[idx] = arr

        for layer_name, d in gru_ckpt.items():
            if all(i in d for i in (0, 1, 2)):
                gru_blocks[f'ckpt:{layer_name}'] = {
                    'kernel': d[0],
                    'recurrent_kernel': d[1],
                    'bias': d[2],
                }
        if all(i in dense_ckpt for i in (0, 1)):
            dense_blocks['ckpt:dense'] = {
                'kernel': dense_ckpt[0],
                'bias': dense_ckpt[1],
            }

    # Build candidate items from discovered blocks (traditional Keras-ish layouts)
    def _extract_layer_name(block_prefix: str) -> str:
        s0 = str(block_prefix)
        if s0.startswith('ckpt:'):
            return s0.split(':', 1)[1]
        parts = [x for x in str(block_prefix).split('/') if x]
        if not parts:
            return str(block_prefix)
        # Heuristic: often .../<layer_name>/gru_cell... or .../<layer_name>/dense...
        for i in range(len(parts) - 1):
            if parts[i + 1].startswith('gru_cell'):
                return parts[i]
        return parts[-1]

    gru_items: list[tuple[str, str, dict[str, Any]]] = [
        (k, _extract_layer_name(k), v)
        for k, v in gru_blocks.items()
        if all(x in v for x in ('kernel', 'recurrent_kernel', 'bias'))
    ]
    dense_items: list[tuple[str, str, dict[str, Any]]] = [
        (k, _extract_layer_name(k), v)
        for k, v in dense_blocks.items()
        if all(x in v for x in ('kernel', 'bias'))
    ]

    def _name_key(prefix: str, name: str):
        if name == prefix:
            return (0, 0)
        m = re.match(rf'^{re.escape(prefix)}_(\d+)$', name)
        if m:
            return (0, int(m.group(1)) + 1)
        return (1, name)

    gru_items.sort(key=lambda t: _name_key('gru', t[1]))
    dense_items.sort(key=lambda t: _name_key('dense', t[1]))

    try:
        logger.info(
            'manual gru load: discovered blocks gru=%s dense=%s',
            [x[1] for x in gru_items[:10]],
            [x[1] for x in dense_items[:10]],
        )
    except Exception:
        pass

    try:
        gru_layers = [l for l in getattr(model, 'layers', []) if isinstance(l, tf.keras.layers.GRU)]
        dense_layers = [l for l in getattr(model, 'layers', []) if isinstance(l, tf.keras.layers.Dense)]
        if not gru_layers or not dense_layers:
            return False
        if len(gru_items) < len(gru_layers) or not dense_items:
            logger.warning(
                'manual gru load: insufficient blocks in h5 (gru_blocks=%s dense_blocks=%s)',
                [x[1] for x in gru_items],
                [x[1] for x in dense_items],
            )
            return False

        for i, layer in enumerate(gru_layers):
            _, lname, v = gru_items[i]
            layer.cell.kernel.assign(np.asarray(v['kernel']))
            layer.cell.recurrent_kernel.assign(np.asarray(v['recurrent_kernel']))
            layer.cell.bias.assign(np.asarray(v['bias']))

        _, dense_lname, d0 = dense_items[0]
        dense_layers[-1].kernel.assign(np.asarray(d0['kernel']))
        dense_layers[-1].bias.assign(np.asarray(d0['bias']))

        logger.info(
            'manual gru load: assigned weights from h5 gru_layers=%s dense_layer=%s',
            [x[1] for x in gru_items[: len(gru_layers)]],
            dense_lname,
        )
        return True
    except Exception as e:
        logger.warning('manual gru load failed: %s', e)
        return False


def _manual_load_lstm_weights_from_h5(model: Any, weights_path: Path) -> bool:
    """手动从 H5 文件加载 LSTM 权重（备用方案）"""
    try:
        import h5py  # type: ignore
        import tensorflow as tf
    except Exception:
        logger.warning('manual lstm load unavailable: missing h5py/tensorflow (weights=%s)', weights_path)
        return False

    logger.info('manual lstm load: attempting to assign weights from h5 (weights=%s)', weights_path)

    def _iter_datasets(g, prefix: str):
        out = []
        for k, v in g.items():
            p = f'{prefix}/{k}' if prefix else str(k)
            try:
                if hasattr(v, 'items'):
                    out.extend(_iter_datasets(v, p))
                else:
                    out.append((p, v[()]))
            except Exception:
                continue
        return out

    try:
        with h5py.File(str(weights_path), 'r') as f:
            try:
                logger.info('manual lstm load: h5 top_keys=%s', list(f.keys())[:30])
            except Exception:
                pass
            pairs = _iter_datasets(f, '')
    except Exception:
        return False

    # 解析 LSTM 和 Dense 权重
    lstm_blocks: dict[str, dict[str, Any]] = {}
    dense_blocks: dict[str, dict[str, Any]] = {}
    for path, arr in pairs:
        p = '/' + str(path).lstrip('/')

        # 匹配 LSTM cell 权重
        ml = re.match(
            r'^(.*/lstm_cell[^/]*)/(kernel(?::0)?|recurrent_kernel(?::0)?|bias(?::0)?)$',
            p,
        )
        if ml:
            prefix = ml.group(1)
            kind = ml.group(2).split(':', 1)[0]
            d = lstm_blocks.setdefault(prefix, {})
            d[kind] = arr
            continue

        # 匹配 Dense 权重
        md = re.match(r'^(.*/dense[^/]*)/(kernel(?::0)?|bias(?::0)?)$', p)
        if md:
            prefix = md.group(1)
            kind = md.group(2).split(':', 1)[0]
            d = dense_blocks.setdefault(prefix, {})
            d[kind] = arr

    # 如果没找到，尝试 TF checkpoint 风格的布局
    if not lstm_blocks and not dense_blocks:
        lstm_ckpt: dict[str, dict[int, Any]] = {}
        dense_ckpt: dict[int, Any] = {}
        for path, arr in pairs:
            s = str(path)

            md0 = re.match(r'^_layer_checkpoint_dependencies\\dense/vars/(\d+)$', s)
            if md0:
                try:
                    dense_ckpt[int(md0.group(1))] = arr
                except Exception:
                    pass
                continue

            ml0 = re.match(r'^_layer_checkpoint_dependencies\\(lstm(?:_\d+)?)\\cell/vars/(\d+)$', s)
            if ml0:
                layer_name = ml0.group(1)
                try:
                    idx = int(ml0.group(2))
                except Exception:
                    continue
                d = lstm_ckpt.setdefault(layer_name, {})
                d[idx] = arr

        for layer_name, d in lstm_ckpt.items():
            if all(i in d for i in (0, 1, 2)):
                lstm_blocks[f'ckpt:{layer_name}'] = {
                    'kernel': d[0],
                    'recurrent_kernel': d[1],
                    'bias': d[2],
                }
        if all(i in dense_ckpt for i in (0, 1)):
            dense_blocks['ckpt:dense'] = {
                'kernel': dense_ckpt[0],
                'bias': dense_ckpt[1],
            }

    if not lstm_blocks and not dense_blocks:
        logger.warning('manual lstm load: no matching datasets found')
        return False

    def _extract_layer_name(block_prefix: str) -> str:
        s0 = str(block_prefix)
        if s0.startswith('ckpt:'):
            return s0.split(':', 1)[1]
        parts = [x for x in str(block_prefix).split('/') if x]
        if not parts:
            return str(block_prefix)
        for i in range(len(parts) - 1):
            if parts[i + 1].startswith('lstm_cell'):
                return parts[i]
        return parts[-1]

    lstm_items: list[tuple[str, str, dict[str, Any]]] = [
        (k, _extract_layer_name(k), v)
        for k, v in lstm_blocks.items()
        if all(x in v for x in ('kernel', 'recurrent_kernel', 'bias'))
    ]
    dense_items: list[tuple[str, str, dict[str, Any]]] = [
        (k, _extract_layer_name(k), v)
        for k, v in dense_blocks.items()
        if all(x in v for x in ('kernel', 'bias'))
    ]

    def _name_key(prefix: str, name: str):
        if name == prefix:
            return (0, 0)
        m = re.match(rf'^{re.escape(prefix)}_(\d+)$', name)
        if m:
            return (0, int(m.group(1)) + 1)
        return (1, name)

    lstm_items.sort(key=lambda t: _name_key('lstm', t[1]))
    dense_items.sort(key=lambda t: _name_key('dense', t[1]))

    try:
        lstm_layers = [l for l in getattr(model, 'layers', []) if isinstance(l, tf.keras.layers.LSTM)]
        dense_layers = [l for l in getattr(model, 'layers', []) if isinstance(l, tf.keras.layers.Dense)]
        if not lstm_layers or not dense_layers:
            return False
        if len(lstm_items) < len(lstm_layers) or not dense_items:
            logger.warning(
                'manual lstm load: insufficient blocks in h5 (lstm_blocks=%s dense_blocks=%s)',
                [x[1] for x in lstm_items],
                [x[1] for x in dense_items],
            )
            return False

        for i, layer in enumerate(lstm_layers):
            _, lname, v = lstm_items[i]
            layer.cell.kernel.assign(np.asarray(v['kernel']))
            layer.cell.recurrent_kernel.assign(np.asarray(v['recurrent_kernel']))
            layer.cell.bias.assign(np.asarray(v['bias']))

        _, dense_lname, d0 = dense_items[0]
        dense_layers[-1].kernel.assign(np.asarray(d0['kernel']))
        dense_layers[-1].bias.assign(np.asarray(d0['bias']))

        logger.info(
            'manual lstm load: assigned weights from h5 lstm_layers=%s dense_layer=%s',
            [x[1] for x in lstm_items[: len(lstm_layers)]],
            dense_lname,
        )
        return True
    except Exception as e:
        logger.warning('manual lstm load failed: %s', e)
        return False


def _load_prediction_model(model_type: str = 'gru') -> Any:
    model_type = model_type.lower()
    if model_type not in {'gru', 'lstm'}:
        raise ValueError(f'Unsupported model type: {model_type}')

    cache = _gru_model_cache if model_type == 'gru' else _lstm_model_cache
    paths = _model_paths(model_type)
    weights = paths['weights']
    if not weights.exists():
        raise FileNotFoundError(f'未找到 {model_type.upper()} 模型权重文件，请先开始训练或上传权重。')

    try:
        size = int(weights.stat().st_size)
    except Exception:
        size = 0

    if size <= 0:
        raise ValueError('GRU模型权重文件为空，请先训练模型或重新上传权重文件')

    head = b''
    try:
        with open(weights, 'rb') as f:
            head = f.read(16)
    except Exception:
        head = b''
    is_hdf5 = (len(head) >= 8 and head[:8] == b'\x89HDF\r\n\x1a\n')

    mtime = float(weights.stat().st_mtime)
    if cache.get('model') is not None and float(cache.get('mtime') or 0.0) >= mtime:
        return cache.get('model')

    meta = _load_model_meta(model_type)
    lookback = int(meta.get('lookback') or 30)

    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass

    model_cfg = meta.get('model') or {}
    units = int(model_cfg.get('units') or (50 if model_type == 'gru' else 128))
    layers = int(model_cfg.get('layers') or 3)
    dropout = float(model_cfg.get('dropout') or (0.2 if model_type != 'lstm' else 0.5))
    learning_rate = float(model_cfg.get('learning_rate') or 0.001)
    feature_count = int(meta.get('feature_count') or 1)
    feature_cols = meta.get('feature_cols') or ['close']

    model = None
    if model_type == 'gru':
        from forecasting.models.gru.model import build_gru_regression_model
        model = build_gru_regression_model(lookback=lookback, feature_count=feature_count, units=units, layers=layers, dropout=dropout, learning_rate=learning_rate)
    elif model_type == 'lstm':
        from forecasting.models.lstm.model import build_lstm_regression_model
        model = build_lstm_regression_model(lookback=lookback, feature_count=feature_count, units=units, layers=layers, dropout=dropout, learning_rate=learning_rate)

    if model is None:
        raise ValueError(f"Failed to build model for type: {model_type}")

    try:
        model.build((None, int(lookback), feature_count))
    except Exception:
        pass

    try:
        model.load_weights(str(weights))
    except Exception as e:
        if model_type == 'gru':
             if _manual_load_gru_weights_from_h5(model, weights):
                cache['model'] = model
                cache['meta'] = meta
                cache['mtime'] = mtime
                return model
        elif model_type == 'lstm':
            if _manual_load_lstm_weights_from_h5(model, weights):
                cache['model'] = model
                cache['meta'] = meta
                cache['mtime'] = mtime
                return model
        raise ValueError(f'{model_type.upper()} 模型权重加载失败: {e}')

    cache['model'] = model
    cache['meta'] = meta
    cache['mtime'] = mtime
    return model


@app.route('/api/predict_stock')
def api_predict_stock():
    stock_code = (request.args.get('stock_code') or '').strip()
    market_type = (request.args.get('market_type') or 'A').strip().upper()
    model_type = (request.args.get('model_type') or 'gru').strip().lower()
    try:
        predict_days = int(request.args.get('days') or 10)
    except Exception:
        predict_days = 10

    if not stock_code:
        return jsonify({'success': False, 'error': 'stock_code不能为空'}), 400
    if model_type not in {'gru', 'lstm'}:
        return jsonify({'success': False, 'error': '不支持的模型类型'}), 400

    code = _normalize_predict_code(stock_code, market_type)

    try:
        model = _load_prediction_model(model_type)
        cache = _gru_model_cache if model_type == 'gru' else _lstm_model_cache
        meta = dict(cache.get('meta') or {})
        lookback = int(meta.get('lookback') or 30)
        total_days = int(meta.get('total_days') or 50)
        feature_cols = meta.get('feature_cols') or ['close']

        # 获取更长的数据以支持特征工程（需要足够历史计算指标）
        fetch_days = max(total_days, 60) + 30  # 额外30天保证指标计算完整
        end = datetime.now().date()
        start = end - timedelta(days=fetch_days * 2)

        provider = get_data_provider()
        df_raw = provider.get_stock_history(code=code, start_date=start.strftime('%Y%m%d'),
                                            end_date=end.strftime('%Y%m%d'), adjust='qfq', market_type=market_type)
        
        if df_raw is None or df_raw.empty:
            return jsonify({'success': False, 'error': f'无法获取 {code} 的历史数据'}), 400

        # 特征工程
        from forecasting.core.feature_engine import get_feature_engine
        from forecasting.utils.indicators import add_indicators
        engine = get_feature_engine()
        df = engine.prepare_features(df_raw, include_sentiment=True)
        
        # 截取最后的窗口（保留足够历史用于指标计算）
        df = df.tail(total_days).reset_index(drop=True)
        if len(df) < lookback:
            return jsonify({'success': False, 'error': f'特征处理后数据量不足 ({len(df)}/{lookback})'}), 400

        # 记录原始价格数据用于后续重新计算指标
        # 需要保留足够长的历史（至少30天）来计算ma20等指标
        history_keep_days = max(30, lookback)
        df_history = df.tail(history_keep_days).copy()
        
        # 初始归一化
        def _normalize_df(df_in: pd.DataFrame, cols: list) -> tuple:
            """归一化指定列，返回归一化后的df和scaling元数据"""
            df_out = df_in.copy()
            scaling = {}
            for col in cols:
                if col not in df_out.columns:
                    continue
                vals = df_out[col].values
                v_min, v_max = float(np.min(vals)), float(np.max(vals))
                denom = v_max - v_min
                if denom < 1e-12:
                    df_out[col] = 0.0
                else:
                    df_out[col] = (vals - v_min) / denom
                scaling[col] = (v_min, v_max)
            return df_out, scaling
        
        df_scaled, scaling_meta = _normalize_df(df_history, feature_cols)
        x_min, x_max = scaling_meta.get('close', (0.0, 1.0))

        # 预测循环：每步重新计算特征
        preds_real: list[float] = []
        df_predict_base = df_history.copy()  # 用于累积预测数据
        
        for step_idx in range(int(predict_days)):
            # 重新计算技术指标（基于当前累积的历史数据）
            cols_to_use = ['date', 'open', 'high', 'low', 'close', 'volume']
            if 'amount' in df_predict_base.columns:
                cols_to_use.append('amount')
            df_with_indicators = add_indicators(df_predict_base[cols_to_use].copy())
            
            # 添加情感得分（复用最后一天的值，因为未来没有新闻）
            if 'sentiment' in feature_cols:
                last_sentiment = df_predict_base['sentiment'].iloc[-1] if 'sentiment' in df_predict_base.columns else 0.0
                df_with_indicators['sentiment'] = last_sentiment
            
            # 处理无穷值和空值
            df_with_indicators = df_with_indicators.replace([np.inf, -np.inf], np.nan)
            df_with_indicators = df_with_indicators.ffill().bfill()
            
            # 确保所有 feature_cols 都存在，缺失的用 0 填充
            for col in feature_cols:
                if col not in df_with_indicators.columns:
                    df_with_indicators[col] = 0.0
            
            # 重新归一化（使用当前数据的min/max）
            df_scaled_step, scaling_step = _normalize_df(df_with_indicators, feature_cols)
            
            # 获取预测窗口
            features_scaled = df_scaled_step[feature_cols].values.astype(np.float32)
            if len(features_scaled) < lookback:
                break
            window = features_scaled[-lookback:].copy()
            
            # 预测
            x = window.reshape((1, lookback, len(feature_cols))).astype(np.float32)
            y_pred = float(model.predict(x, verbose=0)[0][0])
            
            # 反归一化（使用当前步骤的close min/max）
            step_close_min, step_close_max = scaling_step.get('close', (0.0, 1.0))
            pred_real_val = y_pred * (step_close_max - step_close_min) + step_close_min
            preds_real.append(pred_real_val)
            
            # 构造新的一行数据（用于下一次迭代）
            # 对于无法预测的字段，使用合理假设
            last_row = df_predict_base.iloc[-1].copy()
            new_date = pd.Timestamp(last_row['date']) + timedelta(days=1)
            
            new_row = {
                'date': new_date,
                'open': pred_real_val,  # 假设开盘价≈预测收盘价
                'high': pred_real_val * 1.005,  # 假设高点略高
                'low': pred_real_val * 0.995,  # 假设低点略低
                'close': pred_real_val,
                'volume': last_row.get('volume', 0),  # 复用前一天成交量
                'amount': last_row.get('amount', 0),
                'sentiment': last_row.get('sentiment', 0),  # 复用前一天情感
            }
            
            # 添加到历史数据末尾
            df_predict_base = pd.concat([df_predict_base, pd.DataFrame([new_row])], ignore_index=True)
            
            # 保持历史长度不超过需要的大小（避免内存膨胀）
            max_history = history_keep_days + int(predict_days) + 10
            if len(df_predict_base) > max_history:
                df_predict_base = df_predict_base.tail(max_history).reset_index(drop=True)

        hist = []
        # 只返回最后 30 天的历史显示在图表上
        for _, r in df.tail(30).iterrows():
            hist.append({
                'date': pd.to_datetime(r['date']).strftime('%Y-%m-%d'),
                'close': float(r['close']),
            })

        last_dt = pd.to_datetime(df['date'].iloc[-1]).to_pydatetime()
        future_dates = _next_trading_days(last_dt, int(predict_days))
        forecast = []
        for i in range(int(predict_days)):
            forecast.append({
                'date': future_dates[i],
                'close': float(preds_real[i]) if i < len(preds_real) else 0.0,
            })

        return jsonify({
            'success': True,
            'stock_code': code,
            'market_type': market_type,
            'model_type': model_type,
            'lookback': int(lookback),
            'history': hist,
            'forecast': forecast,
            'boundary_date': hist[-1]['date'] if hist else '',
        })
    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        logger.exception('api_predict_stock failed')
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict_gru')
def api_predict_gru():
    # Legacy support
    return api_predict_stock()


def _compute_rsi(close: pd.Series, window: int = 14) -> float:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1]) if len(rsi) else float('nan')
    return 50.0 if np.isnan(val) else val


def _compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal
    return {
        'macd': float(macd.iloc[-1]) if len(macd) else 0.0,
        'macd_signal': float(macd_signal.iloc[-1]) if len(macd_signal) else 0.0,
        'macd_hist': float(hist.iloc[-1]) if len(hist) else 0.0,
    }


def _compute_volatility(close: pd.Series, window: int = 20) -> float:
    ret = close.pct_change()
    vol = ret.rolling(window).std() * np.sqrt(252)
    val = float(vol.iloc[-1]) if len(vol) else float('nan')
    return 0.0 if np.isnan(val) else val


def _score_and_reco(trend_score: float, indicators_score: float, sr_score: float, vv_score: float) -> Dict[str, Any]:
    total = (trend_score + indicators_score + sr_score + vv_score) / 4.0
    total = max(0.0, min(10.0, total))
    total_round = int(round(total))

    if total_round >= 8:
        action = '强烈建议买入'
    elif total_round >= 6:
        action = '建议买入'
    elif total_round >= 4:
        action = '观望'
    else:
        action = '谨慎/卖出'
    return {
        'total_score': total_round,
        'action': action,
    }


def _pick_first_non_empty(info: Dict[str, Any], keys) -> str:
    for k in keys:
        v = info.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() != 'none' and s.lower() != 'nan':
            return s
    return ''


@app.route('/api/stock_data')
def api_stock_data():
    stock_code = (request.args.get('stock_code') or '').strip()
    market_type = (request.args.get('market_type') or 'A').strip()
    period = (request.args.get('period') or '1y').strip()
    request_id = (request.args.get('request_id') or '').strip()

    if not stock_code:
        return jsonify({'error': 'stock_code不能为空'}), 400

    try:
        _progress_set(request_id, 5, 'stock_data_start')
        df = _get_history_df(stock_code, market_type, period)
        if df.empty:
            _progress_set(request_id, 100, 'done_no_data')
            return jsonify({'data': []})

        _progress_set(request_id, 40, 'stock_data_history_ok')

        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA60'] = df['close'].rolling(60).mean()

        _progress_set(request_id, 55, 'stock_data_indicators_ok')

        out = []
        for _, r in df.iterrows():
            out.append({
                'date': r['date'].strftime('%Y-%m-%d'),
                'open': float(r['open']),
                'high': float(r['high']),
                'low': float(r['low']),
                'close': float(r['close']),
                'volume': float(r['volume']) if 'volume' in df.columns and pd.notna(r.get('volume')) else None,
                'amount': float(r['amount']) if 'amount' in df.columns and pd.notna(r.get('amount')) else None,
                'MA5': float(r['MA5']) if pd.notna(r['MA5']) else None,
                'MA20': float(r['MA20']) if pd.notna(r['MA20']) else None,
                'MA60': float(r['MA60']) if pd.notna(r['MA60']) else None,
            })
        _progress_set(request_id, 60, 'stock_data_done')
        return jsonify({'data': out})
    except Exception as e:
        _progress_set(request_id, 100, 'error')
        logger.exception('api_stock_data failed')
        return jsonify({'error': str(e)}), 500


@app.route('/api/enhanced_analysis', methods=['POST'])
def api_enhanced_analysis():
    payload = request.get_json(silent=True) or {}
    stock_code = (payload.get('stock_code') or '').strip()
    market_type = (payload.get('market_type') or 'A').strip()
    period = (payload.get('period') or '1y').strip()
    request_id = (payload.get('request_id') or '').strip()

    if not stock_code:
        return jsonify({'error': 'stock_code不能为空'}), 400

    try:
        _progress_set(request_id, 65, 'analysis_start')
        df = _get_history_df(stock_code, market_type, period)
        if df.empty:
            _progress_set(request_id, 100, 'done_no_data')
            return jsonify({'error': '未找到股票数据'}), 404

        _progress_set(request_id, 72, 'analysis_history_ok')

        close = df['close']
        current_price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else current_price
        price_change_value = current_price - prev_close
        price_change = (price_change_value / prev_close) if prev_close else 0.0

        # Indicators
        rsi = _compute_rsi(close)
        macd = _compute_macd(close)
        volatility = _compute_volatility(close)
        ma5 = float(close.rolling(5).mean().iloc[-1]) if len(close) >= 5 else current_price
        ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else current_price
        ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else current_price

        _progress_set(request_id, 82, 'analysis_indicators_ok')

        # Trend
        if ma5 > ma20 > ma60:
            ma_trend = 'UP'
            ma_status = '多头排列'
            trend_score = 8.5
        elif ma5 < ma20 < ma60:
            ma_trend = 'DOWN'
            ma_status = '空头排列'
            trend_score = 2.5
        else:
            ma_trend = 'SIDE'
            ma_status = '震荡'
            trend_score = 5.0

        # Support/Resistance (simple)
        short_win = min(20, len(df))
        med_win = min(60, len(df))
        short_support = float(df['low'].tail(short_win).min())
        short_resistance = float(df['high'].tail(short_win).max())
        med_support = float(df['low'].tail(med_win).min())
        med_resistance = float(df['high'].tail(med_win).max())

        sr = {
            'support': {
                'short_term': [short_support],
                'medium_term': [med_support],
            },
            'resistance': {
                'short_term': [short_resistance],
                'medium_term': [med_resistance],
            }
        }

        # Indicator score
        indicators_score = 5.0
        if 40 <= rsi <= 60:
            indicators_score += 1.5
        elif rsi < 30:
            indicators_score += 2.0
        elif rsi > 70:
            indicators_score -= 2.0
        if macd['macd'] > macd['macd_signal']:
            indicators_score += 1.5
        else:
            indicators_score -= 1.0
        indicators_score = max(0.0, min(10.0, indicators_score))

        # Support/Resistance score
        dist_to_support = (current_price - short_support) / current_price if current_price else 0.0
        dist_to_res = (short_resistance - current_price) / current_price if current_price else 0.0
        sr_score = 6.0
        if dist_to_support < 0.03:
            sr_score += 2.0
        if dist_to_res < 0.03:
            sr_score -= 2.0
        sr_score = max(0.0, min(10.0, sr_score))

        # Volatility/Volume score
        vv_score = 6.0
        if volatility > 0.5:
            vv_score -= 2.0
        elif volatility < 0.25:
            vv_score += 1.0
        vv_score = max(0.0, min(10.0, vv_score))

        reco = _score_and_reco(trend_score, indicators_score, sr_score, vv_score)

        _progress_set(request_id, 88, 'analysis_scoring_ok')

        # Basic info
        stock_name = stock_code
        industry = '未知行业'
        try:
            provider = get_data_provider()
            info = provider.get_stock_info(stock_code, market_type=market_type) or {}

            mt = (market_type or 'A').strip().upper()
            if mt == 'HK':
                stock_name = _pick_first_non_empty(info, [
                    'comcnname',
                    '股票简称',
                    '名称',
                    'comenname',
                    'org_short_name_cn',
                    'org_name_cn',
                    'org_short_name_en',
                    'org_name_en',
                    'code_name',
                ]) or stock_name
            elif mt == 'US':
                stock_name = _pick_first_non_empty(info, [
                    'org_short_name_cn',
                    'org_name_cn',
                    'org_short_name_en',
                    'org_name_en',
                    'comcnname',
                    '股票简称',
                    '名称',
                    'code_name',
                ]) or stock_name
            else:
                stock_name = _pick_first_non_empty(info, [
                    '股票简称',
                    '证券简称',
                    '股票名称',
                    '名称',
                    'code_name',
                    'name',
                ]) or stock_name
            industry = info.get('行业') or info.get('industry') or industry
        except Exception:
            pass

        _progress_set(request_id, 94, 'analysis_stock_info_ok')

        analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        ai_lines = [
            f"# {stock_name}({stock_code}) 智能分析摘要",
            f"当前价格：{current_price:.2f}，涨跌幅：{price_change*100:.2f}%",
            f"RSI(14)：{rsi:.2f}，MACD：{macd['macd']:.4f} / Signal：{macd['macd_signal']:.4f}",
            f"MA趋势：{ma_status}（MA5={ma5:.2f}, MA20={ma20:.2f}, MA60={ma60:.2f}）",
            f"短期支撑位：{short_support:.2f}，短期压力位：{short_resistance:.2f}",
            f"综合建议：{reco['action']}（综合评分 {reco['total_score']}/10）",
        ]
        ai_analysis = "\n\n".join(ai_lines)

        result: Dict[str, Any] = {
            'basic_info': {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'industry': industry,
                'analysis_date': analysis_date,
            },
            'price_data': {
                'current_price': current_price,
                'price_change': price_change,
                'price_change_value': price_change_value,
            },
            'scores': {
                'total_score': reco['total_score'],
                'trend_score': round(trend_score, 1),
                'indicators_score': round(indicators_score, 1),
                'support_resistance_score': round(sr_score, 1),
                'volatility_volume_score': round(vv_score, 1),
            },
            'recommendation': {
                'action': reco['action'],
            },
            'technical_analysis': {
                'trend': {
                    'ma_trend': ma_trend,
                    'ma_status': ma_status,
                },
                'indicators': {
                    'rsi': rsi,
                    'macd': macd['macd'],
                    'macd_signal': macd['macd_signal'],
                    'volatility': volatility,
                },
                'support_resistance': sr,
            },
            'ai_analysis': ai_analysis,
        }
        _progress_set(request_id, 100, 'done')
        _progress_cleanup()
        return jsonify({'result': result})
    except Exception as e:
        _progress_set(request_id, 100, 'error')
        logger.exception('api_enhanced_analysis failed')
        return jsonify({'error': str(e)}), 500


@app.route('/api/progress')
def api_progress():
    request_id = (request.args.get('request_id') or '').strip()
    if not request_id:
        return jsonify({'error': 'request_id不能为空'}), 400
    data = _progress_get(request_id)
    if not data:
        return jsonify({'percent': 0, 'stage': 'unknown'}), 200
    return jsonify({'percent': int(data.get('percent') or 0), 'stage': data.get('stage') or ''}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", "8888")), debug=True)