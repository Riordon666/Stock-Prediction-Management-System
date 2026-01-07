from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.core.data_provider import get_data_provider

from .model import build_gru_regression_model


@dataclass
class TrainConfig:
    lookback: int = 30
    total_days: int = 50

    epochs_per_stock: int = 1
    batch_size: int = 16

    units: int = 50
    layers: int = 3
    dropout: float = 0.2
    learning_rate: float = 0.001

    save_every: int = 5

    markets: Tuple[str, ...] = ('A', 'HK')

    a_board: str = 'all'
    a_limit: int = 0

    a_stocks_file: str = ''
    hk_stocks_file: str = ''

    steps: int = 100

    reset: bool = False

    load_existing_weights: bool = True


def _now_ts() -> float:
    return float(time.time())


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _model_dir() -> Path:
    return _root_dir() / 'models' / 'gru'


def _training_data_dir() -> Path:
    return _root_dir() / 'training_data'


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write_json(path: Path, data: Dict) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(str(tmp), str(path))


def _append_jsonl(path: Path, row: Dict) -> None:
    _ensure_dir(path.parent)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _stock_key(market_type: str, code: str) -> str:
    return f"{(market_type or '').strip().upper()}:{str(code).strip()}"


def _hash_universe(symbols: Sequence[str]) -> str:
    h = hashlib.sha256()
    for s in symbols:
        h.update((s.strip() + '\n').encode('utf-8'))
    return h.hexdigest()


def _read_symbols_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    raw = p.read_text(encoding='utf-8', errors='ignore')
    out: List[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith('#'):
            continue
        out.append(s)
    return out


def _normalize_hk_code(code: str) -> str:
    c = (code or '').strip()
    if not c:
        return ''
    c = c.replace('.HK', '').replace('.hk', '')
    digits = ''.join([ch for ch in c if ch.isdigit()])
    if digits:
        return digits.zfill(5)
    return c


def _get_universe(cfg: TrainConfig) -> List[Tuple[str, str]]:
    provider = get_data_provider()

    items: List[Tuple[str, str]] = []
    markets = tuple((m or '').strip().upper() for m in cfg.markets if (m or '').strip())

    for m in markets:
        if m == 'A':
            if cfg.a_stocks_file:
                codes = _read_symbols_file(cfg.a_stocks_file)
            else:
                codes = provider.get_board_stocks(cfg.a_board)
                if cfg.a_limit and int(cfg.a_limit) > 0:
                    codes = list(codes)[: int(cfg.a_limit)]
            items.extend((m, c) for c in codes)
            continue

        if m == 'HK':
            if not cfg.hk_stocks_file:
                raise ValueError('HK training requires --hk-stocks-file')
            raw_codes = _read_symbols_file(cfg.hk_stocks_file)
            codes = [_normalize_hk_code(x) for x in raw_codes]
            codes = [x for x in codes if x]
            items.extend((m, c) for c in codes)
            continue

        raise ValueError(f'unsupported market: {m}')

    seen = set()
    dedup: List[Tuple[str, str]] = []
    for mt, code in items:
        key = (mt, str(code).strip())
        if not key[1] or key in seen:
            continue
        seen.add(key)
        dedup.append(key)
    return dedup


def _cache_training_data(
    market_type: str,
    code: str,
    df: pd.DataFrame,
    x_min: float,
    x_max: float,
    stock_info: Optional[Dict] = None,
) -> None:
    td = _training_data_dir() / market_type
    _ensure_dir(td)

    csv_path = td / f'{code}.csv'
    meta_path = td / f'{code}.json'

    out_df = df.copy()
    out_df['date'] = pd.to_datetime(out_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    out_df.to_csv(csv_path, index=False, encoding='utf-8')

    _atomic_write_json(
        meta_path,
        {
            'market_type': market_type,
            'code': code,
            'ts': _now_ts(),
            'days': int(len(out_df)),
            'x_min': float(x_min),
            'x_max': float(x_max),
            'stock_info': stock_info or {},
        },
    )


def _scale_minmax(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    x = np.asarray(x, dtype=np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min < 1e-12:
        return np.zeros_like(x, dtype=np.float32), x_min, x_max
    return ((x - x_min) / (x_max - x_min)).astype(np.float32), x_min, x_max


def _make_windows(series_scaled: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(series_scaled, dtype=np.float32).reshape(-1)
    if len(values) < lookback + 1:
        raise ValueError('not enough values')

    xs = []
    ys = []
    for i in range(lookback, len(values)):
        xs.append(values[i - lookback : i])
        ys.append(values[i])

    x = np.stack(xs, axis=0).reshape((-1, lookback, 1)).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32).reshape((-1, 1))
    return x, y


def _fetch_last_n_days(code: str, market_type: str, total_days: int) -> pd.DataFrame:
    provider = get_data_provider()

    end = datetime.now().date()
    start = end - timedelta(days=int(total_days) * 3)

    start_s = start.strftime('%Y%m%d')
    end_s = end.strftime('%Y%m%d')

    df = provider.get_stock_history(code=code, start_date=start_s, end_date=end_s, adjust='qfq', market_type=market_type)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    if 'date' not in df.columns and '日期' in df.columns:
        df['date'] = df['日期']

    close_col = 'close' if 'close' in df.columns else ('收盘' if '收盘' in df.columns else None)
    if close_col is None:
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['close'] = pd.to_numeric(df[close_col], errors='coerce')
    df = df.dropna(subset=['date', 'close']).sort_values('date')

    df = df.tail(int(total_days)).reset_index(drop=True)
    return df[['date', 'close']]


def _fetch_stock_info(code: str, market_type: str) -> Dict:
    provider = get_data_provider()
    try:
        info = provider.get_stock_info(code=code, market_type=market_type)
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def _paths() -> Dict[str, Path]:
    md = _model_dir()
    ckpt = md / 'checkpoints'
    return {
        'root': md,
        'checkpoints': ckpt,
        'latest_weights': ckpt / 'latest.weights.h5',
        'history': md / 'history.jsonl',
        'state': md / 'state.json',
        'meta': md / 'meta.json',
    }


def _load_state(state_path: Path) -> Optional[Dict]:
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _save_state(paths: Dict[str, Path], state: Dict) -> None:
    _atomic_write_json(paths['state'], state)


def train_loop(cfg: TrainConfig) -> None:
    paths = _paths()
    _ensure_dir(paths['checkpoints'])

    universe = _get_universe(cfg)
    if not universe:
        raise RuntimeError('empty universe')

    market_counts: Dict[str, int] = {}
    for mt, _ in universe:
        market_counts[mt] = market_counts.get(mt, 0) + 1
    print(
        f"[GRU] universe_size={len(universe)} market_counts={market_counts} lookback={cfg.lookback} total_days={cfg.total_days}",
        flush=True,
    )

    universe_key = _hash_universe([f"{mt}:{code}" for mt, code in universe])

    state = _load_state(paths['state'])
    if cfg.reset:
        state = None

    model = build_gru_regression_model(
        lookback=int(cfg.lookback),
        units=int(cfg.units),
        layers=int(cfg.layers),
        dropout=float(cfg.dropout),
        learning_rate=float(cfg.learning_rate),
    )

    if bool(getattr(cfg, 'load_existing_weights', True)) and paths['latest_weights'].exists():
        model.load_weights(str(paths['latest_weights']))

    if state is not None:
        if state.get('universe_key') != universe_key:
            old = str(state.get('universe_key') or '')
            raise ValueError(
                'universe mismatch: stock list changed since last run. '
                'Set RESET=True in forecasting/train_gru.py (or pass --reset) to rebuild progress. '
                f'old_universe_key={old} new_universe_key={universe_key}'
            )

    if state is None:
        state = {
            'universe_key': universe_key,
            'pos': 0,
            'cycle': 0,
            'trained_total': 0,
            'completed_in_cycle': [],
            'ts': _now_ts(),
            'lookback': int(cfg.lookback),
            'total_days': int(cfg.total_days),
            'model': {
                'units': int(cfg.units),
                'layers': int(cfg.layers),
                'dropout': float(cfg.dropout),
                'learning_rate': float(cfg.learning_rate),
            },
        }
        _atomic_write_json(paths['meta'], {
            'ts': _now_ts(),
            'universe_key': universe_key,
            'universe_size': int(len(universe)),
            'lookback': int(cfg.lookback),
            'total_days': int(cfg.total_days),
            'model': state['model'],
        })
        _save_state(paths, state)

    steps_left = int(cfg.steps)
    save_every = max(1, int(cfg.save_every))
    since_save = 0

    while steps_left > 0:
        completed = set((state.get('completed_in_cycle') or []))
        if len(completed) >= len(universe):
            state['completed_in_cycle'] = []
            completed = set()
            state['pos'] = 0
            state['cycle'] = int(state.get('cycle', 0)) + 1

        start_pos = int(state.get('pos', 0))
        found = None
        for off in range(len(universe)):
            idx = (start_pos + off) % len(universe)
            market_type, code = universe[idx]
            key = _stock_key(market_type, code)
            if key not in completed:
                found = (idx, market_type, code, key)
                break

        if found is None:
            state['completed_in_cycle'] = []
            state['pos'] = 0
            state['cycle'] = int(state.get('cycle', 0)) + 1
            continue

        idx, market_type, code, key = found

        df = _fetch_last_n_days(code=code, market_type=market_type, total_days=int(cfg.total_days))
        if df.empty or len(df) < int(cfg.total_days):
            row = {
                'ts': _now_ts(),
                'market_type': market_type,
                'code': code,
                'cycle': int(state.get('cycle', 0)),
                'pos': int(idx),
                'trained_total': int(state.get('trained_total', 0)),
                'status': 'skip_empty',
            }
            _append_jsonl(paths['history'], row)
            completed.add(key)
            state['completed_in_cycle'] = list(completed)
            state['pos'] = int(idx) + 1
            state['ts'] = _now_ts()
            _save_state(paths, state)
            print(
                f"[GRU] cycle={state.get('cycle', 0)} {len(completed)}/{len(universe)} {market_type}:{code} status=skip_empty",
                flush=True,
            )
            steps_left -= 1
            continue

        close = df['close'].to_numpy(dtype=np.float32)
        series_scaled, x_min, x_max = _scale_minmax(close)
        x, y = _make_windows(series_scaled, lookback=int(cfg.lookback))

        expected_samples = int(cfg.total_days) - int(cfg.lookback)
        if expected_samples > 0 and int(x.shape[0]) != expected_samples:
            row = {
                'ts': _now_ts(),
                'market_type': market_type,
                'code': code,
                'cycle': int(state.get('cycle', 0)),
                'pos': int(idx),
                'trained_total': int(state.get('trained_total', 0)),
                'samples': int(x.shape[0]),
                'expected_samples': int(expected_samples),
                'status': 'skip_bad_samples',
            }
            _append_jsonl(paths['history'], row)
            completed.add(key)
            state['completed_in_cycle'] = list(completed)
            state['pos'] = int(idx) + 1
            state['ts'] = _now_ts()
            _save_state(paths, state)
            steps_left -= 1
            continue

        stock_info = _fetch_stock_info(code=code, market_type=market_type)

        _cache_training_data(
            market_type=market_type,
            code=code,
            df=df,
            x_min=x_min,
            x_max=x_max,
            stock_info=stock_info,
        )

        hist = model.fit(
            x,
            y,
            epochs=int(cfg.epochs_per_stock),
            batch_size=int(cfg.batch_size),
            verbose=0,
            shuffle=True,
            validation_split=0.2,
        )

        loss = None
        val_loss = None
        try:
            loss = float(hist.history.get('loss', [None])[-1])
            val_loss = float(hist.history.get('val_loss', [None])[-1])
        except Exception:
            pass

        row = {
            'ts': _now_ts(),
            'market_type': market_type,
            'code': code,
            'cycle': int(state.get('cycle', 0)),
            'pos': int(idx),
            'trained_total': int(state.get('trained_total', 0)) + 1,
            'samples': int(x.shape[0]),
            'loss': loss,
            'val_loss': val_loss,
            'status': 'ok',
        }
        _append_jsonl(paths['history'], row)

        completed.add(key)
        state['completed_in_cycle'] = list(completed)
        state['pos'] = int(idx) + 1
        state['trained_total'] = int(state.get('trained_total', 0)) + 1
        state['ts'] = _now_ts()

        model.save_weights(str(paths['latest_weights']))
        _save_state(paths, state)

        try:
            loss_s = f"{loss:.6f}" if loss is not None else "None"
            val_s = f"{val_loss:.6f}" if val_loss is not None else "None"
        except Exception:
            loss_s = str(loss)
            val_s = str(val_loss)

        print(
            f"[GRU] cycle={state.get('cycle', 0)} {len(completed)}/{len(universe)} {market_type}:{code} status=ok loss={loss_s} val_loss={val_s}",
            flush=True,
        )

        since_save += 1
        if since_save >= save_every:
            since_save = 0

        steps_left -= 1

    model.save_weights(str(paths['latest_weights']))
    _save_state(paths, state)
