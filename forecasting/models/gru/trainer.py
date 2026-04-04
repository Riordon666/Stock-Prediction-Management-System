from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .model import build_gru_regression_model
from ...core.feature_engine import get_feature_engine
from app.core.data_provider import get_data_provider


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

    autoregressive_training: bool = False

    # 缓存的股票列表，避免自动重置时重新获取
    cached_universe: Optional[List[Tuple[str, str]]] = None


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


def _atomic_write_json(path: Path, data: Dict, retries: int = 3, delay: float = 0.25) -> None:
    _ensure_dir(path.parent)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    tmp = path.with_name(f"{path.name}.{int(time.time() * 1000)}.{os.getpid()}.tmp")
    tmp.write_text(payload, encoding='utf-8')

    last_exc: Optional[Exception] = None
    for attempt in range(max(1, int(retries))):
        try:
            os.replace(str(tmp), str(path))
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(delay * (attempt + 1))
        except Exception as exc:
            last_exc = exc
            break

    try:
        path.write_text(payload, encoding='utf-8')
        return
    except Exception as exc:
        last_exc = exc
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

    if last_exc is not None:
        raise last_exc


def _append_jsonl(path: Path, row: Dict) -> None:
    _ensure_dir(path.parent)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _save_json(path: Path, data: Dict) -> None:
    _atomic_write_json(path, data)


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


def _normalize_a_code(code: str) -> str:
    c = (code or '').strip()
    if not c:
        return ''
    c = c.replace('.SH', '').replace('.SZ', '').replace('.BJ', '')
    c = c.replace('.sh', '').replace('.sz', '').replace('.bj', '')
    c = c.replace('sh', '').replace('sz', '').replace('bj', '')
    digits = ''.join([ch for ch in c if ch.isdigit()])
    if len(digits) != 6:
        return ''
    if digits[0] not in {'0', '3', '6'}:
        return ''
    return digits


def _get_universe(cfg: TrainConfig) -> List[Tuple[str, str]]:
    provider = get_data_provider()

    items: List[Tuple[str, str]] = []
    markets = tuple((m or '').strip().upper() for m in cfg.markets if (m or '').strip())

    for m in markets:
        if m == 'A':
            if cfg.a_stocks_file:
                codes = _read_symbols_file(cfg.a_stocks_file)
            else:
                # 重试机制：akshare API 不稳定时多次尝试
                codes = []
                max_retries = 3
                for retry in range(max_retries):
                    codes = provider.get_board_stocks(cfg.a_board)
                    codes = sorted([str(x).strip() for x in codes if str(x).strip()])
                    if codes:
                        break
                    print(f"[WARN] A-share fetch attempt {retry + 1}/{max_retries} failed, retrying...", flush=True)
                    import time
                    time.sleep(2)
                if not codes:
                    print(f"[ERROR] A-share fetch failed after {max_retries} retries", flush=True)
            raw_codes = list(codes)
            codes = [_normalize_a_code(x) for x in raw_codes]
            codes = [x for x in codes if x]
            if cfg.a_limit and int(cfg.a_limit) > 0:
                codes = list(codes)[: int(cfg.a_limit)]
            if not codes:
                src = cfg.a_stocks_file or f"board={cfg.a_board}"
                print(f"[WARN] A-share universe empty (source={src}).", flush=True)
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


def _scale_minmax_multi(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    对多列进行归一化。
    返回: 归一化后的DF, 每个特征的 (min, max) 映射。
    """
    df_scaled = df.copy()
    meta = {}
    for col in feature_cols:
        # 强制转换为数值类型，非法值转为 NaN
        df_scaled[col] = pd.to_numeric(df_scaled[col], errors='coerce')
        # 将 NaN 填充为 0.0（或其他合理默认值）
        df_scaled[col] = df_scaled[col].fillna(0.0)
        vals = df_scaled[col].values
        # 处理全列为空或全列相同的情况
        if vals.size == 0 or not np.isfinite(vals).any():
            df_scaled[col] = 0.0
            meta[col] = (0.0, 0.0)
            continue
        v_min = float(np.nanmin(vals))
        v_max = float(np.nanmax(vals))
        denom = v_max - v_min
        if denom < 1e-12 or not np.isfinite(denom):
            df_scaled[col] = 0.0
        else:
            df_scaled[col] = (vals - v_min) / denom
        meta[col] = (v_min, v_max)
    return df_scaled, meta


def _make_windows_multi(df_scaled: pd.DataFrame, feature_cols: List[str], target_col: str, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建多特征窗口。
    """
    if len(df_scaled) < lookback + 1:
        raise ValueError('not enough values')

    features = df_scaled[feature_cols].values.astype(np.float32)
    targets = df_scaled[target_col].values.astype(np.float32)

    xs = []
    ys = []
    for i in range(lookback, len(df_scaled)):
        xs.append(features[i - lookback : i])
        ys.append(targets[i])

    x = np.stack(xs, axis=0).astype(np.float32) # (samples, lookback, features)
    y = np.asarray(ys, dtype=np.float32).reshape((-1, 1))
    return x, y


def _fetch_last_n_days_enriched(code: str, market_type: str, total_days: int, lookback: int) -> pd.DataFrame:
    provider = get_data_provider()
    engine = get_feature_engine()

    # 为了计算技术指标，我们需要多获取一些历史数据
    fetch_days = int(total_days) + 60 
    end = datetime.now().date()
    start = end - timedelta(days=fetch_days * 2)

    df = provider.get_stock_history(code=code, start_date=start.strftime('%Y%m%d'), 
                                    end_date=end.strftime('%Y%m%d'), adjust='qfq', market_type=market_type)
    if df is None or df.empty:
        return pd.DataFrame()

    # 特征工程
    df = engine.prepare_features(df, include_sentiment=True)
    
    # 只取最后需要的截断长度
    df = df.tail(int(total_days)).reset_index(drop=True)
    return df


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
        'lifetime': md / 'lifetime.json',
    }


def _load_state(state_path: Path) -> Optional[Dict]:
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _load_lifetime(paths: Dict[str, Path]) -> Dict:
    data = _load_json(paths['lifetime']) or {}
    return {
        'runs': int(data.get('runs', 0)),
        'trained_total': int(data.get('trained_total', 0)),
        'cycles_completed': int(data.get('cycles_completed', 0)),
        'ts': float(data.get('ts', 0.0) or 0.0),
    }


def _save_lifetime(paths: Dict[str, Path], lifetime: Dict) -> None:
    lifetime['ts'] = _now_ts()
    _save_json(paths['lifetime'], lifetime)


def _as_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _autoregressive_train(
    model,
    values_scaled: np.ndarray,
    lookback: int,
    epochs: int,
    x_min: float,
    x_max: float,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    values_scaled = np.asarray(values_scaled, dtype=np.float32).reshape(-1)
    steps = int(len(values_scaled) - int(lookback))
    if steps <= 0:
        return None, None, None

    denom = float(x_max - x_min)
    if abs(denom) < 1e-12:
        denom = 1.0

    losses: List[float] = []
    se_scaled: List[float] = []
    se_real: List[float] = []

    for _ in range(max(1, int(epochs))):
        window = values_scaled[: int(lookback)].astype(np.float32).copy()
        for t in range(int(lookback), len(values_scaled)):
            x = window.reshape((1, int(lookback), 1)).astype(np.float32)
            y_true_v = float(values_scaled[t])
            y_true = np.asarray([[y_true_v]], dtype=np.float32)

            pred_v = float(model.predict(x, verbose=0)[0][0])
            se_scaled.append(float((pred_v - y_true_v) ** 2))

            pred_real = pred_v * denom + float(x_min)
            true_real = y_true_v * denom + float(x_min)
            se_real.append(float((pred_real - true_real) ** 2))

            loss = model.train_on_batch(x, y_true)
            loss_f = _as_float(loss)
            if loss_f is not None:
                losses.append(loss_f)

            window = np.concatenate([window[1:], np.asarray([pred_v], dtype=np.float32)], axis=0)

    avg_loss = float(np.mean(losses)) if losses else None
    mse_scaled = float(np.mean(se_scaled)) if se_scaled else None
    mse_real = float(np.mean(se_real)) if se_real else None
    return avg_loss, mse_scaled, mse_real


def _save_state(paths: Dict[str, Path], state: Dict) -> None:
    try:
        _atomic_write_json(paths['state'], state)
    except PermissionError as exc:
        print(
            f"[WARN] save state failed (will retry later): {type(exc).__name__}: {exc}",
            flush=True,
        )


def _save_weights_safe(model, path: Path, retries: int = 3, delay: float = 0.4) -> bool:
    _ensure_dir(path.parent)
    last_exc: Optional[Exception] = None
    
    # Determine the correct suffix to preserve. Keras 3 requires '.weights.h5'
    base_name = path.name
    if base_name.endswith('.weights.h5'):
        stem = base_name[:-11]  # remove '.weights.h5'
        suffix = '.weights.h5'
    else:
        stem = path.stem
        suffix = path.suffix
        
    for attempt in range(max(1, int(retries))):
        tmp = path.with_name(
            f"{stem}.{int(time.time() * 1000)}.{os.getpid()}.{attempt}{suffix}"
        )
        try:
            model.save_weights(str(tmp))
            bak = None
            if path.name == 'latest.weights.h5':
                bak = path.with_name('latest.prev.weights.h5')
            if bak is not None and path.exists():
                try:
                    os.replace(str(path), str(bak))
                except Exception:
                    pass
            os.replace(str(tmp), str(path))
            return True
        except Exception as exc:
            last_exc = exc
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            time.sleep(delay * (attempt + 1))

    if last_exc is not None:
        print(
            f"[WARN] save weights failed (skip): {type(last_exc).__name__}: {last_exc}",
            flush=True,
        )
    return False


def _load_weights_compat(model, path: Path) -> Tuple[bool, Optional[Exception]]:
    # 策略 1：直接调用 Keras 原生 load_weights
    try:
        model.load_weights(str(path))
        return True, None
    except Exception as exc:
        first_exc: Exception = exc

    # 策略 2：使用旧版 Keras 2 的 hdf5_format 加载（跳过 Keras 3 格式）
    try:
        import h5py
        from tensorflow.python.keras.saving import hdf5_format

        with h5py.File(str(path), 'r') as f:
            root_keys = set(list(f.keys()))
            if 'vars' not in root_keys:
                hdf5_format.load_weights_from_hdf5_group(f, model.layers)
                return True, None
    except Exception:
        pass

    # 策略 3：手动从 h5 文件中按名称匹配并逐层赋值权重（兼容 Keras 2 和 Keras 3 格式）
    try:
        import h5py

        def _collect_arrays(group) -> Dict[str, np.ndarray]:
            """递归地从 h5 group 中收集所有数据集（numpy 数组）"""
            result: Dict[str, np.ndarray] = {}
            for key in group.keys():
                item = group[key]
                if hasattr(item, 'keys'):
                    sub = _collect_arrays(item)
                    for sk, sv in sub.items():
                        result[f"{key}/{sk}"] = sv
                else:
                    try:
                        result[key] = item[()]
                    except Exception:
                        pass
            return result

        with h5py.File(str(path), 'r') as f:
            all_arrays = _collect_arrays(f)

        if not all_arrays:
            return False, first_exc

        # 将所有提取出来的数组按 shape 分组，方便后续匹配
        # 获取模型中每一层期望的权重 shapes
        assigned_count = 0
        for layer in model.layers:
            layer_weights = layer.get_weights()
            if not layer_weights:
                continue

            # 为这个 layer 寻找 shape 完全匹配的权重数组
            expected_shapes = [w.shape for w in layer_weights]
            matched_arrays: List[np.ndarray] = []
            used_keys: List[str] = []

            # 先尝试按层名称进行精确匹配
            layer_name = layer.name
            name_matched_arrays: Dict[str, np.ndarray] = {}
            for arr_key, arr_val in all_arrays.items():
                if layer_name in arr_key:
                    name_matched_arrays[arr_key] = arr_val

            if name_matched_arrays:
                # 按 expected shape 的顺序从名称匹配的数组中找
                for shape in expected_shapes:
                    found = False
                    for arr_key, arr_val in name_matched_arrays.items():
                        if arr_key not in used_keys and arr_val.shape == shape:
                            matched_arrays.append(arr_val)
                            used_keys.append(arr_key)
                            found = True
                            break
                    if not found:
                        matched_arrays = []
                        used_keys = []
                        break

            # 如果名称匹配失败，尝试纯 shape 匹配（从剩余的数组中按顺序找）
            if len(matched_arrays) != len(expected_shapes):
                matched_arrays = []
                used_keys_set = set()
                for shape in expected_shapes:
                    found = False
                    for arr_key, arr_val in all_arrays.items():
                        if arr_key not in used_keys_set and arr_val.shape == shape:
                            matched_arrays.append(arr_val)
                            used_keys_set.add(arr_key)
                            found = True
                            break
                    if not found:
                        matched_arrays = []
                        break

            if len(matched_arrays) == len(expected_shapes):
                layer.set_weights(matched_arrays)
                assigned_count += 1
                # 从 all_arrays 中移除已使用的，避免重复分配
                for k in used_keys:
                    all_arrays.pop(k, None)

        if assigned_count > 0:
            print(
                f"[INFO] 手动按层名/shape匹配加载权重成功: 共匹配 {assigned_count} 个层。",
                flush=True,
            )
            return True, None

        return False, first_exc
    except Exception:
        return False, first_exc


def train_loop(cfg: TrainConfig) -> None:
    paths = _paths()
    _ensure_dir(paths['checkpoints'])

    lifetime = _load_lifetime(paths)

    # 优先使用缓存的 universe，避免自动重置时重新获取
    if cfg.cached_universe:
        universe = cfg.cached_universe
    else:
        universe = _get_universe(cfg)
        cfg.cached_universe = universe  # 缓存供后续使用
    if not universe:
        raise RuntimeError('empty universe')

    market_counts: Dict[str, int] = {}
    for mt, _ in universe:
        market_counts[mt] = market_counts.get(mt, 0) + 1
    print(
        f"[GRU] universe_size={len(universe)} market_counts={market_counts} lookback={cfg.lookback} total_days={cfg.total_days}",
        flush=True,
    )

    universe_key = _hash_universe(sorted([f"{mt}:{code}" for mt, code in universe]))

    state = _load_state(paths['state'])
    if cfg.reset:
        state = None

    # 获取示例数据以确定特征数量
    print("[INFO] Warming up feature engine to determine feature count...", flush=True)
    sample_df = _fetch_last_n_days_enriched(universe[0][1], universe[0][0], cfg.total_days, cfg.lookback)
    if sample_df.empty:
        # 降级：如果找不到第一只股票，尝试找一只可以的
        for mt, c in universe[:10]:
            sample_df = _fetch_last_n_days_enriched(c, mt, cfg.total_days, cfg.lookback)
            if not sample_df.empty: break
            
    if sample_df.empty:
        raise RuntimeError("Could not fetch sample data to initialize model.")
        
    feature_cols = [c for c in sample_df.columns if c not in ['date', 'dt_str']]
    feature_count = len(feature_cols)
    print(f"[GRU] Multi-feature enabled: features={feature_cols} count={feature_count}")

    model = build_gru_regression_model(
        lookback=int(cfg.lookback),
        feature_count=feature_count,
        units=int(cfg.units),
        layers=int(cfg.layers),
        dropout=float(cfg.dropout),
        learning_rate=float(cfg.learning_rate),
    )

    if bool(getattr(cfg, 'load_existing_weights', True)) and paths['latest_weights'].exists():
        ok, err = _load_weights_compat(model, paths['latest_weights'])
        if not ok:
            prev = paths['latest_weights'].with_name('latest.prev.weights.h5')
            if prev.exists():
                ok2, err2 = _load_weights_compat(model, prev)
                if ok2:
                    ok, err = True, None
                    print(
                        f"[WARN] latest.weights.h5 load failed; fallback to {prev.name} succeeded.",
                        flush=True,
                    )
                else:
                    err = err2 or err

        if not ok:
            print(
                f"[WARN] load latest.weights.h5 failed: {type(err).__name__ if err else 'Error'}: {err}. "
                "所有权重加载策略均失败，将使用全新模型从头开始训练。",
                flush=True,
            )
            # 重置训练状态，从头开始
            state = None
            cfg.reset = True
        else:
            try:
                print("[INFO] loaded existing weights.", flush=True)
            except Exception:
                pass

    if state is not None:
        if state.get('universe_key') != universe_key:
            old = str(state.get('universe_key') or '')
            raise ValueError(
                'universe mismatch: stock list changed since last run. '
                'Set RESET=True in forecasting/train_gru.py (or pass --reset) to rebuild progress. '
                f'old_universe_key={old} new_universe_key={universe_key}'
            )

    if state is None:
        lifetime['runs'] = int(lifetime.get('runs', 0)) + 1
        _save_lifetime(paths, lifetime)
        state = {
            'universe_key': universe_key,
            'pos': 0,
            'cycle': 0,
            'trained_total': 0,
            'run_id': int(lifetime.get('runs', 1)),
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
            'autoregressive_training': bool(getattr(cfg, 'autoregressive_training', False)),
        }
        _atomic_write_json(paths['meta'], {
            'ts': _now_ts(),
            'universe_key': universe_key,
            'universe_size': int(len(universe)),
            'lookback': int(cfg.lookback),
            'total_days': int(cfg.total_days),
            'model': state['model'],
            'feature_cols': feature_cols,
            'feature_count': feature_count,
            'run_id': int(state.get('run_id', 1)),
            'autoregressive_training': bool(getattr(cfg, 'autoregressive_training', False)),
        })
        _save_state(paths, state)

    steps_left = int(cfg.steps)
    save_every = max(1, int(cfg.save_every))
    since_save = 0
    interrupted = False

    def handle_sigterm(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[WARN] 收到终止信号 (SIGTERM); 正在保存当前进度并退出...", flush=True)

    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        while steps_left > 0 and not interrupted:
            completed = set((state.get('completed_in_cycle') or []))
            if len(completed) >= len(universe):
                state['completed_in_cycle'] = []
                completed = set()
                state['pos'] = 0
                state['cycle'] = int(state.get('cycle', 0)) + 1
                lifetime['cycles_completed'] = int(lifetime.get('cycles_completed', 0)) + 1
                _save_lifetime(paths, lifetime)

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

            df = _fetch_last_n_days_enriched(code=code, market_type=market_type, total_days=int(cfg.total_days), lookback=int(cfg.lookback))
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

            stock_info = _fetch_stock_info(code=code, market_type=market_type)

            # 数据质量校验：检查特征列是否存在严重非法值
            valid_cols = [c for c in feature_cols if c in df.columns]
            if not valid_cols:
                row = {
                    'ts': _now_ts(),
                    'market_type': market_type,
                    'code': code,
                    'cycle': int(state.get('cycle', 0)),
                    'pos': int(idx),
                    'trained_total': int(state.get('trained_total', 0)),
                    'status': 'skip_no_features',
                }
                _append_jsonl(paths['history'], row)
                completed.add(key)
                state['completed_in_cycle'] = list(completed)
                state['pos'] = int(idx) + 1
                state['ts'] = _now_ts()
                _save_state(paths, state)
                print(f"[GRU] cycle={state.get('cycle', 0)} {len(completed)}/{len(universe)} {market_type}:{code} status=skip_no_features", flush=True)
                steps_left -= 1
                continue

            # 归一化多列
            df_scaled, scaling_meta = _scale_minmax_multi(df, valid_cols)
            x_min, x_max = scaling_meta['close']

            _cache_training_data(
                market_type=market_type,
                code=code,
                df=df,
                x_min=x_min,
                x_max=x_max,
                stock_info=stock_info,
            )

            loss = None
            val_loss = None
            mse_scaled = None
            mse_real = None

            if bool(getattr(cfg, 'autoregressive_training', False)):
                # 自回归训练暂不支持多特征
                print("[WARN] Autoregressive training not optimized for multi-feature. Running standard fit.", flush=True)

            x, y = _make_windows_multi(df_scaled, valid_cols, 'close', lookback=int(cfg.lookback))
            hist = model.fit(
                x,
                y,
                epochs=int(cfg.epochs_per_stock),
                batch_size=int(cfg.batch_size),
                verbose=0,
                shuffle=True,
                validation_split=0.2,
            )

            try:
                loss = float(hist.history.get('loss', [None])[-1])
                val_loss = float(hist.history.get('val_loss', [None])[-1])
            except Exception:
                pass

            # 计算 MSE（真实价格空间）
            try:
                y_pred = model.predict(x, verbose=0).flatten()
                y_true = y.flatten()
                # 归一化空间的 MSE
                mse_scaled = float(np.mean((y_pred - y_true) ** 2))
                # 真实价格空间的 MSE
                x_min, x_max = scaling_meta.get('close', (0.0, 1.0))
                denom = x_max - x_min
                if abs(denom) < 1e-12:
                    denom = 1.0
                y_pred_real = y_pred * denom + x_min
                y_true_real = y_true * denom + x_min
                mse_real = float(np.mean((y_pred_real - y_true_real) ** 2))
            except Exception:
                pass

            samples_count = int(len(x))
            row = {
                'ts': _now_ts(),
                'market_type': market_type,
                'code': code,
                'run_id': int(state.get('run_id', 1)),
                'cycle': int(state.get('cycle', 0)),
                'pos': int(idx),
                'trained_total': int(state.get('trained_total', 0)) + 1,
                'samples': samples_count,
                'loss': loss,
                'val_loss': val_loss,
                'mse_scaled': mse_scaled,
                'mse_real': mse_real,
                'autoregressive_training': bool(getattr(cfg, 'autoregressive_training', False)),
                'status': 'ok',
            }
            _append_jsonl(paths['history'], row)

            completed.add(key)
            state['completed_in_cycle'] = list(completed)
            state['pos'] = int(idx) + 1
            state['trained_total'] = int(state.get('trained_total', 0)) + 1
            state['ts'] = _now_ts()

            lifetime['trained_total'] = int(lifetime.get('trained_total', 0)) + 1
            _save_lifetime(paths, lifetime)

            _save_weights_safe(model, paths['latest_weights'])
            _save_state(paths, state)

            try:
                loss_s = f"{loss:.6f}" if loss is not None else "None"
                val_s = f"{val_loss:.6f}" if val_loss is not None else "None"
            except Exception:
                loss_s = str(loss)
                val_s = str(val_loss)

            try:
                mse_scaled_s = f"{mse_scaled:.6f}" if mse_scaled is not None else "None"
                mse_real_s = f"{mse_real:.6f}" if mse_real is not None else "None"
            except Exception:
                mse_scaled_s = str(mse_scaled)
                mse_real_s = str(mse_real)

            print(
                f"[GRU] run={state.get('run_id', 1)} total_cycles={lifetime.get('cycles_completed', 0)} total_trained={lifetime.get('trained_total', 0)} "
                f"cycle={state.get('cycle', 0)} {len(completed)}/{len(universe)} {market_type}:{code} status=ok loss={loss_s} val_loss={val_s} mse_scaled={mse_scaled_s} mse_real={mse_real_s}",
                flush=True,
            )

            since_save += 1
            if since_save >= save_every:
                since_save = 0

            steps_left -= 1
    except KeyboardInterrupt:
        interrupted = True
        print("[WARN] training interrupted by user; saving state/weights.", flush=True)
    finally:
        _save_weights_safe(model, paths['latest_weights'])
        _save_state(paths, state)
        if interrupted:
            print("[INFO] training stopped safely.", flush=True)
