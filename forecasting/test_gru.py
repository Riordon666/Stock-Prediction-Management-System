from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from app.core.data_provider import get_data_provider

from forecasting.models.gru.model import build_gru_regression_model


# ========== 可直接修改的测试参数（改完保存后直接运行本文件即可）==========
MARKET_TYPE = 'A'  # 'A' 或者 'HK'
CODE = '600519'  # 填A股代码或者HK股代码

LOOKBACK = 30
TOTAL_DAYS = 50

WEIGHTS_PATH = str((Path(__file__).resolve().parent / 'models' / 'gru' / 'checkpoints' / 'latest.weights.h5'))


def _scale_minmax(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min < 1e-12:
        return np.zeros_like(x, dtype=np.float32), x_min, x_max
    return ((x - x_min) / (x_max - x_min)).astype(np.float32), x_min, x_max


def _inverse_minmax(x_scaled: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    x_scaled = np.asarray(x_scaled, dtype=np.float32)
    if x_max - x_min < 1e-12:
        return np.full_like(x_scaled, fill_value=x_min, dtype=np.float32)
    return x_scaled * (x_max - x_min) + x_min


def _fetch_last_n_days(code: str, market_type: str, total_days: int) -> pd.DataFrame:
    provider = get_data_provider()

    end = datetime.now().date()
    start = end - timedelta(days=int(total_days) * 3)

    start_s = start.strftime('%Y%m%d')
    end_s = end.strftime('%Y%m%d')

    df = provider.get_stock_history(
        code=code,
        start_date=start_s,
        end_date=end_s,
        adjust='qfq',
        market_type=market_type,
    )

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


def predict_next_close() -> None:
    df = _fetch_last_n_days(code=CODE, market_type=MARKET_TYPE, total_days=int(TOTAL_DAYS))
    if df.empty or len(df) < int(TOTAL_DAYS):
        raise RuntimeError(f'not enough data: got {len(df)} rows')

    close = df['close'].to_numpy(dtype=np.float32)
    scaled, x_min, x_max = _scale_minmax(close)

    if len(scaled) < int(LOOKBACK):
        raise RuntimeError('not enough values for lookback')

    x_last = scaled[-int(LOOKBACK) :].reshape((1, int(LOOKBACK), 1)).astype(np.float32)

    model = build_gru_regression_model(lookback=int(LOOKBACK))
    weights_path = Path(WEIGHTS_PATH)
    if not weights_path.exists():
        raise FileNotFoundError(str(weights_path))
    model.load_weights(str(weights_path))

    y_scaled = model.predict(x_last, verbose=0).reshape(-1)
    y = _inverse_minmax(y_scaled, x_min=x_min, x_max=x_max).reshape(-1)

    last_date = df.iloc[-1]['date']
    last_close = float(df.iloc[-1]['close'])

    print(f"market={MARKET_TYPE} code={CODE}")
    print(f"data_range={df.iloc[0]['date'].date()} -> {last_date.date()} rows={len(df)}")
    print(f"last_close={last_close:.4f}")
    print(f"pred_next_close={float(y[0]):.4f}")


if __name__ == '__main__':
    predict_next_close()
