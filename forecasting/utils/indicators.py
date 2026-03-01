# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    为数据框添加技术指标。要求输入包含: open, high, low, close, volume。
    """
    df = df.copy()
    
    # 映射中文字段到英文字段
    mapping = {
        '日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', 
        '收盘': 'close', '成交量': 'volume', '成交额': 'amount'
    }
    df.rename(columns=mapping, inplace=True)
    
    # 确保列存在且为数值型
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # 如果缺少成交量，尝试用0填充 (避坑)
            if col == 'volume':
                df['volume'] = 0
            else:
                raise KeyError(f"Missing required column: {col}")

    # 1. 移动平均线 (MA)
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    
    # 2. 指数移动平均线 (EMA)
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # 3. MACD
    df['macd_diff'] = df['ema12'] - df['ema26']
    df['macd_dea'] = df['macd_diff'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = (df['macd_diff'] - df['macd_dea']) * 2
    
    # 4. RSI (14日)
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 5. KDJ (9, 3, 3)
    low_list = df['low'].rolling(window=9).min()
    high_list = df['high'].rolling(window=9).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    df['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
    df['kdj_d'] = df['kdj_k'].ewm(com=2, adjust=False).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    
    # 6. 成交量移动平均
    df['v_ma5'] = df['volume'].rolling(window=5).mean()
    df['v_ma10'] = df['volume'].rolling(window=10).mean()
    
    # 7. 价格变动率 (ROC - 简单的周期涨幅)
    df['change'] = df['close'].pct_change()
    
    # 填充空值 (由于rolling window产生的NaN)
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    
    return df
