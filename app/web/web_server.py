# -*- coding: utf-8 -*-
"""
基础Web服务器架构 - 股票智能分析系统
版本：v1.0.0
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

# Flask相关导入
from flask import Flask, render_template, request, jsonify

import numpy as np
import pandas as pd

# 环境变量加载
from dotenv import load_dotenv

from ..core.data_provider import get_data_provider

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
    if market_type not in {'A'}:
        raise ValueError('当前仅支持A股市场分析')

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=_period_to_days(period))

    provider = get_data_provider()
    df = provider.get_stock_history(
        code=stock_code,
        start_date=_fmt_yyyymmdd(start_dt),
        end_date=_fmt_yyyymmdd(end_dt),
        adjust='qfq',
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
        action = '强烈买入'
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


@app.route('/api/stock_data')
def api_stock_data():
    stock_code = (request.args.get('stock_code') or '').strip()
    market_type = (request.args.get('market_type') or 'A').strip()
    period = (request.args.get('period') or '1y').strip()

    if not stock_code:
        return jsonify({'error': 'stock_code不能为空'}), 400

    try:
        df = _get_history_df(stock_code, market_type, period)
        if df.empty:
            return jsonify({'data': []})

        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA60'] = df['close'].rolling(60).mean()

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
        return jsonify({'data': out})
    except Exception as e:
        logger.exception('api_stock_data failed')
        return jsonify({'error': str(e)}), 500


@app.route('/api/enhanced_analysis', methods=['POST'])
def api_enhanced_analysis():
    payload = request.get_json(silent=True) or {}
    stock_code = (payload.get('stock_code') or '').strip()
    market_type = (payload.get('market_type') or 'A').strip()
    period = (payload.get('period') or '1y').strip()

    if not stock_code:
        return jsonify({'error': 'stock_code不能为空'}), 400

    try:
        df = _get_history_df(stock_code, market_type, period)
        if df.empty:
            return jsonify({'error': '未找到股票数据'}), 404

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

        # Basic info
        stock_name = stock_code
        industry = '未知行业'
        try:
            provider = get_data_provider()
            info = provider.get_stock_info(stock_code) or {}
            stock_name = info.get('股票简称') or info.get('code_name') or info.get('名称') or stock_name
            industry = info.get('行业') or info.get('industry') or industry
        except Exception:
            pass

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
        return jsonify({'result': result})
    except Exception as e:
        logger.exception('api_enhanced_analysis failed')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", "8888")), debug=True)