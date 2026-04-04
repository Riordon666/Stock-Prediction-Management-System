# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from ..utils.indicators import add_indicators
from ..utils.sentiment import get_sentiment_analyzer
from app.core.news_fetcher import news_fetcher

logger = logging.getLogger(__name__)

class FeatureEngine:
    """特征工程核心类：合并价格、指标与情感数据"""
    
    def __init__(self):
        self.sentiment_analyzer = get_sentiment_analyzer()

    def prepare_features(self, df: pd.DataFrame, include_sentiment: bool = True) -> pd.DataFrame:
        """
        准备全量特征
        df: 基础行情数据 (date, open, high, low, close, volume)
        """
        if df.empty:
            return df
            
        # 1. 基础转换
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # 2. 添加技术指标
        df = add_indicators(df)
        
        # 3. 添加情感分析 (如果需要)
        if include_sentiment:
            df = self._add_sentiment_scores(df)
        
        # 4. 处理无穷值（除零等异常情况）
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 5. 最后检查并填充空值
        df = df.ffill().bfill()
        
        return df

    def _add_sentiment_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """为每一天分配情感得分"""
        if df.empty:
            return df
            
        start_date = df['date'].min()
        end_date = df['date'].max()
        
        # 获取近期的所有新闻 (根据df的时间跨度)
        days_span = (end_date - start_date).days + 1
        news_list = news_fetcher.get_latest_news(days=days_span + 2, limit=2000)
        
        if not news_list:
            df['sentiment'] = 0.0
            return df
            
        # 按日期对新闻分组
        news_by_date = {}
        for item in news_list:
            try:
                # 假设新闻日期格式为 YYYY-MM-DD
                dt_str = item.get('date', '')[:10]
                if dt_str not in news_by_date:
                    news_by_date[dt_str] = []
                news_by_date[dt_str].append(item)
            except Exception:
                continue
        
        # 计算每日平均得分
        daily_scores = {}
        for dt_str, items in news_by_date.items():
            daily_scores[dt_str] = self.sentiment_analyzer.analyze_news_list(items)
            
        # 映射到 DataFrame
        df['dt_str'] = df['date'].dt.strftime('%Y-%m-%d')
        df['sentiment'] = df['dt_str'].map(daily_scores).fillna(0.0)
        
        # 平滑处理：情感具有持续性，使用移动平均
        df['sentiment'] = df['sentiment'].rolling(window=3, min_periods=1).mean()
        
        df.drop(columns=['dt_str'], inplace=True)
        return df

# 单例
_feature_engine = None

def get_feature_engine() -> FeatureEngine:
    global _feature_engine
    if _feature_engine is None:
        _feature_engine = FeatureEngine()
    return _feature_engine
