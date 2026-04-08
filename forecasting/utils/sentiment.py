# -*- coding: utf-8 -*-
import logging
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """新闻情感分析模块"""
    
    def __init__(self, model_name: str = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"):
        self.model_name = model_name
        self.analyzer = None
        self._initialized = False

    def _init_model(self):
        if self._initialized:
            return
        try:
            from transformers import pipeline
            # 增加对 HuggingFace 环境的连接性判断或直接捕捉所有加载异常
            # 使用一个更标准的基础模型，或者如果加载失败则彻底降级
            logger.info(f"Attempting to load sentiment model: {self.model_name}")
            self.analyzer = pipeline("sentiment-analysis", model=self.model_name, device=-1, framework='pt')
            self._initialized = True
            logger.info(f"Sentiment model {self.model_name} initialized.")
        except Exception as e:
            logger.warning(f"Sentiment model load failed ({e}), using keyword-based fallback.")
            self.analyzer = None
            self._initialized = True # 标记为已尝试初始化，避免重复报错

    def get_sentiment_score(self, text: str) -> float:
        """
        对单条文本进行情感打分。
        返回结果范围建议在 [-1, 1] 之间：
        1: 强烈看涨
        0: 中性
        -1: 强烈看跌
        """
        if not text or not text.strip():
            return 0.0
            
        self._init_model()
        
        if not self.analyzer:
            # 简单降级：极简关键词打分（临时替代）
            pos_words = ['涨', '利好', '拉升', '增持', '走强', '反弹', '机会', '突破']
            neg_words = ['跌', '利空', '下行', '减持', '走弱', '跳水', '风险', '跌破']
            
            score = 0.0
            for w in pos_words:
                if w in text: score += 0.2
            for w in neg_words:
                if w in text: score -= 0.2
            return max(-1.0, min(1.0, score))

        try:
            # Roberta 情感分析通常返回 Positive/Negative
            result = self.analyzer(text[:512])[0]
            label = result['label'] # 具体取决于模型的标签定义，这里假设是标准情感标签
            score = result['score']
            
            # 这里需要根据具体模型的 Label 映射到数值
            # 假设 Label 为 'Positive' -> 1, 'Negative' -> -1
            if 'Positive' in label or '正' in label or 'High' in label:
                return score
            elif 'Negative' in label or '负' in label or 'Low' in label:
                return -score
            else:
                return 0.0
        except Exception:
            return 0.0

    def analyze_news_list(self, news_items: List[Dict]) -> float:
        """
        分析一组新闻，返回平均情感得分。
        """
        if not news_items:
            return 0.0
            
        scores = []
        for item in news_items:
            content = item.get('content') or item.get('title', '')
            if content:
                scores.append(self.get_sentiment_score(content))
        
        if not scores:
            return 0.0
        return float(np.mean(scores))

_analyzer = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer
