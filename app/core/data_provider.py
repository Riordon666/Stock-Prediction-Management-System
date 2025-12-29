# -*- coding: utf-8 -*-
"""
统一数据提供层 - 老王说：调数据就找我，别管底下用的啥！
单例模式，全局共享
"""
import logging
from typing import List, Dict, Optional
import pandas as pd

from .fallback_manager import FallbackManager

logger = logging.getLogger(__name__)


class DataProvider:
    """统一数据提供层，封装多数据源故障转移"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._init_adapters()
        self._initialized = True

    def _init_adapters(self):
        """初始化适配器"""
        self.akshare = None
        self.baostock = None
        adapters = []

        try:
            from ..adapters.akshare_adapter import AkshareAdapter
            self.akshare = AkshareAdapter()
            adapters.append(self.akshare)
        except Exception as e:
            logger.exception("AkshareAdapter不可用")

        try:
            from ..adapters.baostock_adapter import BaostockAdapter
            self.baostock = BaostockAdapter()
            adapters.append(self.baostock)
        except Exception as e:
            logger.exception("BaostockAdapter不可用")

        if not adapters:
            raise RuntimeError("未检测到可用数据源适配器，请安装 akshare 或 baostock")

        self.fallback = FallbackManager(adapters)
        logger.info(f"DataProvider初始化完成，数据源: {[a.name for a in adapters]}")

    def get_stock_history(self, code: str, start_date: str, end_date: str,
                          adjust: str = "qfq", market_type: str = "A") -> pd.DataFrame:
        """获取股票历史K线"""
        return self.fallback.execute('get_stock_history', code, start_date, end_date, adjust, market_type, allow_empty_result=True)

    def get_index_stocks(self, index_code: str) -> List[str]:
        """获取指数成分股"""
        return self.fallback.execute('get_index_stocks', index_code)

    def get_stock_info(self, code: str, market_type: str = "A") -> Dict:
        """获取股票基本信息"""
        return self.fallback.execute('get_stock_info', code, market_type)

    def get_financial_data(self, code: str) -> Dict:
        """获取财务数据"""
        return self.fallback.execute('get_financial_data', code)

    # ========== akshare专有方法（无baostock备用）==========

    def get_board_stocks(self, board: str) -> List[str]:
        """获取板块股票列表（仅akshare支持）"""
        return self.akshare.get_board_stocks(board)

    def get_industry_list(self) -> pd.DataFrame:
        """获取行业板块列表（仅akshare支持）"""
        return self.akshare.get_industry_list()

    def get_industry_stocks(self, industry: str) -> List[str]:
        """获取行业成分股（仅akshare支持）"""
        return self.akshare.get_industry_stocks(industry)

    def get_concept_stocks(self, concept: str) -> List[str]:
        """获取概念板块成分股代码列表"""
        return self.akshare.get_concept_stocks(concept)

    def get_concept_stocks_detail(self, concept: str) -> List[Dict]:
        """获取概念板块成分股详细信息（含名称、价格等）"""
        return self.akshare.get_concept_stocks_detail(concept)

    def get_capital_flow(self, code: str) -> Dict:
        """获取资金流向（仅akshare支持）"""
        return self.akshare.get_capital_flow(code)

    def get_north_flow(self) -> pd.DataFrame:
        """获取北向资金（仅akshare支持）"""
        return self.akshare.get_north_flow()

    # ========== 状态管理 ==========

    def health_check(self) -> Dict:
        """健康检查"""
        return {
            'akshare': self.akshare.health_check(),
            'baostock': self.baostock.health_check(),
        }

    def get_status(self) -> Dict:
        """获取数据源状态"""
        return self.fallback.get_status()

    def reset_status(self):
        """重置数据源状态"""
        self.fallback.reset_status()


# 全局单例
_data_provider = None


def get_data_provider() -> DataProvider:
    """获取DataProvider单例"""
    global _data_provider
    if _data_provider is None:
        _data_provider = DataProvider()
    return _data_provider
