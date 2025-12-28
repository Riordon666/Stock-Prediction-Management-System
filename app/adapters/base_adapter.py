# -*- coding: utf-8 -*-
"""
数据源适配器基类 - 老王说：所有数据源都得按这个规矩来！
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import pandas as pd


class BaseAdapter(ABC):
    """数据源适配器基类"""

    @abstractmethod
    def get_stock_history(self, code: str, start_date: str, end_date: str,
                          adjust: str = "qfq") -> pd.DataFrame:
        """获取股票历史K线

        Args:
            code: 股票代码，如 000001
            start_date: 开始日期，格式 20240101 或 2024-01-01
            end_date: 结束日期
            adjust: 复权类型 qfq前复权/hfq后复权/空字符串不复权

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, amount
        """
        pass

    @abstractmethod
    def get_index_stocks(self, index_code: str) -> List[str]:
        """获取指数成分股

        Args:
            index_code: 指数代码，如 000300(沪深300), 000905(中证500)

        Returns:
            股票代码列表
        """
        pass

    @abstractmethod
    def get_stock_info(self, code: str) -> Dict:
        """获取股票基本信息"""
        pass

    @abstractmethod
    def get_financial_data(self, code: str) -> Dict:
        """获取财务数据"""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """健康检查，返回数据源是否可用"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """数据源名称"""
        pass
