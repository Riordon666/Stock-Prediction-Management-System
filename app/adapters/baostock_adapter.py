# -*- coding: utf-8 -*-
"""
baostock数据源适配器 - 老王说：akshare全挂了就靠你了！
注意：baostock数据是T+1的，没有实时数据
"""
import baostock as bs
import pandas as pd
from typing import List, Dict
from .base_adapter import BaseAdapter


class BaostockAdapter(BaseAdapter):
    """baostock数据源适配器"""

    def __init__(self):
        self._logged_in = False

    @property
    def name(self) -> str:
        return "baostock"

    def _ensure_login(self):
        """确保已登录"""
        if not self._logged_in:
            lg = bs.login()
            if lg.error_code != '0':
                raise Exception(f"baostock登录失败: {lg.error_msg}")
            self._logged_in = True

    def _convert_code(self, code: str) -> str:
        """转换股票代码格式：000001 -> sh.000001 或 sz.000001"""
        code = code.replace('.SH', '').replace('.SZ', '').replace('sh.', '').replace('sz.', '')
        if code.startswith('6'):
            return f"sh.{code}"
        return f"sz.{code}"

    def _format_date(self, date_str: str) -> str:
        """转换日期格式 20240101 -> 2024-01-01"""
        if len(date_str) == 8 and '-' not in date_str:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return date_str

    def get_stock_history(self, code: str, start_date: str, end_date: str,
                          adjust: str = "qfq") -> pd.DataFrame:
        """获取股票历史K线"""
        self._ensure_login()
        bs_code = self._convert_code(code)
        start_date = self._format_date(start_date)
        end_date = self._format_date(end_date)

        # 复权标志：1后复权 2前复权 3不复权
        adjust_map = {'hfq': '1', 'qfq': '2', '': '3'}
        adjustflag = adjust_map.get(adjust, '2')

        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjustflag
        )

        data_list = []
        while rs.error_code == '0' and rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        return df

    def get_index_stocks(self, index_code: str) -> List[str]:
        """获取指数成分股"""
        self._ensure_login()

        index_map = {
            '000300': bs.query_hs300_stocks,
            '000905': bs.query_zz500_stocks,
            '000016': bs.query_sz50_stocks,
        }

        query_func = index_map.get(index_code)
        if not query_func:
            return []

        rs = query_func()
        stocks = []
        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            code = row[1].replace('sh.', '').replace('sz.', '') if len(row) > 1 else ''
            if code:
                stocks.append(code)
        return stocks

    def get_stock_info(self, code: str) -> Dict:
        """获取股票基本信息"""
        self._ensure_login()
        bs_code = self._convert_code(code)

        rs = bs.query_stock_basic(code=bs_code)
        if rs.error_code == '0' and rs.next():
            return dict(zip(rs.fields, rs.get_row_data()))
        return {}

    def get_financial_data(self, code: str) -> Dict:
        """获取财务数据"""
        self._ensure_login()
        bs_code = self._convert_code(code)
        result = {}

        # 盈利能力
        try:
            rs = bs.query_profit_data(code=bs_code, year=2024, quarter=3)
            if rs.error_code == '0' and rs.next():
                result['profit'] = dict(zip(rs.fields, rs.get_row_data()))
        except Exception:
            pass

        # 成长能力
        try:
            rs = bs.query_growth_data(code=bs_code, year=2024, quarter=3)
            if rs.error_code == '0' and rs.next():
                result['growth'] = dict(zip(rs.fields, rs.get_row_data()))
        except Exception:
            pass

        return result

    def health_check(self) -> bool:
        """健康检查"""
        try:
            self._ensure_login()
            rs = bs.query_trade_dates(start_date="2024-01-01", end_date="2024-01-02")
            return rs.error_code == '0'
        except Exception:
            return False

    def __del__(self):
        if self._logged_in:
            try:
                bs.logout()
            except Exception:
                pass
