# 数据源适配器模块
from .base_adapter import BaseAdapter

try:
    from .akshare_adapter import AkshareAdapter
except Exception:
    AkshareAdapter = None

try:
    from .baostock_adapter import BaostockAdapter
except Exception:
    BaostockAdapter = None

__all__ = ['BaseAdapter', 'AkshareAdapter', 'BaostockAdapter']
