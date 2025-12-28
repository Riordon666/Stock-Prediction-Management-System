# -*- coding: utf-8 -*-
"""
故障转移管理器 - 老王说：接口挂了？自动给你换一个！
"""
import time
import logging
from typing import List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class FallbackManager:
    """故障转移管理器，支持多数据源自动切换"""

    def __init__(self, adapters: List, max_retries: int = 2, retry_delay: float = 0.5):
        """
        Args:
            adapters: 按优先级排序的适配器列表
            max_retries: 每个适配器最大重试次数
            retry_delay: 重试间隔（秒）
        """
        self.adapters = adapters
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # 适配器健康状态
        self._adapter_status = {a.name: True for a in adapters}
        # 失败计数
        self._fail_count = {a.name: 0 for a in adapters}

    def execute(self, method_name: str, *args, **kwargs) -> Any:
        """执行方法，自动故障转移

        Args:
            method_name: 要调用的方法名
            *args, **kwargs: 方法参数

        Returns:
            方法返回值

        Raises:
            Exception: 所有数据源均不可用时抛出
        """
        last_error = None
        tried_adapters = []

        for adapter in self.adapters:
            adapter_name = adapter.name

            # 跳过已标记为不可用的适配器（但失败次数超过阈值才跳过）
            if self._fail_count.get(adapter_name, 0) >= 5:
                continue

            # 检查适配器是否有该方法
            if not hasattr(adapter, method_name):
                continue

            tried_adapters.append(adapter_name)

            for retry in range(self.max_retries):
                try:
                    method = getattr(adapter, method_name)
                    result = method(*args, **kwargs)

                    # 检查结果是否有效
                    if self._is_valid_result(result):
                        # 成功，重置失败计数
                        self._fail_count[adapter_name] = 0
                        self._adapter_status[adapter_name] = True
                        return result

                except Exception as e:
                    last_error = e
                    logger.warning(f"[{adapter_name}] {method_name} 失败(重试{retry+1}/{self.max_retries}): {e}")

                    if retry < self.max_retries - 1:
                        time.sleep(self.retry_delay)

            # 该适配器所有重试都失败了
            self._fail_count[adapter_name] = self._fail_count.get(adapter_name, 0) + 1
            logger.warning(f"[{adapter_name}] 失败次数: {self._fail_count[adapter_name]}")

        # 所有适配器都失败了
        error_msg = f"所有数据源均不可用 (尝试了: {tried_adapters})"
        if last_error:
            error_msg += f", 最后错误: {last_error}"
        raise Exception(error_msg)

    def _is_valid_result(self, result: Any) -> bool:
        """检查结果是否有效"""
        if result is None:
            return False
        if isinstance(result, pd.DataFrame) and result.empty:
            return False
        if isinstance(result, (list, dict)) and len(result) == 0:
            return False
        return True

    def reset_status(self):
        """重置所有适配器状态"""
        for adapter in self.adapters:
            self._adapter_status[adapter.name] = True
            self._fail_count[adapter.name] = 0

    def get_status(self) -> dict:
        """获取适配器状态"""
        return {
            'status': self._adapter_status.copy(),
            'fail_count': self._fail_count.copy()
        }
