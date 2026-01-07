# -*- coding: utf-8 -*-
"""新闻数据获取模块

功能:
- 获取财联社电报新闻数据并缓存到本地，避免重复内容
- 提供获取最近新闻的接口
"""

import hashlib
import json
import logging
import os
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super(DateEncoder, self).default(obj)


class NewsFetcher:
    def __init__(self, save_dir: str = "data/news"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.last_fetch_time: Optional[datetime] = None
        self.news_hashes = set()
        self._load_existing_hashes()

    def _load_existing_hashes(self) -> None:
        try:
            today = datetime.now()
            for i in range(3):
                d = today - timedelta(days=i)
                filename = self.get_news_filename(d)
                if not os.path.exists(filename):
                    continue

                with open(filename, 'r', encoding='utf-8') as f:
                    try:
                        news_data = json.load(f)
                    except json.JSONDecodeError:
                        continue

                for item in (news_data or []):
                    h = item.get('hash')
                    if h:
                        self.news_hashes.add(h)
                        continue
                    content = item.get('content')
                    if content:
                        self.news_hashes.add(self._calculate_hash(content))

            logger.info("已加载 %s 条新闻哈希值", len(self.news_hashes))
        except Exception:
            logger.exception("加载现有新闻哈希值时出错")
            self.news_hashes = set()

    def _calculate_hash(self, content: Any) -> str:
        return hashlib.md5(str(content).encode('utf-8')).hexdigest()

    def get_news_filename(self, d: Optional[datetime] = None) -> str:
        if d is None:
            date_str = datetime.now().strftime('%Y%m%d')
        else:
            date_str = d.strftime('%Y%m%d')
        return os.path.join(self.save_dir, f"news_{date_str}.json")

    def fetch_and_save(self) -> bool:
        try:
            try:
                import akshare as ak
            except Exception as e:
                logger.warning("akshare 未安装或不可用，无法获取快讯: %s", e)
                return False

            now = datetime.now()
            logger.info("开始获取财联社电报数据")

            df = ak.stock_info_global_cls(symbol="全部")
            if df is None or df.empty:
                logger.warning("获取的财联社电报数据为空")
                return False

            total_count = 0
            new_count = 0
            news_list: List[Dict[str, Any]] = []

            for _, row in df.iterrows():
                total_count += 1

                content = str(row.get("内容", "") or "")
                if not content.strip():
                    continue

                content_hash = self._calculate_hash(content)
                if content_hash in self.news_hashes:
                    continue

                self.news_hashes.add(content_hash)
                new_count += 1

                pub_date = row.get("发布日期", "")
                if isinstance(pub_date, (datetime, date)):
                    pub_date = pub_date.isoformat()
                else:
                    pub_date = str(pub_date)

                pub_time = row.get("发布时间", "")
                if isinstance(pub_time, (datetime, date)):
                    pub_time = pub_time.isoformat()
                else:
                    pub_time = str(pub_time)

                news_item = {
                    "title": str(row.get("标题", "") or ""),
                    "content": content,
                    "date": pub_date,
                    "time": pub_time,
                    "datetime": f"{pub_date} {pub_time}",
                    "fetch_time": now.strftime('%Y-%m-%d %H:%M:%S'),
                    "hash": content_hash,
                }
                news_list.append(news_item)

            if not news_list:
                logger.info("没有新的新闻数据需要保存 (共检查 %s 条)", total_count)
                return True

            filename = self.get_news_filename()
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = []
                merged_news = (existing_data or []) + news_list
                merged_news.sort(key=lambda x: x.get('datetime', ''), reverse=True)
            else:
                merged_news = sorted(news_list, key=lambda x: x.get('datetime', ''), reverse=True)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(merged_news, f, ensure_ascii=False, indent=2, cls=DateEncoder)

            logger.info(
                "成功保存 %s 条新闻数据 (共检查 %s 条，过滤重复 %s 条)",
                new_count,
                total_count,
                total_count - new_count,
            )
            self.last_fetch_time = now
            return True
        except Exception:
            logger.exception("获取或保存新闻数据时出错")
            return False

    def get_latest_news(self, days: int = 1, limit: int = 50) -> List[Dict[str, Any]]:
        news_data: List[Dict[str, Any]] = []
        today = datetime.now()

        for i in range(max(1, int(days))):
            d = today - timedelta(days=i)
            filename = self.get_news_filename(d)
            if not os.path.exists(filename):
                continue
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    news_data.extend(data)
            except Exception:
                logger.exception("读取新闻文件失败: %s", filename)

        unique_news: Dict[str, Dict[str, Any]] = {}
        for item in news_data:
            item_hash = item.get('hash')
            if not item_hash and item.get('content'):
                item_hash = self._calculate_hash(item.get('content'))
            if item_hash and item_hash not in unique_news:
                unique_news[item_hash] = item

        deduplicated_news = list(unique_news.values())
        deduplicated_news.sort(key=lambda x: x.get('datetime', ''), reverse=True)
        return deduplicated_news[: max(1, int(limit))]


news_fetcher = NewsFetcher()


def fetch_news_task() -> None:
    logger.info("开始执行新闻获取任务")
    news_fetcher.fetch_and_save()
    logger.info("新闻获取任务完成，5分钟后自动获取新闻")


_scheduler_started = False
_scheduler_lock = threading.Lock()


def start_news_scheduler(interval_seconds: int = 300) -> None:
    global _scheduler_started
    with _scheduler_lock:
        if _scheduler_started:
            return
        _scheduler_started = True

    def _run_scheduler() -> None:
        while True:
            try:
                fetch_news_task()
                time.sleep(max(30, int(interval_seconds)))
            except Exception:
                logger.exception("新闻定时任务执行出错")
                time.sleep(60)

    t = threading.Thread(target=_run_scheduler, daemon=True)
    t.start()
    logger.info("新闻获取定时任务已启动")
