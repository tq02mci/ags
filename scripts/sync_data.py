#!/usr/bin/env python3
"""
数据同步脚本
"""
import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.config import settings
from src.data_collection.collectors import DataSyncManager, AKShareCollector


def setup_logging():
    """配置日志"""
    log_file = settings.LOGS_DIR / f"sync_{datetime.now().strftime('%Y%m%d')}.log"
    logger.add(log_file, rotation="10 MB", retention="7 days")


async def sync_stock_list():
    """同步股票列表"""
    logger.info("开始同步股票列表...")
    manager = DataSyncManager()
    count = await manager.akshare.sync_stock_list()
    logger.info(f"股票列表同步完成，共 {count} 只股票")
    return count


async def sync_daily_data():
    """同步日线数据"""
    logger.info("开始同步日线数据...")
    manager = DataSyncManager()
    count = await manager.akshare.sync_daily_data()
    logger.info(f"日线数据同步完成，共 {count} 条记录")
    return count


async def full_sync():
    """全量同步"""
    logger.info("开始全量数据同步...")
    manager = DataSyncManager()
    await manager.full_sync()
    logger.info("全量同步完成")


async def incremental_sync(days: int = 1):
    """增量同步"""
    logger.info(f"开始增量同步，最近 {days} 天...")
    manager = DataSyncManager()
    await manager.incremental_sync(days)
    logger.info("增量同步完成")


async def sync_single_stock(ts_code: str, days: int = 252 * 3):
    """同步单只股票历史数据"""
    from datetime import datetime, timedelta

    logger.info(f"开始同步 {ts_code} 历史数据，最近 {days} 天...")

    collector = AKShareCollector()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    symbol = ts_code.split('.')[0]
    df = collector.get_daily_data(symbol, start_date, end_date)

    if df.empty:
        logger.warning(f"{ts_code} 无数据")
        return 0

    # 保存到数据库
    count = await collector.save_to_supabase('stock_daily', df, conflict_cols=['ts_code', 'trade_date'])
    logger.info(f"{ts_code} 同步完成，共 {count} 条记录")
    return count


def main():
    parser = argparse.ArgumentParser(description="A股数据同步工具")
    parser.add_argument(
        "--type",
        choices=["stock_list", "daily", "full", "incremental"],
        default="daily",
        help="同步类型"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="增量同步天数"
    )
    parser.add_argument(
        "--stock",
        type=str,
        help="指定股票代码"
    )

    args = parser.parse_args()

    setup_logging()

    logger.info(f"启动数据同步，类型: {args.type}")

    try:
        if args.type == "stock_list":
            asyncio.run(sync_stock_list())
        elif args.type == "daily":
            asyncio.run(sync_daily_data())
        elif args.type == "full":
            asyncio.run(full_sync())
        elif args.type == "incremental":
            asyncio.run(incremental_sync(args.days))

        logger.info("数据同步任务完成")

    except Exception as e:
        logger.error(f"数据同步失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
