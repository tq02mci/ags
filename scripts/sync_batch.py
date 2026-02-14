#!/usr/bin/env python3
"""
分批同步历史数据
可以指定批次范围和每批股票数量
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.config import settings
from src.data_collection.collectors import AKShareCollector
from src.database.connection import get_supabase_client

# 状态文件
STATE_FILE = Path(__file__).parent / ".batch_sync_state.json"


def load_state():
    """加载同步状态"""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"last_batch": 0, "completed": [], "failed": [], "total_stocks": 0}


def save_state(state):
    """保存同步状态"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)


async def sync_batch(start_idx: int, end_idx: int, batch_size: int = 100):
    """同步指定批次的数据"""
    state = load_state()

    supabase = get_supabase_client()
    collector = AKShareCollector()

    # 获取股票列表（先查询总数，再分页获取）
    logger.info("获取股票列表...")

    # 先获取总数
    count_result = supabase.table('stocks_info').select('*', count='exact').eq('list_status', 'L').execute()
    total_count = count_result.count if hasattr(count_result, 'count') else len(count_result.data)
    logger.info(f"数据库中共有 {total_count} 只上市股票")

    # 分页获取所有股票
    stocks = []
    page_size = 1000
    offset = 0

    while len(stocks) < total_count:
        logger.info(f"正在获取第 {offset+1} 到 {min(offset+page_size, total_count)} 只股票...")
        page_result = supabase.table('stocks_info')\
            .select('ts_code,list_status')\
            .eq('list_status', 'L')\
            .order('ts_code')\
            .limit(page_size)\
            .offset(offset)\
            .execute()

        page_data = page_result.data
        if not page_data or len(page_data) == 0:
            break

        stocks.extend(page_data)
        offset += len(page_data)

        if len(page_data) < page_size:
            break

    logger.info(f"总共获取到 {len(stocks)} 只股票")

    if not stocks:
        logger.error("没有获取到股票列表")
        return

    state["total_stocks"] = len(stocks)
    save_state(state)

    logger.info(f"总共 {len(stocks)} 只股票，同步批次 {start_idx} 到 {end_idx}")

    # 计算日期范围（最近3年）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)

    # 同步指定范围
    batch_stocks = stocks[start_idx:end_idx]
    success_count = 0
    fail_count = 0

    for i, stock in enumerate(batch_stocks):
        ts_code = stock['ts_code']
        symbol = ts_code.split('.')[0]
        current_idx = start_idx + i

        try:
            df = collector.get_daily_data(
                symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            if not df.empty:
                await collector.save_to_supabase(
                    'stock_daily',
                    df,
                    conflict_cols=['ts_code', 'trade_date']
                )
                success_count += 1
                state["completed"].append(ts_code)

            if (i + 1) % 10 == 0:
                logger.info(f"批次进度: {i + 1}/{len(batch_stocks)} (总进度: {current_idx + 1}/{len(stocks)})")
                save_state(state)

            # 限速
            await asyncio.sleep(0.3)

        except Exception as e:
            logger.error(f"同步 {ts_code} 失败: {e}")
            fail_count += 1
            state["failed"].append(ts_code)
            save_state(state)

    state["last_batch"] = end_idx
    save_state(state)

    logger.info(f"批次同步完成: 成功 {success_count}, 失败 {fail_count}")
    logger.info(f"总体进度: {end_idx}/{len(stocks)} ({end_idx / len(stocks) * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="分批同步历史数据")
    parser.add_argument("--start", type=int, default=0, help="起始索引")
    parser.add_argument("--end", type=int, default=None, help="结束索引（默认到末尾）")
    parser.add_argument("--batch-size", type=int, default=100, help="每批处理数量")

    args = parser.parse_args()

    # 设置日志
    logger.add(settings.LOGS_DIR / f"batch_sync_{datetime.now().strftime('%Y%m%d')}.log")

    logger.info(f"启动分批同步: 从索引 {args.start} 开始")

    asyncio.run(sync_batch(args.start, args.end or args.start + args.batch_size, args.batch_size))


if __name__ == "__main__":
    main()
