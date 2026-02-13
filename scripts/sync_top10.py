#!/usr/bin/env python3
"""同步10只热门股票历史数据"""
import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.collectors import AKShareCollector

async def main():
    collector = AKShareCollector()

    # 10只热门股票
    stocks = [
        '000001.SZ',  # 平安银行
        '000002.SZ',  # 万科A
        '000858.SZ',  # 五粮液
        '002594.SZ',  # 比亚迪
        '300750.SZ',  # 宁德时代
        '600519.SH',  # 贵州茅台
        '601318.SH',  # 中国平安
        '601012.SH',  # 隆基绿能
        '603288.SH',  # 海天味业
        '000568.SZ',  # 泸州老窖
    ]

    start = (datetime.now() - timedelta(days=252*3)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')

    success = 0
    for ts_code in stocks:
        try:
            symbol = ts_code.split('.')[0]
            print(f"正在同步 {ts_code}...")
            df = collector.get_daily_data(symbol, start, end)
            if not df.empty:
                await collector.save_to_supabase('stock_daily', df, ['ts_code', 'trade_date'])
                print(f"✅ {ts_code} 同步完成: {len(df)} 条")
                success += 1
            else:
                print(f"⚠️ {ts_code} 无数据")
        except Exception as e:
            print(f"❌ {ts_code} 失败: {e}")

        time.sleep(2)  # 降低请求频率

    print(f"\n完成: {success}/{len(stocks)} 只股票同步成功")

if __name__ == "__main__":
    asyncio.run(main())
