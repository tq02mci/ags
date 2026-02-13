#!/usr/bin/env python3
"""同步10只热门股票历史数据 - 使用 baostock 作为备选"""
import asyncio
import sys
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import get_supabase_client
import pandas as pd

def get_daily_data_baostock(symbol, start_date, end_date):
    """使用 baostock 获取日线数据"""
    import baostock as bs

    # 登录
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return pd.DataFrame()

    # 获取数据
    rs = bs.query_history_k_data_plus(
        symbol,
        "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="3"  # 前复权
    )

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    bs.logout()

    if not data_list:
        return pd.DataFrame()

    df = pd.DataFrame(data_list, columns=rs.fields)
    df = df.rename(columns={
        'date': 'trade_date',
        'code': 'ts_code',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'preclose': 'pre_close',
        'volume': 'vol',
        'amount': 'amount',
        'turn': 'turnover',
        'pctChg': 'pct_change'
    })

    # 转换数据类型
    for col in ['open', 'high', 'low', 'close', 'pre_close', 'amount', 'turnover', 'pct_change']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')

    # 修复 ts_code 格式
    df['ts_code'] = df['ts_code'].str.replace('sh.', '').str.replace('sz.', '')
    df['ts_code'] = df['ts_code'].apply(lambda x: f"{x}.{'SH' if x.startswith('6') else 'SZ'}")

    return df

async def main():
    supabase = get_supabase_client()

    # 10只热门股票 (baostock 格式: sh.600519 或 sz.000001)
    stocks = [
        ('000001.SZ', 'sz.000001'),  # 平安银行
        ('000002.SZ', 'sz.000002'),  # 万科A
        ('000858.SZ', 'sz.000858'),  # 五粮液
        ('002594.SZ', 'sz.002594'),  # 比亚迪
        ('300750.SZ', 'sz.300750'),  # 宁德时代
        ('600519.SH', 'sh.600519'),  # 贵州茅台
        ('601318.SH', 'sh.601318'),  # 中国平安
        ('601012.SH', 'sh.601012'),  # 隆基绿能
        ('603288.SH', 'sh.603288'),  # 海天味业
        ('000568.SZ', 'sz.000568'),  # 泸州老窖
    ]

    start = (datetime.now() - timedelta(days=252*3)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')

    print(f"同步时间范围: {start} 至 {end}")
    print("=" * 50)

    success = 0
    for ts_code, bs_code in stocks:
        try:
            print(f"\n正在同步 {ts_code}...")

            # 只使用 baostock，避免 AKShare 的 IP 限制
            df = get_daily_data_baostock(bs_code, start, end)

            if not df.empty:
                # 保存到数据库
                records = df.replace({pd.NaT: None, float('nan'): None}).to_dict('records')
                supabase.table('stock_daily').upsert(records, on_conflict='ts_code,trade_date').execute()
                print(f"✅ {ts_code} 同步完成: {len(df)} 条")
                success += 1
            else:
                print(f"⚠️ {ts_code} 无数据")
        except Exception as e:
            print(f"❌ {ts_code} 失败: {e}")
            import traceback
            traceback.print_exc()

        # 随机延迟 5-10 秒，避免请求过快
        delay = random.uniform(5, 10)
        print(f"  等待 {delay:.1f} 秒...")
        time.sleep(delay)

    print(f"\n{'=' * 50}")
    print(f"完成: {success}/{len(stocks)} 只股票同步成功")

if __name__ == "__main__":
    asyncio.run(main())
