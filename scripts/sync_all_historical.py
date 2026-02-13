#!/usr/bin/env python3
"""
同步全部A股历史数据 - 使用 baostock (避免 AKShare IP限制)
支持断点续传，可在 Codespace 后台运行
"""
import asyncio
import sys
import time
import random
import json
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import get_supabase_client
from src.config import settings
import pandas as pd
import baostock as bs

# 状态文件，用于断点续传
STATE_FILE = Path(__file__).parent / ".sync_state.json"


def load_state():
    """加载同步状态"""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": [], "last_sync": None}


def save_state(state):
    """保存同步状态"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def get_daily_data_baostock(symbol, start_date, end_date):
    """使用 baostock 获取日线数据"""
    rs = bs.query_history_k_data_plus(
        symbol,
        "date,code,open,high,low,close,preclose,volume,amount,pctChg",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="3"
    )

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

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
        'pctChg': 'pct_change'
    })

    # 只保留数据库中存在的字段
    db_columns = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount', 'pct_change']
    df = df[[col for col in db_columns if col in df.columns]]

    # 转换数据类型
    for col in ['open', 'high', 'low', 'close', 'pre_close', 'amount', 'pct_change']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')

    # 修复 ts_code 格式
    df['ts_code'] = df['ts_code'].str.replace('sh.', '').str.replace('sz.', '')
    df['ts_code'] = df['ts_code'].apply(lambda x: f"{x}.{'SH' if x.startswith('6') else 'SZ'}")

    return df


async def sync_stock(supabase, ts_code, bs_code, start_date, end_date, state):
    """同步单只股票"""
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 同步 {ts_code}...", end=' ')

        df = get_daily_data_baostock(bs_code, start_date, end_date)

        if df.empty:
            print("无数据")
            return False

        # 分批保存（每批1000条）
        batch_size = 1000
        total_saved = 0
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            records = batch.replace({pd.NaT: None, float('nan'): None}).to_dict('records')
            supabase.table('stock_daily').upsert(records, on_conflict='ts_code,trade_date').execute()
            total_saved += len(records)

        print(f"✅ {total_saved} 条")

        # 更新状态
        state["completed"].append(ts_code)
        state["last_sync"] = datetime.now().isoformat()
        save_state(state)

        return True

    except Exception as e:
        print(f"❌ 失败: {e}")
        state["failed"].append({"ts_code": ts_code, "error": str(e)})
        save_state(state)
        return False


async def main():
    """主函数"""
    print("=" * 60)
    print("A股全量历史数据同步")
    print("=" * 60)

    # 加载状态
    state = load_state()
    print(f"\n已同步: {len(state['completed'])} 只, 失败: {len(state['failed'])} 只")

    # 获取所有股票列表
    supabase = get_supabase_client()
    result = supabase.table('stocks_info').select('ts_code').execute()
    all_stocks = [r['ts_code'] for r in result.data]

    print(f"股票总数: {len(all_stocks)} 只")

    # 过滤已完成的
    pending_stocks = [s for s in all_stocks if s not in state["completed"]]
    print(f"待同步: {len(pending_stocks)} 只")

    if not pending_stocks:
        print("\n✅ 所有股票已同步完成！")
        return

    # 时间范围：最近3年
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=252*3)).strftime('%Y-%m-%d')
    print(f"时间范围: {start_date} 至 {end_date}")
    print("=" * 60)

    # 登录 baostock
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return

    try:
        # 同步所有股票
        total = len(pending_stocks)
        for idx, ts_code in enumerate(pending_stocks, 1):
            print(f"\n[{idx}/{total}] ", end='')

            # 转换为 baostock 格式
            symbol = ts_code.split('.')[0]
            exchange = 'sh' if ts_code.endswith('.SH') else 'sz'
            bs_code = f"{exchange}.{symbol}"

            await sync_stock(supabase, ts_code, bs_code, start_date, end_date, state)

            # 随机延迟 3-8 秒，避免请求过快
            if idx < total:
                delay = random.uniform(3, 8)
                time.sleep(delay)

    finally:
        bs.logout()

    # 统计结果
    print("\n" + "=" * 60)
    print("同步完成统计:")
    print(f"  成功: {len(state['completed'])} 只")
    print(f"  失败: {len(state['failed'])} 只")

    if state['failed']:
        print("\n失败列表:")
        for item in state['failed']:
            print(f"  - {item['ts_code']}: {item['error']}")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
