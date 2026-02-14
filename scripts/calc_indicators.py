#!/usr/bin/env python3
"""
技术指标计算脚本
"""
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.analysis.technical import TechnicalAnalyzer
from src.config import settings
from src.database.connection import get_supabase_client


def calc_indicators_for_stock(ts_code: str, days: int = 120):
    """计算单只股票的技术指标"""
    logger.info(f"计算 {ts_code} 的技术指标...")

    supabase = get_supabase_client()

    # 获取日线数据
    result = supabase.table("stock_daily").select("*").eq("ts_code", ts_code).order("trade_date", desc=False).limit(days).execute()

    if not result.data or len(result.data) < 30:
        logger.warning(f"{ts_code} 数据不足，跳过")
        return 0

    df = pd.DataFrame(result.data)

    # 计算技术指标
    analyzer = TechnicalAnalyzer()
    df = analyzer.calculate_all(df)

    # 选择需要的列
    indicator_cols = [
        "ts_code", "trade_date",
        "ma5", "ma10", "ma20", "ma30", "ma60", "ma120", "ma250",
        "macd_dif", "macd_dea", "macd_bar",
        "kdj_k", "kdj_d", "kdj_j",
        "rsi6", "rsi12", "rsi24",
        "boll_upper", "boll_mid", "boll_lower",
        "cci", "wr", "obv", "atr",
        "vol_ma5", "vol_ma10", "vol_ma20"
    ]

    # 只保留存在的列
    df = df[[col for col in indicator_cols if col in df.columns]]

    # 删除 NaN 值
    df = df.dropna()

    if df.empty:
        return 0

    # 保存到数据库
    records = df.to_dict('records')
    batch_size = 500

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table("technical_indicators").upsert(batch, on_conflict="ts_code,trade_date").execute()

    return len(df)


def calc_all_indicators():
    """计算所有股票的技术指标"""
    logger.info("开始计算技术指标...")

    supabase = get_supabase_client()

    # 获取所有上市股票
    stocks = supabase.table("stocks_info").select("ts_code").eq("list_status", "L").limit(10000).execute()

    total = 0
    for i, stock in enumerate(stocks.data):
        try:
            count = calc_indicators_for_stock(stock["ts_code"])
            total += count

            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1}/{len(stocks.data)} 只股票")

        except Exception as e:
            logger.error(f"计算 {stock['ts_code']} 失败: {e}")
            continue

    logger.info(f"技术指标计算完成，共 {total} 条记录")


def main():
    logger.add(settings.LOGS_DIR / f"indicators_{datetime.now().strftime('%Y%m%d')}.log")
    calc_all_indicators()


if __name__ == "__main__":
    main()
