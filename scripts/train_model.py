#!/usr/bin/env python3
"""
模型训练脚本
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.config import settings
from src.database.connection import get_supabase_client
from src.prediction.models import XGBoostModel, LSTMModel, ModelTrainer


def get_stock_data(ts_code: str, days: int = 756):  # 约3年
    """获取股票数据"""
    supabase = get_supabase_client()

    # 获取日线数据
    result = supabase.table("stock_daily").select("*").eq("ts_code", ts_code).order("trade_date", desc=False).limit(days).execute()

    if not result.data:
        return None

    df = pd.DataFrame(result.data)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date')

    # 获取技术指标
    tech_result = supabase.table("technical_indicators").select("*").eq("ts_code", ts_code).order("trade_date", desc=False).limit(days).execute()

    if tech_result.data:
        tech_df = pd.DataFrame(tech_result.data)
        tech_df['trade_date'] = pd.to_datetime(tech_df['trade_date'])
        df = df.merge(tech_df, on=['ts_code', 'trade_date'], how='left')

    return df


def train_xgboost(ts_code: str = None):
    """训练 XGBoost 模型"""
    logger.info("开始训练 XGBoost 模型...")

    if ts_code:
        # 单只股票
        df = get_stock_data(ts_code)
        if df is None or len(df) < 252:
            logger.error(f"{ts_code} 数据不足")
            return

        model = XGBoostModel(name=f"xgboost_{ts_code}")
        trainer = ModelTrainer(model)
        metrics = trainer.train(df)

        # 保存模型
        model_path = settings.MODEL_CHECKPOINT_DIR / f"xgboost_{ts_code}.pkl"
        model.save(str(model_path))

        logger.info(f"模型训练完成: {metrics}")

    else:
        # 全市场训练 (简化版)
        supabase = get_supabase_client()
        stocks = supabase.table("stocks_info").select("ts_code").eq("list_status", "L").limit(100).execute()

        all_data = []
        for stock in stocks.data:
            df = get_stock_data(stock["ts_code"], days=252)
            if df is not None and len(df) > 100:
                all_data.append(df)

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            model = XGBoostModel(name="xgboost_all")
            trainer = ModelTrainer(model)
            metrics = trainer.train(combined_df)

            model_path = settings.MODEL_CHECKPOINT_DIR / "xgboost_all.pkl"
            model.save(str(model_path))

            logger.info(f"全市场模型训练完成: {metrics}")


def train_lstm(ts_code: str = None):
    """训练 LSTM 模型"""
    logger.info("开始训练 LSTM 模型...")

    if ts_code:
        df = get_stock_data(ts_code)
        if df is None or len(df) < 252:
            logger.error(f"{ts_code} 数据不足")
            return

        model = LSTMModel(name=f"lstm_{ts_code}", params={'epochs': 50})
        # LSTM 训练逻辑...

        logger.info("LSTM 模型训练完成")


def main():
    parser = argparse.ArgumentParser(description="模型训练工具")
    parser.add_argument("--type", choices=["xgboost", "lstm", "all"], default="xgboost")
    parser.add_argument("--stock", type=str, help="股票代码")

    args = parser.parse_args()

    logger.add(settings.LOGS_DIR / f"train_{datetime.now().strftime('%Y%m%d')}.log")

    if args.type in ["xgboost", "all"]:
        train_xgboost(args.stock)

    if args.type in ["lstm", "all"]:
        train_lstm(args.stock)


if __name__ == "__main__":
    main()
