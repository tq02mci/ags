#!/usr/bin/env python3
"""
数据质量检查工具
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.config import settings
from src.database.connection import get_supabase_client


class DataQualityChecker:
    """数据质量检查器"""

    def __init__(self):
        self.supabase = get_supabase_client()
        self.issues = []

    def check_stock_list(self) -> dict:
        """检查股票列表数据"""
        logger.info("检查股票列表数据...")

        result = self.supabase.table("stocks_info").select("*").limit(10000).execute()
        df = pd.DataFrame(result.data)

        issues = []
        stats = {
            "total_stocks": len(df),
            "with_name": df['name'].notna().sum(),
            "with_industry": df['industry'].notna().sum(),
            "with_list_date": df['list_date'].notna().sum(),
            "missing_name": df['name'].isna().sum(),
            "missing_industry": df['industry'].isna().sum(),
        }

        # 检查缺失值
        if stats["missing_name"] > 0:
            issues.append(f"发现 {stats['missing_name']} 只股票缺少名称")

        if stats["missing_industry"] > 0:
            issues.append(f"发现 {stats['missing_industry']} 只股票缺少行业信息")

        logger.info(f"股票列表统计: {stats}")

        return {
            "status": "ok" if not issues else "warning",
            "stats": stats,
            "issues": issues
        }

    def check_daily_data(self, days: int = 30) -> dict:
        """检查日线数据质量"""
        logger.info(f"检查最近 {days} 天日线数据...")

        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        result = self.supabase.table("stock_daily")\
            .select("*")\
            .gte("trade_date", start_date)\
            .execute()

        df = pd.DataFrame(result.data)

        if df.empty:
            return {
                "status": "error",
                "message": "未找到日线数据",
                "issues": ["最近 {} 天无日线数据".format(days)]
            }

        issues = []

        # 检查价格异常
        price_anomalies = df[
            (df['close'] <= 0) |
            (df['high'] < df['low']) |
            (df['close'] > df['high'] * 1.1) |
            (df['close'] < df['low'] * 0.9)
        ]

        if not price_anomalies.empty:
            issues.append(f"发现 {len(price_anomalies)} 条价格异常数据")

        # 检查成交量异常
        volume_anomalies = df[df['vol'] <= 0]
        if not volume_anomalies.empty:
            issues.append(f"发现 {len(volume_anomalies)} 条成交量异常数据")

        # 检查涨跌幅异常
        df['calculated_change'] = (df['close'] - df['pre_close']) / df['pre_close'] * 100
        df['change_diff'] = abs(df['calculated_change'] - df['pct_change'])
        change_anomalies = df[df['change_diff'] > 0.1]  # 差异超过0.1%

        if not change_anomalies.empty:
            issues.append(f"发现 {len(change_anomalies)} 条涨跌幅计算不一致")

        # 统计
        stats = {
            "total_records": len(df),
            "date_range": f"{df['trade_date'].min()} to {df['trade_date'].max()}",
            "unique_stocks": df['ts_code'].nunique(),
            "price_anomalies": len(price_anomalies),
            "volume_anomalies": len(volume_anomalies),
            "change_anomalies": len(change_anomalies)
        }

        logger.info(f"日线数据统计: {stats}")

        return {
            "status": "ok" if not issues else "warning",
            "stats": stats,
            "issues": issues
        }

    def check_technical_indicators(self, sample_size: int = 100) -> dict:
        """检查技术指标"""
        logger.info("检查技术指标...")

        # 随机抽样检查
        stocks = self.supabase.table("stocks_info").select("ts_code").limit(sample_size).execute()

        issues = []
        total_checked = 0
        missing_indicators = 0

        for stock in stocks.data:
            result = self.supabase.table("technical_indicators")\
                .select("*")\
                .eq("ts_code", stock['ts_code'])\
                .order("trade_date", desc=True)\
                .limit(1)\
                .execute()

            if result.data:
                total_checked += 1
                row = result.data[0]
                # 检查关键指标是否存在
                if not all(k in row for k in ['ma5', 'macd_bar', 'rsi6']):
                    missing_indicators += 1

        if missing_indicators > 0:
            issues.append(f"{missing_indicators}/{total_checked} 只股票缺少关键技术指标")

        stats = {
            "checked_stocks": total_checked,
            "missing_indicators": missing_indicators
        }

        return {
            "status": "ok" if not issues else "warning",
            "stats": stats,
            "issues": issues
        }

    def check_data_freshness(self) -> dict:
        """检查数据新鲜度"""
        logger.info("检查数据新鲜度...")

        issues = []

        # 检查最新交易日
        result = self.supabase.table("stock_daily")\
            .select("trade_date")\
            .order("trade_date", desc=True)\
            .limit(1)\
            .execute()

        if not result.data:
            return {
                "status": "error",
                "issues": ["无日线数据"]
            }

        latest_date = pd.to_datetime(result.data[0]['trade_date'])
        today = datetime.now()
        days_diff = (today - latest_date).days

        if days_diff > 3:
            issues.append(f"数据已落后 {days_diff} 天，最新数据日期: {latest_date.strftime('%Y-%m-%d')}")

        stats = {
            "latest_date": latest_date.strftime('%Y-%m-%d'),
            "days_behind": days_diff
        }

        return {
            "status": "ok" if not issues else "warning",
            "stats": stats,
            "issues": issues
        }

    def check_data_completeness(self, days: int = 30) -> dict:
        """检查数据完整性"""
        logger.info(f"检查最近 {days} 天数据完整性...")

        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        # 获取交易日历
        calendar = self.supabase.table("trade_calendar")\
            .select("cal_date")\
            .gte("cal_date", start_date)\
            .eq("is_open", True)\
            .execute()

        trade_dates = set(pd.to_datetime(d['cal_date']).strftime('%Y-%m-%d') for d in calendar.data)

        # 获取实际数据日期
        result = self.supabase.table("stock_daily")\
            .select("trade_date")\
            .gte("trade_date", start_date)\
            .execute()

        actual_dates = set(pd.to_datetime(d['trade_date']).strftime('%Y-%m-%d') for d in result.data)

        # 找出缺失的交易日
        missing_dates = trade_dates - actual_dates

        issues = []
        if missing_dates:
            issues.append(f"缺少 {len(missing_dates)} 个交易日数据: {sorted(missing_dates)[:5]}")

        stats = {
            "expected_dates": len(trade_dates),
            "actual_dates": len(actual_dates),
            "missing_dates": len(missing_dates)
        }

        return {
            "status": "ok" if not issues else "warning",
            "stats": stats,
            "issues": issues
        }

    def run_all_checks(self) -> dict:
        """运行所有检查"""
        logger.info("=" * 50)
        logger.info("开始数据质量检查")
        logger.info("=" * 50)

        results = {
            "stock_list": self.check_stock_list(),
            "daily_data": self.check_daily_data(),
            "technical_indicators": self.check_technical_indicators(),
            "data_freshness": self.check_data_freshness(),
            "data_completeness": self.check_data_completeness()
        }

        # 汇总
        all_issues = []
        for check_name, result in results.items():
            all_issues.extend(result.get('issues', []))

        summary = {
            "check_time": datetime.now().isoformat(),
            "overall_status": "error" if any(r['status'] == 'error' for r in results.values()) else \
                            "warning" if any(r['status'] == 'warning' for r in results.values()) else "ok",
            "total_issues": len(all_issues),
            "issues": all_issues,
            "details": results
        }

        logger.info("=" * 50)
        logger.info(f"检查完成: {summary['overall_status']}")
        logger.info(f"发现问题: {summary['total_issues']} 个")
        if all_issues:
            for issue in all_issues:
                logger.warning(f"  - {issue}")
        logger.info("=" * 50)

        return summary


def main():
    parser = argparse.ArgumentParser(description="数据质量检查工具")
    parser.add_argument(
        "--check",
        choices=["all", "stock_list", "daily", "indicators", "freshness", "completeness"],
        default="all",
        help="检查类型"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出结果到文件 (JSON格式)"
    )

    args = parser.parse_args()

    checker = DataQualityChecker()

    if args.check == "all":
        result = checker.run_all_checks()
    elif args.check == "stock_list":
        result = checker.check_stock_list()
    elif args.check == "daily":
        result = checker.check_daily_data()
    elif args.check == "indicators":
        result = checker.check_technical_indicators()
    elif args.check == "freshness":
        result = checker.check_data_freshness()
    elif args.check == "completeness":
        result = checker.check_data_completeness()

    # 输出结果
    import json
    print(json.dumps(result, indent=2, default=str))

    # 保存到文件
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"结果已保存到 {args.output}")

    # 如果有严重错误，返回非零退出码
    if result.get('status') == 'error' or result.get('overall_status') == 'error':
        sys.exit(1)


if __name__ == "__main__":
    main()
