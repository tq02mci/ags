#!/usr/bin/env python3
"""
滚动前向验证脚本
用于模型校准和历史回测验证

用法:
    # 对单只股票进行滚动验证
    python scripts/rolling_validation.py --ts-code 000001.SZ --start-date 2025-01-01 --end-date 2025-02-01

    # 验证所有股票
    python scripts/rolling_validation.py --all-stocks --start-date 2025-01-01

    # 加载已有验证结果进行分析
    python scripts/rolling_validation.py --analyze --results-dir models/validation/
"""
import argparse
import json
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
from src.prediction.walk_forward_validation import WalkForwardValidator, ModelCalibrator


def run_single_stock_validation(ts_code: str, start_date: str, end_date: str, model_type: str = 'xgboost'):
    """对单只股票进行滚动验证"""
    logger.info(f"=" * 60)
    logger.info(f"开始滚动验证: {ts_code}")
    logger.info(f"验证区间: {start_date} ~ {end_date}")
    logger.info(f"=" * 60)

    # 创建验证器
    validator = WalkForwardValidator(
        model_type=model_type,
        train_days=252,    # 使用1年历史数据训练
        test_days=1,       # 预测T+1
        step_days=1,       # 每天滚动
        min_train_samples=60
    )

    # 运行验证
    results_df = validator.run_walk_forward_validation(ts_code, start_date, end_date)

    if results_df.empty:
        logger.error("验证结果为空")
        return None

    # 分析结果
    analysis = validator.analyze_results()

    logger.info("\n" + "=" * 60)
    logger.info("验证结果汇总")
    logger.info("=" * 60)
    logger.info(f"验证窗口数: {analysis['total_windows']}")
    logger.info(f"平均准确率: {analysis['avg_accuracy']:.3f} (±{analysis['std_accuracy']:.3f})")
    logger.info(f"平均精确率: {analysis['avg_precision']:.3f}")
    logger.info(f"平均召回率: {analysis['avg_recall']:.3f}")
    logger.info(f"平均F1分数: {analysis['avg_f1']:.3f}")
    logger.info(f"平均AUC: {analysis['avg_auc']:.3f}")
    logger.info(f"准确率趋势: {analysis['accuracy_trend']}")

    # 保存结果
    output_dir = validator.save_results()
    logger.info(f"\n结果已保存到: {output_dir}")

    # 模型校准
    if validator.predictions_history:
        calibrator = ModelCalibrator()

        logger.info("\n正在进行模型校准...")

        # 校准预测
        calibrated = calibrator.calibrate_predictions(validator.predictions_history)

        # 优化阈值
        optimal_threshold = calibrator.optimize_threshold(calibrated, metric='f1')

        # 生成校准报告
        calibration_report = calibrator.generate_calibration_report(calibrated)

        # 保存校准结果
        with open(output_dir / 'calibration_report.json', 'w') as f:
            json.dump({
                'optimal_threshold': optimal_threshold,
                'calibration_report': calibration_report
            }, f, indent=2, default=str)

        logger.info(f"最优预测阈值: {optimal_threshold:.3f}")

    # 绘制图表
    try:
        plot_path = output_dir / 'validation_results.png'
        validator.plot_results(str(plot_path))
    except Exception as e:
        logger.warning(f"绘图失败: {e}")

    return {
        'ts_code': ts_code,
        'analysis': analysis,
        'results_df': results_df,
        'output_dir': str(output_dir)
    }


def run_all_stocks_validation(start_date: str, end_date: str = None, max_stocks: int = 100):
    """对多只股票进行滚动验证"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    supabase = get_supabase_client()

    # 获取股票列表
    stocks = supabase.table("stocks_info")\
        .select("ts_code")\
        .eq("list_status", "L")\
        .limit(max_stocks)\
        .execute()

    logger.info(f"准备对 {len(stocks.data)} 只股票进行验证")

    all_results = []

    for i, stock in enumerate(stocks.data):
        ts_code = stock['ts_code']

        try:
            result = run_single_stock_validation(ts_code, start_date, end_date)
            if result:
                all_results.append({
                    'ts_code': ts_code,
                    'accuracy': result['analysis']['avg_accuracy'],
                    'f1': result['analysis']['avg_f1']
                })

            logger.info(f"\n进度: {i+1}/{len(stocks.data)} 完成\n")

        except Exception as e:
            logger.error(f"验证 {ts_code} 失败: {e}")
            continue

    # 汇总结果
    if all_results:
        summary_df = pd.DataFrame(all_results)

        logger.info("\n" + "=" * 60)
        logger.info("全市场验证汇总")
        logger.info("=" * 60)
        logger.info(f"平均准确率: {summary_df['accuracy'].mean():.3f}")
        logger.info(f"平均F1: {summary_df['f1'].mean():.3f}")
        logger.info(f"准确率中位数: {summary_df['accuracy'].median():.3f}")
        logger.info(f"最佳股票: {summary_df.loc[summary_df['accuracy'].idxmax(), 'ts_code']}")

        # 保存汇总
        summary_path = settings.MODEL_CHECKPOINT_DIR / 'validation' / 'summary_all_stocks.csv'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)

        return summary_df

    return None


def analyze_historical_results(results_dir: str):
    """分析历史验证结果"""
    results_path = Path(results_dir)

    if not results_path.exists():
        logger.error(f"结果目录不存在: {results_dir}")
        return

    # 读取所有验证结果
    metrics_files = list(results_path.glob('validation_metrics_*.csv'))
    pred_files = list(results_path.glob('predictions_history_*.csv'))

    if not metrics_files:
        logger.error("未找到验证结果文件")
        return

    logger.info(f"找到 {len(metrics_files)} 个验证结果文件")

    # 合并所有结果
    all_metrics = []
    for f in metrics_files:
        df = pd.read_csv(f)
        all_metrics.append(df)

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_metrics['date'] = pd.to_datetime(combined_metrics['date'])

    # 统计分析
    logger.info("\n" + "=" * 60)
    logger.info("历史验证结果分析")
    logger.info("=" * 60)

    logger.info(f"\n总体统计:")
    logger.info(f"  验证窗口数: {len(combined_metrics)}")
    logger.info(f"  日期范围: {combined_metrics['date'].min()} ~ {combined_metrics['date'].max()}")
    logger.info(f"  平均准确率: {combined_metrics['accuracy'].mean():.3f}")
    logger.info(f"  平均F1: {combined_metrics['f1'].mean():.3f}")
    logger.info(f"  准确率标准差: {combined_metrics['accuracy'].std():.3f}")

    # 时间序列分析
    logger.info(f"\n时间趋势:")
    early_acc = combined_metrics.nsmallest(int(len(combined_metrics) * 0.2), 'date')['accuracy'].mean()
    late_acc = combined_metrics.nlargest(int(len(combined_metrics) * 0.2), 'date')['accuracy'].mean()
    logger.info(f"  早期准确率: {early_acc:.3f}")
    logger.info(f"  近期准确率: {late_acc:.3f}")
    logger.info(f"  趋势: {'改善' if late_acc > early_acc else '下降'}")

    # 分布分析
    logger.info(f"\n准确率分布:")
    logger.info(f"  >0.6: {(combined_metrics['accuracy'] > 0.6).sum()} 次 ({(combined_metrics['accuracy'] > 0.6).mean()*100:.1f}%)")
    logger.info(f"  0.5-0.6: {((combined_metrics['accuracy'] >= 0.5) & (combined_metrics['accuracy'] <= 0.6)).sum()} 次")
    logger.info(f"  <0.5: {(combined_metrics['accuracy'] < 0.5).sum()} 次 ({(combined_metrics['accuracy'] < 0.5).mean()*100:.1f}%)")

    # 混淆矩阵
    logger.info(f"\n累计混淆矩阵:")
    tp = combined_metrics['true_positive'].sum()
    fp = combined_metrics['false_positive'].sum()
    tn = combined_metrics['true_negative'].sum()
    fn = combined_metrics['false_negative'].sum()
    logger.info(f"  True Positive:  {tp}")
    logger.info(f"  False Positive: {fp}")
    logger.info(f"  True Negative:  {tn}")
    logger.info(f"  False Negative: {fn}")

    # 保存分析结果
    analysis_output = results_path / 'historical_analysis.json'
    with open(analysis_output, 'w') as f:
        json.dump({
            'total_windows': len(combined_metrics),
            'avg_accuracy': combined_metrics['accuracy'].mean(),
            'avg_f1': combined_metrics['f1'].mean(),
            'accuracy_trend': 'improving' if late_acc > early_acc else 'declining',
            'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}
        }, f, indent=2, default=str)

    logger.info(f"\n分析结果已保存到: {analysis_output}")


def backtest_predictions(prediction_file: str):
    """使用预测结果进行回测"""
    pred_df = pd.read_csv(prediction_file)

    logger.info(f"\n预测回测分析")
    logger.info(f"预测记录数: {len(pred_df)}")

    # 基本准确率
    accuracy = (pred_df['predicted'] == pred_df['actual']).mean()
    logger.info(f"方向准确率: {accuracy:.3f}")

    # 按概率分桶的准确率
    pred_df['prob_bin'] = pd.cut(pred_df['probability'], bins=5, labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])
    bin_accuracy = pred_df.groupby('prob_bin').apply(lambda x: (x['predicted'] == x['actual']).mean())

    logger.info(f"\n按置信度的准确率:")
    for prob_bin, acc in bin_accuracy.items():
        count = pred_df[pred_df['prob_bin'] == prob_bin].shape[0]
        logger.info(f"  {prob_bin}: {acc:.3f} (n={count})")

    # 模拟交易收益
    pred_df['strategy_return'] = pred_df.apply(
        lambda row: row['close'] * 0.01 * (1 if row['predicted'] == 1 else -1)
        if row['predicted'] == row['actual'] else
        row['close'] * 0.01 * (-1 if row['predicted'] == 1 else 1),
        axis=1
    )

    total_return = pred_df['strategy_return'].sum()
    logger.info(f"\n模拟策略累计收益: {total_return:.2f}")


def main():
    parser = argparse.ArgumentParser(description="滚动前向验证工具")

    # 验证参数
    parser.add_argument("--ts-code", type=str, help="股票代码 (如: 000001.SZ)")
    parser.add_argument("--all-stocks", action="store_true", help="验证所有股票")
    parser.add_argument("--start-date", type=str, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--model-type", type=str, default="xgboost", choices=["xgboost", "lightgbm", "randomforest", "logistic"], help="模型类型")
    parser.add_argument("--max-stocks", type=int, default=100, help="最大股票数量")

    # 分析模式
    parser.add_argument("--analyze", action="store_true", help="分析已有验证结果")
    parser.add_argument("--results-dir", type=str, help="验证结果目录")

    # 回测模式
    parser.add_argument("--backtest", type=str, help="预测结果文件路径，进行回测分析")

    args = parser.parse_args()

    # 设置日志
    logger.add(settings.LOGS_DIR / f"validation_{datetime.now().strftime('%Y%m%d')}.log")

    # 设置默认日期
    if not args.start_date:
        # 默认从2025-01-01开始
        args.start_date = "2025-01-01"

    if not args.end_date:
        args.end_date = datetime.now().strftime('%Y-%m-%d')

    # 执行对应功能
    if args.backtest:
        backtest_predictions(args.backtest)
    elif args.analyze:
        if not args.results_dir:
            args.results_dir = settings.MODEL_CHECKPOINT_DIR / 'validation'
        analyze_historical_results(args.results_dir)
    elif args.all_stocks:
        run_all_stocks_validation(args.start_date, args.end_date, args.max_stocks)
    elif args.ts_code:
        run_single_stock_validation(args.ts_code, args.start_date, args.end_date, args.model_type)
    else:
        parser.print_help()
        logger.error("请指定 --ts-code 或 --all-stocks")
        sys.exit(1)


if __name__ == "__main__":
    main()
