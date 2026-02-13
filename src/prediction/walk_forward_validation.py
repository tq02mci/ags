"""
滚动前向验证框架 (Walk-Forward Validation)
用于模型校准和回测
"""
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.config import settings
from src.database.connection import get_supabase_client
from src.prediction.multi_factor_model import MultiFactorFeatureEngineer, MultiFactorPredictor


@dataclass
class ValidationMetrics:
    """验证指标"""
    date: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    predictions_count: int
    prediction_mean: float
    prediction_std: float


class WalkForwardValidator:
    """滚动前向验证器"""

    def __init__(
        self,
        model_type: str = 'xgboost',
        train_days: int = 252,      # 训练窗口: 1年
        test_days: int = 5,          # 测试窗口: 5天
        step_days: int = 5,          # 步长: 5天
        min_train_samples: int = 100
    ):
        self.model_type = model_type
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.min_train_samples = min_train_samples

        self.supabase = get_supabase_client()
        self.feature_engineer = MultiFactorFeatureEngineer()

        # 验证结果存储
        self.validation_results: List[ValidationMetrics] = []
        self.predictions_history: List[Dict] = []

    def get_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表"""
        result = self.supabase.table("trade_calendar")\
            .select("cal_date")\
            .gte("cal_date", start_date)\
            .lte("cal_date", end_date)\
            .eq("is_open", True)\
            .order("cal_date")\
            .execute()

        return [d['cal_date'] for d in result.data]

    def run_single_validation(
        self,
        ts_code: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str
    ) -> Optional[ValidationMetrics]:
        """运行单次验证"""
        try:
            # 训练数据
            logger.info(f"训练: {train_start} ~ {train_end}")
            train_df = self.feature_engineer.create_features(ts_code, train_end)

            if train_df.empty or len(train_df) < self.min_train_samples:
                logger.warning(f"训练样本不足: {len(train_df)}")
                return None

            # 测试数据
            logger.info(f"测试: {test_start} ~ {test_end}")
            test_df = self.feature_engineer.create_features(ts_code, test_end)

            if test_df.empty:
                logger.warning("测试数据为空")
                return None

            # 只保留测试期间的数据
            test_df = test_df[
                (test_df['trade_date'] >= test_start) &
                (test_df['trade_date'] <= test_end)
            ]

            if test_df.empty:
                return None

            # 训练模型
            model = MultiFactorPredictor(model_type=self.model_type)
            model.train(train_df)

            # 预测
            X_test, y_true, _ = model.prepare_features(test_df)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model.model, 'predict_proba') else y_pred

            # 保存预测结果
            for i, row in test_df.iterrows():
                self.predictions_history.append({
                    'ts_code': ts_code,
                    'trade_date': row['trade_date'].strftime('%Y-%m-%d'),
                    'predict_date': test_start,
                    'predicted': int(y_pred[i] if i < len(y_pred) else y_pred[-1]),
                    'probability': float(y_proba[i] if i < len(y_proba) else y_proba[-1]),
                    'actual': int(row['target']),
                    'close': float(row['close']),
                    'is_correct': int(y_pred[i] if i < len(y_pred) else y_pred[-1]) == int(row['target'])
                })

            # 计算指标
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            metrics = ValidationMetrics(
                date=test_start,
                accuracy=accuracy_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred, zero_division=0),
                recall=recall_score(y_true, y_pred, zero_division=0),
                f1=f1_score(y_true, y_pred, zero_division=0),
                auc=roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else 0.5,
                true_positive=int(tp),
                false_positive=int(fp),
                true_negative=int(tn),
                false_negative=int(fn),
                predictions_count=len(y_pred),
                prediction_mean=float(y_pred.mean()),
                prediction_std=float(y_pred.std())
            )

            logger.info(f"验证结果: 准确率={metrics.accuracy:.3f}, F1={metrics.f1:.3f}")

            return metrics

        except Exception as e:
            logger.error(f"验证失败: {e}")
            return None

    def run_walk_forward_validation(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        运行滚动前向验证

        例如: 从 2025-01-01 开始
        - 用 2024-01-01 ~ 2024-12-31 训练，预测 2025-01-02
        - 用 2024-01-06 ~ 2025-01-06 训练，预测 2025-01-07
        - 以此类推...
        """
        logger.info(f"开始滚动验证: {ts_code} ({start_date} ~ {end_date})")

        # 获取交易日列表
        trade_dates = self.get_trade_dates(start_date, end_date)

        if len(trade_dates) < self.test_days:
            logger.error("交易日数量不足")
            return pd.DataFrame()

        # 滚动验证
        for i in range(0, len(trade_dates), self.step_days):
            # 确定窗口
            test_start_idx = i
            test_end_idx = min(i + self.test_days, len(trade_dates))

            if test_end_idx >= len(trade_dates):
                break

            test_start = trade_dates[test_start_idx]
            test_end = trade_dates[test_end_idx - 1]
            train_end = trade_dates[test_start_idx - 1] if test_start_idx > 0 else test_start
            train_start = (datetime.strptime(train_end, '%Y-%m-%d') - timedelta(days=self.train_days)).strftime('%Y-%m-%d')

            logger.info(f"\n验证窗口 {len(self.validation_results) + 1}:")

            # 运行验证
            metrics = self.run_single_validation(
                ts_code=ts_code,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )

            if metrics:
                self.validation_results.append(metrics)

        # 转换为 DataFrame
        if not self.validation_results:
            return pd.DataFrame()

        results_df = pd.DataFrame([asdict(m) for m in self.validation_results])

        logger.info(f"\n滚动验证完成: 共 {len(results_df)} 个窗口")
        logger.info(f"平均准确率: {results_df['accuracy'].mean():.3f}")
        logger.info(f"平均F1: {results_df['f1'].mean():.3f}")

        return results_df

    def analyze_results(self) -> Dict:
        """分析验证结果"""
        if not self.validation_results:
            return {}

        df = pd.DataFrame([asdict(m) for m in self.validation_results])

        # 整体统计
        analysis = {
            'total_windows': len(df),
            'date_range': f"{df['date'].min()} ~ {df['date'].max()}",
            'avg_accuracy': df['accuracy'].mean(),
            'std_accuracy': df['accuracy'].std(),
            'avg_precision': df['precision'].mean(),
            'avg_recall': df['recall'].mean(),
            'avg_f1': df['f1'].mean(),
            'avg_auc': df['auc'].mean(),

            # 预测分布
            'prediction_bias': df['prediction_mean'].mean(),  # 预测偏向
            'prediction_stability': df['prediction_std'].mean(),  # 预测稳定性

            # 准确性趋势
            'accuracy_trend': 'improving' if df['accuracy'].iloc[-10:].mean() > df['accuracy'].iloc[:10].mean() else 'stable',

            # 混淆矩阵累计
            'total_tp': df['true_positive'].sum(),
            'total_fp': df['false_positive'].sum(),
            'total_tn': df['true_negative'].sum(),
            'total_fn': df['false_negative'].sum(),
        }

        # 计算校准度 (Calibration)
        if self.predictions_history:
            pred_df = pd.DataFrame(self.predictions_history)
            pred_df['prob_bin'] = pd.cut(pred_df['probability'], bins=10)
            calibration = pred_df.groupby('prob_bin').agg({
                'actual': 'mean',
                'probability': 'mean'
            }).reset_index()
            calibration['calibration_error'] = abs(calibration['actual'] - calibration['probability'])
            analysis['calibration_error'] = calibration['calibration_error'].mean()

        return analysis

    def save_results(self, output_dir: str = None):
        """保存验证结果"""
        if output_dir is None:
            output_dir = settings.MODEL_CHECKPOINT_DIR / 'validation'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存指标
        if self.validation_results:
            metrics_df = pd.DataFrame([asdict(m) for m in self.validation_results])
            metrics_df.to_csv(output_dir / f'validation_metrics_{timestamp}.csv', index=False)

        # 保存预测历史
        if self.predictions_history:
            pred_df = pd.DataFrame(self.predictions_history)
            pred_df.to_csv(output_dir / f'predictions_history_{timestamp}.csv', index=False)

        # 保存分析结果
        analysis = self.analyze_results()
        with open(output_dir / f'analysis_{timestamp}.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        logger.info(f"验证结果已保存到 {output_dir}")

        return output_dir

    def plot_results(self, output_path: str = None):
        """可视化验证结果"""
        try:
            import matplotlib.pyplot as plt

            if not self.validation_results:
                logger.warning("没有验证结果可绘制")
                return

            df = pd.DataFrame([asdict(m) for m in self.validation_results])
            df['date'] = pd.to_datetime(df['date'])

            fig, axes = plt.subplots(3, 2, figsize=(14, 12))

            # 1. 准确率随时间变化
            axes[0, 0].plot(df['date'], df['accuracy'], marker='o', label='Accuracy')
            axes[0, 0].axhline(y=df['accuracy'].mean(), color='r', linestyle='--', label='Mean')
            axes[0, 0].set_title('Accuracy Over Time')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # 2. F1分数
            axes[0, 1].plot(df['date'], df['f1'], marker='o', color='green', label='F1 Score')
            axes[0, 1].axhline(y=df['f1'].mean(), color='r', linestyle='--', label='Mean')
            axes[0, 1].set_title('F1 Score Over Time')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # 3. 精确率和召回率
            axes[1, 0].plot(df['date'], df['precision'], marker='o', label='Precision')
            axes[1, 0].plot(df['date'], df['recall'], marker='s', label='Recall')
            axes[1, 0].set_title('Precision vs Recall')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            # 4. AUC
            axes[1, 1].plot(df['date'], df['auc'], marker='o', color='purple', label='AUC')
            axes[1, 1].axhline(y=0.5, color='r', linestyle='--', label='Random')
            axes[1, 1].set_title('AUC Over Time')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

            # 5. 预测分布
            if self.predictions_history:
                pred_df = pd.DataFrame(self.predictions_history)
                axes[2, 0].hist(pred_df['probability'], bins=20, edgecolor='black')
                axes[2, 0].set_title('Prediction Probability Distribution')
                axes[2, 0].set_xlabel('Probability')
                axes[2, 0].set_ylabel('Count')

            # 6. 混淆矩阵累计
            tn = df['true_negative'].sum()
            fp = df['false_positive'].sum()
            fn = df['false_negative'].sum()
            tp = df['true_positive'].sum()

            cm = np.array([[tn, fp], [fn, tp]])
            im = axes[2, 1].imshow(cm, cmap='Blues')
            axes[2, 1].set_title('Cumulative Confusion Matrix')
            axes[2, 1].set_xticks([0, 1])
            axes[2, 1].set_yticks([0, 1])
            axes[2, 1].set_xticklabels(['Pred 0', 'Pred 1'])
            axes[2, 1].set_yticklabels(['True 0', 'True 1'])

            # 添加数值标注
            for i in range(2):
                for j in range(2):
                    axes[2, 1].text(j, i, cm[i, j], ha='center', va='center', fontsize=12)

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"图表已保存到 {output_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib 未安装，跳过绘图")


class ModelCalibrator:
    """模型校准器 - 基于历史验证结果优化模型"""

    def __init__(self):
        self.supabase = get_supabase_client()

    def calibrate_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """
        校准预测概率
        使用 Platt Scaling 或 Isotonic Regression
        """
        try:
            from sklearn.calibration import CalibratedClassifierCV

            pred_df = pd.DataFrame(predictions)

            if len(pred_df) < 100:
                logger.warning("样本不足，跳过校准")
                return predictions

            # 计算校准后的概率
            # 简单方法: 根据历史准确率调整
            calibration_map = {}

            for prob_bin in np.arange(0, 1, 0.1):
                mask = (pred_df['probability'] >= prob_bin) & (pred_df['probability'] < prob_bin + 0.1)
                if mask.sum() > 0:
                    actual_rate = pred_df[mask]['actual'].mean()
                    calibration_map[prob_bin] = actual_rate

            # 应用校准
            calibrated = []
            for pred in predictions:
                prob = pred['probability']
                bin_start = int(prob * 10) / 10
                calibrated_prob = calibration_map.get(bin_start, prob)

                calibrated.append({
                    **pred,
                    'probability_raw': prob,
                    'probability': calibrated_prob,
                    'probability_confidence': 1 - abs(prob - calibrated_prob)
                })

            return calibrated

        except Exception as e:
            logger.error(f"校准失败: {e}")
            return predictions

    def optimize_threshold(self, predictions: List[Dict], metric: str = 'f1') -> float:
        """
        优化分类阈值
        """
        pred_df = pd.DataFrame(predictions)

        best_threshold = 0.5
        best_score = 0

        for threshold in np.arange(0.3, 0.8, 0.05):
            pred_df['pred_label'] = (pred_df['probability'] >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(pred_df['actual'], pred_df['pred_label'])
            elif metric == 'accuracy':
                score = accuracy_score(pred_df['actual'], pred_df['pred_label'])
            elif metric == 'precision':
                score = precision_score(pred_df['actual'], pred_df['pred_label'])
            else:
                score = f1_score(pred_df['actual'], pred_df['pred_label'])

            if score > best_score:
                best_score = score
                best_threshold = threshold

        logger.info(f"最优阈值: {best_threshold:.2f}, 对应{metric}: {best_score:.3f}")

        return best_threshold

    def generate_calibration_report(self, predictions: List[Dict]) -> Dict:
        """生成校准报告"""
        pred_df = pd.DataFrame(predictions)

        report = {
            'total_predictions': len(pred_df),
            'accuracy': accuracy_score(pred_df['actual'], pred_df['predicted']),
            'directional_accuracy': (
                (pred_df['predicted'] == pred_df['actual']).sum() / len(pred_df)
            ),
            'avg_confidence': pred_df['probability'].mean(),
            'confidence_std': pred_df['probability'].std(),
        }

        # 按预测概率分桶统计
        pred_df['prob_bin'] = pd.cut(pred_df['probability'], bins=5)
        bin_stats = pred_df.groupby('prob_bin').agg({
            'actual': 'mean',
            'predicted': 'count'
        }).reset_index()
        bin_stats.columns = ['probability_range', 'actual_rate', 'count']
        report['probability_bins'] = bin_stats.to_dict('records')

        return report
