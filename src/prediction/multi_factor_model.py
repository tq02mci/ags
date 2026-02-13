"""
多因子预测模型
整合行情数据、技术指标、新闻情感、资金流向等多维因子
"""
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.database.connection import get_supabase_client


class MultiFactorFeatureEngineer:
    """多因子特征工程"""

    def __init__(self):
        self.supabase = get_supabase_client()

    def get_price_features(self, ts_code: str, end_date: str, days: int = 60) -> pd.DataFrame:
        """获取价格相关特征"""
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')

        result = self.supabase.table("stock_daily")\
            .select("*")\
            .eq("ts_code", ts_code)\
            .gte("trade_date", start_date)\
            .lte("trade_date", end_date)\
            .order("trade_date")\
            .execute()

        if not result.data:
            return pd.DataFrame()

        df = pd.DataFrame(result.data)
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        # 价格动量特征
        for period in [5, 10, 20]:
            df[f'return_{period}d'] = df['close'].pct_change(period)
            df[f'volatility_{period}d'] = df['close'].pct_change().rolling(period).std() * np.sqrt(252)

        # 趋势特征
        df['price_ma5_ratio'] = df['close'] / df['close'].rolling(5).mean()
        df['price_ma20_ratio'] = df['close'] / df['close'].rolling(20).mean()

        return df

    def get_technical_features(self, ts_code: str, end_date: str, days: int = 60) -> pd.DataFrame:
        """获取技术指标特征"""
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')

        result = self.supabase.table("technical_indicators")\
            .select("*")\
            .eq("ts_code", ts_code)\
            .gte("trade_date", start_date)\
            .lte("trade_date", end_date)\
            .order("trade_date")\
            .execute()

        if not result.data:
            return pd.DataFrame()

        df = pd.DataFrame(result.data)
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        # MACD特征
        df['macd_signal'] = np.where(df['macd_bar'] > 0, 1, -1)
        df['macd_strength'] = abs(df['macd_bar']) / df['close']

        # RSI特征
        df['rsi_zone'] = pd.cut(df['rsi6'], bins=[0, 30, 70, 100], labels=[-1, 0, 1])

        # 布林带位置
        df['boll_position'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'])

        return df

    def get_sentiment_features(self, ts_code: str, end_date: str, days: int = 7) -> pd.DataFrame:
        """获取新闻情感特征"""
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')

        # 获取新闻数据
        result = self.supabase.table("stock_news")\
            .select("publish_time, sentiment_score, sentiment_label")\
            .eq("ts_code", ts_code)\
            .gte("publish_time", start_date)\
            .lte("publish_time", end_date + ' 23:59:59')\
            .execute()

        if not result.data:
            return pd.DataFrame()

        df = pd.DataFrame(result.data)
        df['publish_time'] = pd.to_datetime(df['publish_time'])
        df['trade_date'] = df['publish_time'].dt.date

        # 按日期聚合情感特征
        sentiment_daily = df.groupby('trade_date').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'sentiment_label': lambda x: (x == 'positive').sum() - (x == 'negative').sum()
        }).reset_index()

        sentiment_daily.columns = ['trade_date', 'sentiment_mean', 'sentiment_std',
                                   'news_count', 'sentiment_net']
        sentiment_daily['trade_date'] = pd.to_datetime(sentiment_daily['trade_date'])

        return sentiment_daily

    def get_money_flow_features(self, ts_code: str, end_date: str, days: int = 20) -> pd.DataFrame:
        """获取资金流向特征"""
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')

        result = self.supabase.table("money_flow")\
            .select("*")\
            .eq("ts_code", ts_code)\
            .gte("trade_date", start_date)\
            .lte("trade_date", end_date)\
            .order("trade_date")\
            .execute()

        if not result.data:
            return pd.DataFrame()

        df = pd.DataFrame(result.data)
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        # 主力资金趋势
        df['main_net_ma5'] = df['main_net_amount'].rolling(5).mean()
        df['main_net_ma10'] = df['main_net_amount'].rolling(10).mean()
        df['main_trend'] = np.where(df['main_net_ma5'] > df['main_net_ma10'], 1, -1)

        # 大单占比
        df['huge_ratio'] = (df['huge_buy'] + df['huge_sell']) / (df['close'] * df.get('vol', 1))

        return df

    def get_longhu_features(self, ts_code: str, end_date: str, days: int = 30) -> pd.DataFrame:
        """获取龙虎榜特征"""
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')

        result = self.supabase.table("longhu_bang")\
            .select("trade_date, net_amount, buy_amount, sell_amount")\
            .eq("ts_code", ts_code)\
            .gte("trade_date", start_date)\
            .lte("trade_date", end_date)\
            .execute()

        if not result.data:
            return pd.DataFrame()

        df = pd.DataFrame(result.data)
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        # 龙虎榜净买入累计
        df['longhu_net_cum'] = df['net_amount'].cumsum()
        df['longhu_count_7d'] = df['trade_date'].rolling('7D').count()

        return df

    def get_market_features(self, end_date: str, days: int = 20) -> pd.DataFrame:
        """获取市场情绪特征"""
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')

        # 获取全市场涨跌统计
        result = self.supabase.table("stock_daily")\
            .select("trade_date, pct_change")\
            .gte("trade_date", start_date)\
            .lte("trade_date", end_date)\
            .execute()

        if not result.data:
            return pd.DataFrame()

        df = pd.DataFrame(result.data)
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        # 市场涨跌统计
        market_stats = df.groupby('trade_date').agg({
            'pct_change': ['mean', 'std', lambda x: (x > 0).sum(), lambda x: (x < 0).sum()]
        }).reset_index()

        market_stats.columns = ['trade_date', 'market_return_mean', 'market_volatility',
                                'up_count', 'down_count']
        market_stats['advance_decline_ratio'] = market_stats['up_count'] / (market_stats['down_count'] + 1)

        return market_stats

    def create_features(self, ts_code: str, end_date: str) -> pd.DataFrame:
        """创建完整特征集"""
        # 获取各类特征
        price_df = self.get_price_features(ts_code, end_date)
        if price_df.empty:
            return pd.DataFrame()

        tech_df = self.get_technical_features(ts_code, end_date)
        sentiment_df = self.get_sentiment_features(ts_code, end_date)
        moneyflow_df = self.get_money_flow_features(ts_code, end_date)
        longhu_df = self.get_longhu_features(ts_code, end_date)
        market_df = self.get_market_features(end_date)

        # 合并特征
        df = price_df[['ts_code', 'trade_date', 'close', 'vol']].copy()

        # 合并价格特征
        price_features = [col for col in price_df.columns if col not in ['ts_code', 'trade_date', 'close', 'vol']]
        df = df.merge(price_df[['trade_date'] + price_features], on='trade_date', how='left')

        # 合并技术指标
        if not tech_df.empty:
            tech_features = ['trade_date', 'macd_signal', 'macd_strength', 'rsi_zone', 'boll_position']
            df = df.merge(tech_df[tech_features], on='trade_date', how='left')

        # 合并情感特征
        if not sentiment_df.empty:
            df = df.merge(sentiment_df, on='trade_date', how='left')
            df['sentiment_mean'] = df['sentiment_mean'].fillna(0)
            df['news_count'] = df['news_count'].fillna(0)

        # 合并资金流向
        if not moneyflow_df.empty:
            flow_features = ['trade_date', 'main_net_amount', 'main_trend', 'huge_ratio']
            df = df.merge(moneyflow_df[flow_features], on='trade_date', how='left')

        # 合并龙虎榜
        if not longhu_df.empty:
            df = df.merge(longhu_df[['trade_date', 'longhu_net_cum']], on='trade_date', how='left')
            df['has_longhu'] = df['longhu_net_cum'].notna().astype(int)
            df['longhu_net_cum'] = df['longhu_net_cum'].fillna(0)

        # 合并市场特征
        if not market_df.empty:
            df = df.merge(market_df[['trade_date', 'market_return_mean', 'advance_decline_ratio']],
                         on='trade_date', how='left')

        # 创建目标变量 (未来1天涨跌)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

        return df.dropna()


class MultiFactorPredictor:
    """多因子预测器"""

    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.feature_importance = {}

        # 特征列定义
        self.feature_cols = [
            # 价格特征
            'return_5d', 'return_10d', 'return_20d',
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'price_ma5_ratio', 'price_ma20_ratio',

            # 技术指标
            'macd_signal', 'macd_strength', 'rsi_zone', 'boll_position',

            # 情感特征
            'sentiment_mean', 'sentiment_std', 'news_count', 'sentiment_net',

            # 资金流向
            'main_net_amount', 'main_trend', 'huge_ratio',

            # 龙虎榜
            'longhu_net_cum', 'has_longhu',

            # 市场情绪
            'market_return_mean', 'advance_decline_ratio'
        ]

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备特征和标签"""
        # 过滤存在的列
        available_cols = [col for col in self.feature_cols if col in df.columns]

        if len(available_cols) < 5:
            raise ValueError(f"可用特征太少: {len(available_cols)}")

        X = df[available_cols].values
        y = df['target'].values

        return X, y, available_cols

    def train(self, df: pd.DataFrame) -> Dict:
        """训练模型"""
        X, y, available_cols = self.prepare_features(df)
        self.feature_cols = available_cols

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 根据模型类型选择算法
        if self.model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                self.model = XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            except ImportError:
                logger.warning("XGBoost 未安装，使用随机森林")
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'lightgbm':
            try:
                from lightgbm import LGBMClassifier
                self.model = LGBMClassifier(n_estimators=100, random_state=42)
            except ImportError:
                logger.warning("LightGBM 未安装，使用随机森林")
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        # 训练
        self.model.fit(X_scaled, y)

        # 计算特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(available_cols, self.model.feature_importances_))

        logger.info(f"模型训练完成，使用 {len(available_cols)} 个特征")

        return {
            'model_type': self.model_type,
            'features': available_cols,
            'samples': len(y),
            'positive_ratio': y.mean()
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path: str):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'created_at': datetime.now().isoformat()
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"模型已保存到 {path}")

    def load(self, path: str):
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        self.feature_importance = model_data.get('feature_importance', {})
        self.model_type = model_data.get('model_type', 'xgboost')

        logger.info(f"模型已从 {path} 加载")
