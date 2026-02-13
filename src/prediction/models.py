"""
A股预测模型模块
包含 LSTM、XGBoost、LightGBM 等模型
"""
import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class BaseModel(ABC):
    """预测模型基类"""

    def __init__(self, name: str, params: Optional[Dict] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.scaler = None
        self.feature_cols = []
        self.is_trained = False
        self.training_info = {}

    @abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备特征"""
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        pass

    def save(self, path: str) -> None:
        """保存模型"""
        model_data = {
            'name': self.name,
            'params': self.params,
            'feature_cols': self.feature_cols,
            'is_trained': self.is_trained,
            'training_info': self.training_info,
            'scaler': self.scaler
        }

        if self.model is not None:
            model_data['model'] = self.model

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"模型已保存到 {path}")

    def load(self, path: str) -> None:
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.name = model_data['name']
        self.params = model_data['params']
        self.feature_cols = model_data['feature_cols']
        self.is_trained = model_data['is_trained']
        self.training_info = model_data['training_info']
        self.scaler = model_data.get('scaler')
        self.model = model_data.get('model')

        logger.info(f"模型已从 {path} 加载")


class FeatureEngineer:
    """特征工程"""

    @staticmethod
    def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """创建价格特征"""
        df = df.copy()

        # 价格变化
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['close'].diff()

        # 高低价特征
        df['high_low_pct'] = (df['high'] - df['low']) / df['low']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']

        # 波动率
        df['volatility_5'] = df['price_change'].rolling(window=5).std()
        df['volatility_20'] = df['price_change'].rolling(window=20).std()

        # 价格位置
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        return df

    @staticmethod
    def create_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """创建成交量特征"""
        df = df.copy()

        # 成交量变化
        df['vol_change'] = df['vol'].pct_change()
        df['vol_ma5'] = df['vol'].rolling(window=5).mean()
        df['vol_ma20'] = df['vol'].rolling(window=20).mean()

        # 成交量比率
        df['vol_ratio'] = df['vol'] / df['vol_ma5']

        # 量价配合
        df['price_vol_corr'] = df['close'].rolling(window=20).corr(df['vol'])

        return df

    @staticmethod
    def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """创建技术指标特征"""
        df = df.copy()

        # 均线差
        df['ma5_20_diff'] = df.get('ma5', df['close'].rolling(5).mean()) - df.get('ma20', df['close'].rolling(20).mean())
        df['ma20_60_diff'] = df.get('ma20', df['close'].rolling(20).mean()) - df.get('ma60', df['close'].rolling(60).mean())

        # MACD 特征
        if 'macd_bar' in df.columns:
            df['macd_sign'] = np.sign(df['macd_bar'])
            df['macd_change'] = df['macd_bar'].diff()

        # RSI 特征
        if 'rsi6' in df.columns:
            df['rsi6_norm'] = df['rsi6'] / 100

        # 布林带位置
        if 'boll_upper' in df.columns:
            df['boll_position'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'])

        return df

    @staticmethod
    def create_lag_features(df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """创建滞后特征"""
        df = df.copy()

        for lag in lags:
            df[f'close_lag{lag}'] = df['close'].shift(lag)
            df[f'vol_lag{lag}'] = df['vol'].shift(lag)
            df[f'return_lag{lag}'] = df['close'].pct_change(lag)

        return df

    @classmethod
    def create_all_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """创建所有特征"""
        df = df.copy()
        df = cls.create_price_features(df)
        df = cls.create_volume_features(df)
        df = cls.create_technical_features(df)
        df = cls.create_lag_features(df)
        return df


class LSTMModel(BaseModel):
    """LSTM 时序预测模型"""

    def __init__(self, name: str = "LSTM", params: Optional[Dict] = None):
        default_params = {
            'sequence_length': 20,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32
        }
        default_params.update(params or {})
        super().__init__(name, default_params)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备特征"""
        df = FeatureEngineer.create_all_features(df)

        # 选择特征列
        self.feature_cols = [
            'close', 'vol', 'price_change', 'high_low_pct',
            'vol_ratio', 'ma5_20_diff', 'boll_position'
        ]

        # 只保留存在的列
        self.feature_cols = [col for col in self.feature_cols if col in df.columns]

        return df

    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备序列数据"""
        seq_len = self.params['sequence_length']

        # 标准化
        data = df[self.feature_cols].values
        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(len(data_scaled) - seq_len):
            X.append(data_scaled[i:i + seq_len])
            # 预测下一天的收益率方向
            y.append(1 if df['close'].iloc[i + seq_len] > df['close'].iloc[i + seq_len - 1] else 0)

        return np.array(X), np.array(y)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """训练 LSTM 模型"""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            # 定义 LSTM 模型
            class LSTMNetwork(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, dropout):
                    super().__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers

                    self.lstm = nn.LSTM(
                        input_size, hidden_size, num_layers,
                        batch_first=True, dropout=dropout
                    )
                    self.fc = nn.Linear(hidden_size, 2)
                    self.dropout = nn.Dropout(dropout)

                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

                    out, _ = self.lstm(x, (h0, c0))
                    out = self.dropout(out[:, -1, :])
                    out = self.fc(out)
                    return out

            # 准备数据
            X_tensor = torch.FloatTensor(X_train)
            y_tensor = torch.LongTensor(y_train)
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True)

            # 初始化模型
            input_size = X_train.shape[2]
            self.model = LSTMNetwork(
                input_size,
                self.params['hidden_size'],
                self.params['num_layers'],
                self.params['dropout']
            )

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])

            # 训练
            self.model.train()
            for epoch in range(self.params['epochs']):
                total_loss = 0
                for X_batch, y_batch in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.params['epochs']}, Loss: {total_loss / len(dataloader):.4f}")

            self.is_trained = True
            self.training_info = {
                'trained_at': datetime.now().isoformat(),
                'samples': len(X_train),
                'epochs': self.params['epochs']
            }

        except ImportError:
            logger.error("PyTorch 未安装，无法训练 LSTM 模型")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        import torch

        self.model.eval()
        X_tensor = torch.FloatTensor(X)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        import torch
        import torch.nn.functional as F

        self.model.eval()
        X_tensor = torch.FloatTensor(X)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = F.softmax(outputs, dim=1)

        return probs.numpy()


class XGBoostModel(BaseModel):
    """XGBoost 预测模型"""

    def __init__(self, name: str = "XGBoost", params: Optional[Dict] = None):
        default_params = {
            'max_depth': 6,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        default_params.update(params or {})
        super().__init__(name, default_params)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备特征"""
        df = FeatureEngineer.create_all_features(df)

        self.feature_cols = [
            'close', 'open', 'high', 'low', 'vol', 'amount',
            'price_change', 'vol_ratio', 'ma5_20_diff',
            'close_lag1', 'close_lag5', 'return_lag5'
        ]

        self.feature_cols = [col for col in self.feature_cols if col in df.columns]

        return df

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """训练 XGBoost 模型"""
        from xgboost import XGBClassifier

        self.model = XGBClassifier(**self.params)
        self.model.fit(X_train, y_train)

        self.is_trained = True
        self.training_info = {
            'trained_at': datetime.now().isoformat(),
            'samples': len(X_train),
            'features': len(self.feature_cols)
        }

        logger.info(f"XGBoost 模型训练完成，特征数: {len(self.feature_cols)}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        return self.model.predict_proba(X)

    def feature_importance(self) -> pd.DataFrame:
        """特征重要性"""
        if not self.is_trained:
            raise ValueError("模型未训练")

        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)


class ModelEnsemble:
    """模型集成"""

    def __init__(self, models: Optional[List[BaseModel]] = None):
        self.models = models or []
        self.weights = []

    def add_model(self, model: BaseModel, weight: float = 1.0):
        """添加模型"""
        self.models.append(model)
        self.weights.append(weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """集成预测"""
        if not self.models:
            raise ValueError("没有可用的模型")

        predictions = []
        for model in self.models:
            if model.is_trained:
                pred = model.predict(X)
                predictions.append(pred)

        if not predictions:
            raise ValueError("没有训练好的模型")

        # 简单投票
        predictions = np.array(predictions)
        return np.round(predictions.mean(axis=0)).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """集成预测概率"""
        probabilities = []

        for model in self.models:
            if hasattr(model, 'predict_proba') and model.is_trained:
                prob = model.predict_proba(X)
                probabilities.append(prob)

        if not probabilities:
            raise ValueError("没有模型支持概率预测")

        # 加权平均
        probabilities = np.array(probabilities)
        weights = np.array(self.weights[:len(probabilities)])
        weights = weights / weights.sum()

        return np.average(probabilities, axis=0, weights=weights)


class ModelTrainer:
    """模型训练器"""

    def __init__(self, model: BaseModel):
        self.model = model
        self.metrics = {}

    def prepare_data(self, df: pd.DataFrame, target_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        # 准备特征
        df = self.model.prepare_features(df)

        # 创建目标变量 (未来N天的涨跌方向)
        df['target'] = (df['close'].shift(-target_horizon) > df['close']).astype(int)

        # 删除缺失值
        df = df.dropna()

        # 提取特征和目标
        X = df[self.model.feature_cols].values
        y = df['target'].values

        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
        """划分训练测试集"""
        split_idx = int(len(X) * (1 - test_size))
        return (
            X[:split_idx], X[split_idx:],
            y[:split_idx], y[split_idx:]
        )

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """训练模型"""
        logger.info(f"开始训练 {self.model.name} 模型...")

        # 准备数据
        X, y = self.prepare_data(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)

        logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

        # 训练
        self.model.train(X_train, y_train)

        # 评估
        self.metrics = self.evaluate(X_test, y_test)

        logger.info(f"模型训练完成，测试集准确率: {self.metrics['accuracy']:.4f}")

        return self.metrics

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估模型"""
        y_pred = self.model.predict(X_test)

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred) if len(set(y_test)) > 2 else None,
            'samples': len(y_test)
        }
