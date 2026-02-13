"""
A股技术分析模块
计算各种技术指标
"""
from typing import Optional, List

import numpy as np
import pandas as pd
from loguru import logger


class TechnicalAnalyzer:
    """技术分析器"""

    @staticmethod
    def calculate_ma(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 30, 60, 120, 250]) -> pd.DataFrame:
        """计算移动平均线"""
        for period in periods:
            df[f'ma{period}'] = df['close'].rolling(window=period).mean()
        return df

    @staticmethod
    def calculate_ema(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 30, 60]) -> pd.DataFrame:
        """计算指数移动平均线"""
        for period in periods:
            df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """计算 MACD 指标"""
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd_dif'] = exp1 - exp2
        df['macd_dea'] = df['macd_dif'].ewm(span=signal, adjust=False).mean()
        df['macd_bar'] = 2 * (df['macd_dif'] - df['macd_dea'])
        return df

    @staticmethod
    def calculate_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """计算 KDJ 指标"""
        low_list = df['low'].rolling(window=n, min_periods=n).min()
        high_list = df['high'].rolling(window=n, min_periods=n).max()
        rsv = (df['close'] - low_list) / (high_list - low_list) * 100

        df['kdj_k'] = rsv.ewm(alpha=1/m1, adjust=False).mean()
        df['kdj_d'] = df['kdj_k'].ewm(alpha=1/m2, adjust=False).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        return df

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, periods: List[int] = [6, 12, 24]) -> pd.DataFrame:
        """计算 RSI 指标"""
        for period in periods:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi{period}'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def calculate_boll(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """计算布林带"""
        df['boll_mid'] = df['close'].rolling(window=period).mean()
        df['boll_std'] = df['close'].rolling(window=period).std()
        df['boll_upper'] = df['boll_mid'] + (df['boll_std'] * std_dev)
        df['boll_lower'] = df['boll_mid'] - (df['boll_std'] * std_dev)
        df.drop('boll_std', axis=1, inplace=True)
        return df

    @staticmethod
    def calculate_cci(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算 CCI 指标"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mean_dev = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (tp - sma_tp) / (0.015 * mean_dev)
        return df

    @staticmethod
    def calculate_wr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算威廉指标"""
        high_highest = df['high'].rolling(window=period).max()
        low_lowest = df['low'].rolling(window=period).min()
        df['wr'] = (high_highest - df['close']) / (high_highest - low_lowest) * -100
        return df

    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
        """计算 OBV 指标"""
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['vol'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['vol'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        return df

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算 ATR 指标"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(period).mean()
        return df

    @staticmethod
    def calculate_vol_ma(df: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """计算成交量均线"""
        for period in periods:
            df[f'vol_ma{period}'] = df['vol'].rolling(window=period).mean()
        return df

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        df = df.copy()

        # 确保数据按日期排序
        df = df.sort_values('trade_date')

        # 计算各种指标
        df = self.calculate_ma(df)
        df = self.calculate_ema(df)
        df = self.calculate_macd(df)
        df = self.calculate_kdj(df)
        df = self.calculate_rsi(df)
        df = self.calculate_boll(df)
        df = self.calculate_cci(df)
        df = self.calculate_wr(df)
        df = self.calculate_obv(df)
        df = self.calculate_atr(df)
        df = self.calculate_vol_ma(df)

        return df


class SignalGenerator:
    """交易信号生成器"""

    def __init__(self):
        self.analyzer = TechnicalAnalyzer()

    def generate_ma_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成均线交易信号"""
        df = df.copy()

        # 金叉: 短期均线上穿长期均线
        df['ma_golden_cross'] = (df['ma5'] > df['ma20']) & (df['ma5'].shift(1) <= df['ma20'].shift(1))

        # 死叉: 短期均线下穿长期均线
        df['ma_death_cross'] = (df['ma5'] < df['ma20']) & (df['ma5'].shift(1) >= df['ma20'].shift(1))

        # 多头排列: 短期 > 中期 > 长期
        df['ma_bull_arrange'] = (df['ma5'] > df['ma20']) & (df['ma20'] > df['ma60'])

        # 空头排列: 短期 < 中期 < 长期
        df['ma_bear_arrange'] = (df['ma5'] < df['ma20']) & (df['ma20'] < df['ma60'])

        return df

    def generate_macd_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成 MACD 交易信号"""
        df = df.copy()

        # MACD 金叉
        df['macd_golden_cross'] = (df['macd_dif'] > df['macd_dea']) & (df['macd_dif'].shift(1) <= df['macd_dea'].shift(1))

        # MACD 死叉
        df['macd_death_cross'] = (df['macd_dif'] < df['macd_dea']) & (df['macd_dif'].shift(1) >= df['macd_dea'].shift(1))

        # MACD 红柱 (买入信号增强)
        df['macd_red'] = df['macd_bar'] > 0

        # MACD 绿柱 (卖出信号增强)
        df['macd_green'] = df['macd_bar'] < 0

        return df

    def generate_kdj_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成 KDJ 交易信号"""
        df = df.copy()

        # KDJ 金叉
        df['kdj_golden_cross'] = (df['kdj_k'] > df['kdj_d']) & (df['kdj_k'].shift(1) <= df['kdj_d'].shift(1))

        # KDJ 死叉
        df['kdj_death_cross'] = (df['kdj_k'] < df['kdj_d']) & (df['kdj_k'].shift(1) >= df['kdj_d'].shift(1))

        # 超卖 (J < 0)
        df['kdj_oversold'] = df['kdj_j'] < 0

        # 超买 (J > 100)
        df['kdj_overbought'] = df['kdj_j'] > 100

        return df

    def generate_rsi_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成 RSI 交易信号"""
        df = df.copy()

        # RSI 超卖
        df['rsi_oversold'] = df['rsi6'] < 20

        # RSI 超买
        df['rsi_overbought'] = df['rsi6'] > 80

        # RSI 金叉 (短期上穿长期)
        df['rsi_golden_cross'] = (df['rsi6'] > df['rsi24']) & (df['rsi6'].shift(1) <= df['rsi24'].shift(1))

        return df

    def generate_boll_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成布林带交易信号"""
        df = df.copy()

        # 触及上轨
        df['boll_touch_upper'] = df['close'] >= df['boll_upper']

        # 触及下轨
        df['boll_touch_lower'] = df['close'] <= df['boll_lower']

        # 突破中轨向上
        df['boll_break_up'] = (df['close'] > df['boll_mid']) & (df['close'].shift(1) <= df['boll_mid'].shift(1))

        # 突破中轨向下
        df['boll_break_down'] = (df['close'] < df['boll_mid']) & (df['close'].shift(1) >= df['boll_mid'].shift(1))

        return df

    def generate_composite_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成综合交易信号"""
        df = df.copy()

        # 买入信号: 多指标共振
        df['buy_signal'] = (
            df.get('ma_golden_cross', False) |
            (df.get('macd_golden_cross', False) & df.get('ma_bull_arrange', False)) |
            (df.get('kdj_oversold', False) & df.get('rsi_oversold', False)) |
            df.get('boll_touch_lower', False)
        )

        # 卖出信号
        df['sell_signal'] = (
            df.get('ma_death_cross', False) |
            (df.get('macd_death_cross', False) & df.get('ma_bear_arrange', False)) |
            (df.get('kdj_overbought', False) & df.get('rsi_overbought', False)) |
            df.get('boll_touch_upper', False)
        )

        return df

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """完整分析并生成所有信号"""
        # 先计算技术指标
        df = self.analyzer.calculate_all(df)

        # 生成交易信号
        df = self.generate_ma_signals(df)
        df = self.generate_macd_signals(df)
        df = self.generate_kdj_signals(df)
        df = self.generate_rsi_signals(df)
        df = self.generate_boll_signals(df)
        df = self.generate_composite_signals(df)

        return df
