"""
技术指标模块单元测试
"""
import unittest

import numpy as np
import pandas as pd

from src.analysis.technical import TechnicalAnalyzer, SignalGenerator


class TestTechnicalAnalyzer(unittest.TestCase):
    """测试技术分析器"""

    def setUp(self):
        """准备测试数据"""
        # 创建模拟K线数据
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='B')

        self.df = pd.DataFrame({
            'trade_date': dates,
            'ts_code': ['000001.SZ'] * 100,
            'open': 10 + np.random.randn(100).cumsum() * 0.5,
            'high': 10 + np.random.randn(100).cumsum() * 0.5 + 1,
            'low': 10 + np.random.randn(100).cumsum() * 0.5 - 1,
            'close': 10 + np.random.randn(100).cumsum() * 0.5,
            'vol': np.random.randint(1000000, 10000000, 100),
        })

        # 确保 high >= close >= low
        self.df['high'] = self.df[['open', 'close']].max(axis=1) + abs(np.random.randn(100)) * 0.5
        self.df['low'] = self.df[['open', 'close']].min(axis=1) - abs(np.random.randn(100)) * 0.5

        self.analyzer = TechnicalAnalyzer()

    def test_calculate_ma(self):
        """测试移动平均线计算"""
        df = self.analyzer.calculate_ma(self.df.copy())

        # 检查是否计算了各个周期
        self.assertIn('ma5', df.columns)
        self.assertIn('ma10', df.columns)
        self.assertIn('ma20', df.columns)

        # 检查MA5的计算
        expected_ma5 = df['close'].rolling(window=5).mean()
        pd.testing.assert_series_equal(df['ma5'], expected_ma5)

    def test_calculate_macd(self):
        """测试MACD计算"""
        df = self.analyzer.calculate_macd(self.df.copy())

        self.assertIn('macd_dif', df.columns)
        self.assertIn('macd_dea', df.columns)
        self.assertIn('macd_bar', df.columns)

        # MACD柱状图 = 2 * (DIF - DEA)
        expected_bar = 2 * (df['macd_dif'] - df['macd_dea'])
        pd.testing.assert_series_equal(df['macd_bar'], expected_bar)

    def test_calculate_rsi(self):
        """测试RSI计算"""
        df = self.analyzer.calculate_rsi(self.df.copy())

        self.assertIn('rsi6', df.columns)
        self.assertIn('rsi12', df.columns)

        # RSI应在0-100范围内
        self.assertTrue(df['rsi6'].dropna().between(0, 100).all())
        self.assertTrue(df['rsi12'].dropna().between(0, 100).all())

    def test_calculate_boll(self):
        """测试布林带计算"""
        df = self.analyzer.calculate_boll(self.df.copy())

        self.assertIn('boll_upper', df.columns)
        self.assertIn('boll_mid', df.columns)
        self.assertIn('boll_lower', df.columns)

        # 上轨 > 中轨 > 下轨
        valid_data = df.dropna()
        self.assertTrue((valid_data['boll_upper'] >= valid_data['boll_mid']).all())
        self.assertTrue((valid_data['boll_mid'] >= valid_data['boll_lower']).all())

    def test_calculate_all(self):
        """测试计算所有指标"""
        df = self.analyzer.calculate_all(self.df.copy())

        # 检查是否包含所有指标
        expected_cols = ['ma5', 'ma20', 'macd_dif', 'kdj_k', 'rsi6', 'boll_upper', 'cci']
        for col in expected_cols:
            self.assertIn(col, df.columns)


class TestSignalGenerator(unittest.TestCase):
    """测试信号生成器"""

    def setUp(self):
        """准备测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50, freq='B')

        self.df = pd.DataFrame({
            'trade_date': dates,
            'ts_code': ['000001.SZ'] * 50,
            'open': 10 + np.cumsum(np.random.randn(50) * 0.1),
            'high': 10 + np.cumsum(np.random.randn(50) * 0.1) + 0.5,
            'low': 10 + np.cumsum(np.random.randn(50) * 0.1) - 0.5,
            'close': 10 + np.cumsum(np.random.randn(50) * 0.1),
            'vol': np.random.randint(1000000, 10000000, 50),
        })

        self.generator = SignalGenerator()

    def test_generate_ma_signals(self):
        """测试MA信号生成"""
        df = self.analyzer.calculate_all(self.df.copy())
        df = self.generator.generate_ma_signals(df)

        self.assertIn('ma_golden_cross', df.columns)
        self.assertIn('ma_death_cross', df.columns)
        self.assertIn('ma_bull_arrange', df.columns)

    def test_generate_macd_signals(self):
        """测试MACD信号生成"""
        df = self.analyzer.calculate_all(self.df.copy())
        df = self.generator.generate_macd_signals(df)

        self.assertIn('macd_golden_cross', df.columns)
        self.assertIn('macd_death_cross', df.columns)
        self.assertIn('macd_red', df.columns)


class TestBacktest(unittest.TestCase):
    """测试回测系统"""

    def setUp(self):
        """准备测试数据"""
        from src.analysis.backtest import BacktestEngine

        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')

        data = []
        price = 10.0
        for date in dates:
            change = np.random.normal(0, 0.02)
            price = price * (1 + change)

            data.append({
                'trade_date': date,
                'ts_code': '000001.SZ',
                'open': price * 0.99,
                'high': price * 1.02,
                'low': price * 0.98,
                'close': price,
                'vol': np.random.randint(1000000, 10000000),
            })

        self.df = pd.DataFrame(data)
        self.engine = BacktestEngine(initial_capital=1000000)

    def test_initialization(self):
        """测试回测引擎初始化"""
        self.assertEqual(self.engine.initial_capital, 1000000)
        self.assertEqual(self.engine.commission_rate, 0.0003)


if __name__ == '__main__':
    unittest.main()
