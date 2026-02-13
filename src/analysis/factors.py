"""
A股多因子模型
包含估值、成长、质量、动量、情绪等因子
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class FactorCalculator:
    """因子计算器"""

    @staticmethod
    def calculate_valuation_factors(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算估值因子
        需要: close, total_mv, eps, bps, revenue_ps
        """
        df = df.copy()

        # PE (市盈率)
        if 'eps' in df.columns:
            df['pe'] = df['close'] / df['eps']
            df['pe_ttm'] = df['close'] / df.get('eps_ttm', df['eps'])

        # PB (市净率)
        if 'bps' in df.columns:
            df['pb'] = df['close'] / df['bps']

        # PS (市销率)
        if 'revenue_ps' in df.columns and df['revenue_ps'].notna().any():
            df['ps'] = df['close'] / df['revenue_ps']

        # PCF (市现率)
        if 'ocfps' in df.columns and df['ocfps'].notna().any():
            df['pcf'] = df['close'] / df['ocfps']

        # EV/EBITDA (企业价值倍数)
        if all(col in df.columns for col in ['total_mv', 'total_liab', 'cash', 'ebitda']):
            df['ev'] = df['total_mv'] + df['total_liab'] - df['cash']
            df['ev_ebitda'] = df['ev'] / df['ebitda']

        # 股息率
        if 'dividend_yield' in df.columns:
            df['dividend_yield'] = df['dividend_yield']

        return df

    @staticmethod
    def calculate_growth_factors(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成长因子
        需要: 多期财务数据
        """
        df = df.copy()

        # 营收增长率
        if 'total_revenue' in df.columns:
            df['revenue_growth'] = df['total_revenue'].pct_change(4)  # 季度同比
            df['revenue_growth_yoy'] = df['total_revenue'].pct_change(12)  # 年度同比

        # 净利润增长率
        if 'net_income' in df.columns:
            df['profit_growth'] = df['net_income'].pct_change(4)
            df['profit_growth_yoy'] = df['net_income'].pct_change(12)

        # 扣非净利润增长率
        if 'profit_dedt' in df.columns:
            df['profit_dedt_growth'] = df['profit_dedt'].pct_change(4)

        # EPS增长率
        if 'eps' in df.columns:
            df['eps_growth'] = df['eps'].pct_change(4)

        # 总资产增长率
        if 'total_assets' in df.columns:
            df['assets_growth'] = df['total_assets'].pct_change(4)

        # 净资产增长率
        if 'total_hldr_eqy_exc_min_int' in df.columns:
            df['equity_growth'] = df['total_hldr_eqy_exc_min_int'].pct_change(4)

        return df

    @staticmethod
    def calculate_quality_factors(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算质量因子
        """
        df = df.copy()

        # ROE (净资产收益率)
        if all(col in df.columns for col in ['net_income', 'total_hldr_eqy_exc_min_int']):
            df['roe'] = df['net_income'] / df['total_hldr_eqy_exc_min_int']
            df['roe_ttm'] = df['net_income'].rolling(4).sum() / df['total_hldr_eqy_exc_min_int']

        # ROA (总资产收益率)
        if all(col in df.columns for col in ['net_income', 'total_assets']):
            df['roa'] = df['net_income'] / df['total_assets']

        # ROIC (投入资本回报率)
        if all(col in df.columns for col in ['ebit', 'total_assets', 'current_liab', 'cash']):
            invested_capital = df['total_assets'] - df['current_liab'] - df['cash']
            df['roic'] = df['ebit'] * (1 - 0.25) / invested_capital  # 假设税率25%

        # 毛利率
        if all(col in df.columns for col in ['gross_profit', 'total_revenue']):
            df['gross_margin'] = df['gross_profit'] / df['total_revenue']

        # 净利率
        if all(col in df.columns for col in ['net_income', 'total_revenue']):
            df['net_margin'] = df['net_income'] / df['total_revenue']

        # 营业利润率
        if all(col in df.columns for col in ['oper_profit', 'total_revenue']):
            df['oper_margin'] = df['oper_profit'] / df['total_revenue']

        # 资产负债率
        if all(col in df.columns for col in ['total_liab', 'total_assets']):
            df['debt_to_assets'] = df['total_liab'] / df['total_assets']

        # 流动比率
        if all(col in df.columns for col in ['total_current_assets', 'total_current_liab']):
            df['current_ratio'] = df['total_current_assets'] / df['total_current_liab']

        # 速动比率
        if all(col in df.columns for col in ['total_current_assets', 'inventories', 'total_current_liab']):
            df['quick_ratio'] = (df['total_current_assets'] - df['inventories']) / df['total_current_liab']

        return df

    @staticmethod
    def calculate_momentum_factors(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算动量因子
        需要: close
        """
        df = df.copy()

        # 价格动量
        for period in [5, 10, 20, 60, 120]:
            df[f'return_{period}d'] = df['close'].pct_change(period)

        # 收益趋势 (过去N天平均收益率)
        for period in [5, 10, 20]:
            df[f'return_avg_{period}d'] = df['close'].pct_change().rolling(period).mean()

        # 收益波动率
        for period in [20, 60]:
            df[f'volatility_{period}d'] = df['close'].pct_change().rolling(period).std() * np.sqrt(252)

        # 最大回撤
        df['cummax'] = df['close'].cummax()
        df['drawdown'] = (df['close'] - df['cummax']) / df['cummax']
        df['max_drawdown_20d'] = df['drawdown'].rolling(20).min()

        # 价格位置 (当前价格在近期高低点之间的位置)
        df['high_20d'] = df['close'].rolling(20).max()
        df['low_20d'] = df['close'].rolling(20).min()
        df['price_position'] = (df['close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'])

        return df

    @staticmethod
    def calculate_sentiment_factors(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算情绪因子
        """
        df = df.copy()

        # 换手率
        if 'vol' in df.columns and 'float_shares' in df.columns:
            df['turnover'] = df['vol'] / df['float_shares']
            df['turnover_ma5'] = df['turnover'].rolling(5).mean()
            df['turnover_ratio'] = df['turnover'] / df['turnover_ma5']

        # 量比 (当日成交量/过去5日平均)
        if 'vol' in df.columns:
            df['vol_ma5'] = df['vol'].rolling(5).mean()
            df['volume_ratio'] = df['vol'] / df['vol_ma5']

        # 振幅
        df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1)
        df['amplitude_ma5'] = df['amplitude'].rolling(5).mean()

        # 涨停/跌停次数统计
        df['is_limit_up'] = df['pct_change'] >= 9.9
        df['is_limit_down'] = df['pct_change'] <= -9.9
        df['limit_up_count_20d'] = df['is_limit_up'].rolling(20).sum()
        df['limit_down_count_20d'] = df['is_limit_down'].rolling(20).sum()

        return df

    @staticmethod
    def calculate_technical_factors(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术因子
        """
        df = df.copy()

        # 均线偏离度
        for ma_period in [5, 10, 20, 60]:
            ma_col = f'ma{ma_period}'
            if ma_col in df.columns:
                df[f'{ma_col}_dev'] = (df['close'] - df[ma_col]) / df[ma_col]

        # 均线交叉
        if all(col in df.columns for col in ['ma5', 'ma20']):
            df['ma5_above_ma20'] = (df['ma5'] > df['ma20']).astype(int)
            df['ma_cross'] = df['ma5_above_ma20'].diff()

        # MACD强度
        if 'macd_bar' in df.columns:
            df['macd_strength'] = abs(df['macd_bar']) / df['close']
            df['macd_direction'] = np.sign(df['macd_bar'])

        # RSI位置
        if 'rsi6' in df.columns:
            df['rsi_zone'] = pd.cut(df['rsi6'], bins=[0, 30, 70, 100], labels=['oversold', 'neutral', 'overbought'])

        # 布林带位置
        if all(col in df.columns for col in ['boll_upper', 'boll_lower', 'close']):
            df['boll_position'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'])
            df['boll_width'] = (df['boll_upper'] - df['boll_lower']) / df['boll_mid']

        return df

    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有因子"""
        df = df.copy()

        df = self.calculate_valuation_factors(df)
        df = self.calculate_growth_factors(df)
        df = self.calculate_quality_factors(df)
        df = self.calculate_momentum_factors(df)
        df = self.calculate_sentiment_factors(df)
        df = self.calculate_technical_factors(df)

        return df


class FactorScorer:
    """因子评分器"""

    def __init__(self):
        self.factor_weights = {
            'valuation': 0.2,
            'growth': 0.2,
            'quality': 0.2,
            'momentum': 0.2,
            'sentiment': 0.2
        }

    def rank_factor(self, series: pd.Series, ascending: bool = True) -> pd.Series:
        """因子排名 (0-1标准化)"""
        rank = series.rank(ascending=ascending, pct=True)
        return rank.fillna(0.5)

    def score_valuation(self, df: pd.DataFrame) -> pd.Series:
        """
        估值因子评分 (低估值得分高)
        """
        scores = pd.Series(0.5, index=df.index)

        if 'pe' in df.columns:
            scores += self.rank_factor(df['pe'], ascending=True) * 0.3
        if 'pb' in df.columns:
            scores += self.rank_factor(df['pb'], ascending=True) * 0.3
        if 'ps' in df.columns:
            scores += self.rank_factor(df['ps'], ascending=True) * 0.2
        if 'dividend_yield' in df.columns:
            scores += self.rank_factor(df['dividend_yield'], ascending=False) * 0.2

        return scores

    def score_growth(self, df: pd.DataFrame) -> pd.Series:
        """成长因子评分"""
        scores = pd.Series(0.5, index=df.index)

        if 'revenue_growth' in df.columns:
            scores += self.rank_factor(df['revenue_growth'], ascending=False) * 0.3
        if 'profit_growth' in df.columns:
            scores += self.rank_factor(df['profit_growth'], ascending=False) * 0.3
        if 'eps_growth' in df.columns:
            scores += self.rank_factor(df['eps_growth'], ascending=False) * 0.2
        if 'assets_growth' in df.columns:
            scores += self.rank_factor(df['assets_growth'], ascending=False) * 0.2

        return scores

    def score_quality(self, df: pd.DataFrame) -> pd.Series:
        """质量因子评分"""
        scores = pd.Series(0.5, index=df.index)

        if 'roe' in df.columns:
            scores += self.rank_factor(df['roe'], ascending=False) * 0.25
        if 'roa' in df.columns:
            scores += self.rank_factor(df['roa'], ascending=False) * 0.25
        if 'gross_margin' in df.columns:
            scores += self.rank_factor(df['gross_margin'], ascending=False) * 0.25
        if 'debt_to_assets' in df.columns:
            scores += self.rank_factor(df['debt_to_assets'], ascending=True) * 0.25

        return scores

    def score_momentum(self, df: pd.DataFrame) -> pd.Series:
        """动量因子评分"""
        scores = pd.Series(0.5, index=df.index)

        if 'return_20d' in df.columns:
            scores += self.rank_factor(df['return_20d'], ascending=False) * 0.3
        if 'return_60d' in df.columns:
            scores += self.rank_factor(df['return_60d'], ascending=False) * 0.3
        if 'volatility_20d' in df.columns:
            scores += self.rank_factor(df['volatility_20d'], ascending=True) * 0.2
        if 'max_drawdown_20d' in df.columns:
            scores += self.rank_factor(df['max_drawdown_20d'], ascending=True) * 0.2

        return scores

    def score_sentiment(self, df: pd.DataFrame) -> pd.Series:
        """情绪因子评分"""
        scores = pd.Series(0.5, index=df.index)

        if 'turnover_ratio' in df.columns:
            # 量比适中较好 (不过高不过低)
            turnover_score = 1 - abs(df['turnover_ratio'] - 1)
            scores += turnover_score.fillna(0.5) * 0.3

        if 'volume_ratio' in df.columns:
            scores += self.rank_factor(df['volume_ratio'], ascending=False) * 0.3

        if 'limit_up_count_20d' in df.columns:
            scores += self.rank_factor(df['limit_up_count_20d'], ascending=False) * 0.2

        if 'amplitude' in df.columns:
            scores += self.rank_factor(df['amplitude'], ascending=False) * 0.2

        return scores

    def calculate_total_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算综合评分"""
        df = df.copy()

        # 各维度评分
        df['score_valuation'] = self.score_valuation(df)
        df['score_growth'] = self.score_growth(df)
        df['score_quality'] = self.score_quality(df)
        df['score_momentum'] = self.score_momentum(df)
        df['score_sentiment'] = self.score_sentiment(df)

        # 综合评分 (加权平均)
        df['total_score'] = (
            df['score_valuation'] * self.factor_weights['valuation'] +
            df['score_growth'] * self.factor_weights['growth'] +
            df['score_quality'] * self.factor_weights['quality'] +
            df['score_momentum'] * self.factor_weights['momentum'] +
            df['score_sentiment'] * self.factor_weights['sentiment']
        )

        return df


class FactorStrategy:
    """多因子选股策略"""

    def __init__(
        self,
        min_score: float = 0.6,
        max_stocks: int = 20,
        rebalance_days: int = 20
    ):
        self.min_score = min_score
        self.max_stocks = max_stocks
        self.rebalance_days = rebalance_days
        self.scorer = FactorScorer()
        self.calculator = FactorCalculator()

    def select_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        选股逻辑
        """
        # 计算所有因子
        df = self.calculator.calculate_all_factors(df)

        # 计算评分
        df = self.scorer.calculate_total_score(df)

        # 过滤
        df = df[df['total_score'] >= self.min_score]

        # 排序选择前N
        df = df.sort_values('total_score', ascending=False).head(self.max_stocks)

        return df[[
            'ts_code', 'name', 'close', 'total_score',
            'score_valuation', 'score_growth', 'score_quality',
            'score_momentum', 'score_sentiment'
        ]]

    def generate_report(self, df: pd.DataFrame) -> Dict:
        """生成因子分析报告"""
        report = {
            'total_stocks': len(df),
            'avg_score': df['total_score'].mean(),
            'score_distribution': df['total_score'].describe().to_dict(),
            'top_stocks': df.nlargest(5, 'total_score')[['ts_code', 'name', 'total_score']].to_dict('records'),
            'factor_contribution': {
                'valuation': df['score_valuation'].mean(),
                'growth': df['score_growth'].mean(),
                'quality': df['score_quality'].mean(),
                'momentum': df['score_momentum'].mean(),
                'sentiment': df['score_sentiment'].mean()
            }
        }
        return report
