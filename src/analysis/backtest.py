"""
A股回测系统
支持多种策略的回测和绩效分析
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"      # 市价单
    LIMIT = "limit"        # 限价单


@dataclass
class Order:
    """订单"""
    id: str
    ts_code: str
    side: OrderSide
    order_type: OrderType
    price: float
    volume: int
    create_time: datetime
    filled_volume: int = 0
    filled_price: float = 0.0
    status: str = "pending"  # pending, filled, cancelled, rejected


@dataclass
class Trade:
    """成交记录"""
    order_id: str
    ts_code: str
    side: OrderSide
    price: float
    volume: int
    trade_time: datetime
    commission: float = 0.0


@dataclass
class Position:
    """持仓"""
    ts_code: str
    volume: int = 0
    avg_cost: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def update_price(self, price: float):
        """更新价格"""
        self.market_value = self.volume * price
        self.unrealized_pnl = self.market_value - self.volume * self.avg_cost


@dataclass
class Portfolio:
    """投资组合"""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    total_value: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0

    def update_value(self):
        """更新总价值"""
        position_value = sum(p.market_value for p in self.positions.values())
        self.total_value = self.cash + position_value


class BacktestEngine:
    """回测引擎"""

    def __init__(
        self,
        initial_capital: float = 1000000.0,
        commission_rate: float = 0.0003,
        slippage: float = 0.001,
        min_commission: float = 5.0
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.min_commission = min_commission

        # 回测状态
        self.portfolio: Optional[Portfolio] = None
        self.current_date: Optional[datetime] = None
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.daily_values: List[Dict] = []
        self.daily_positions: List[Dict] = []

        # 历史数据
        self.price_data: Optional[pd.DataFrame] = None
        self.signal_data: Optional[pd.DataFrame] = None

    def reset(self):
        """重置回测状态"""
        self.portfolio = Portfolio(cash=self.initial_capital)
        self.orders = []
        self.trades = []
        self.daily_values = []
        self.daily_positions = []
        self.current_date = None

    def set_data(
        self,
        price_data: pd.DataFrame,
        signal_data: Optional[pd.DataFrame] = None
    ):
        """设置回测数据"""
        self.price_data = price_data.copy()
        self.signal_data = signal_data.copy() if signal_data is not None else None

        # 确保数据按日期排序
        self.price_data = self.price_data.sort_values('trade_date')

    def buy(
        self,
        ts_code: str,
        volume: Optional[int] = None,
        percent: Optional[float] = None,
        price: Optional[float] = None
    ) -> bool:
        """买入"""
        if self.current_date is None or self.price_data is None:
            return False

        # 获取当前价格
        current_data = self.price_data[self.price_data['trade_date'] == self.current_date]
        if current_data.empty:
            return False

        current_price = price or current_data['close'].values[0]
        current_price = current_price * (1 + self.slippage)  # 考虑滑点

        # 计算买入数量
        if volume is not None:
            buy_volume = volume
        elif percent is not None:
            buy_amount = self.portfolio.cash * percent
            buy_volume = int(buy_amount / current_price / 100) * 100  # 100股为单位
        else:
            # 全仓买入
            buy_volume = int(self.portfolio.cash / current_price / 100) * 100

        if buy_volume <= 0:
            return False

        # 计算费用
        total_cost = buy_volume * current_price
        commission = max(total_cost * self.commission_rate, self.min_commission)

        if total_cost + commission > self.portfolio.cash:
            # 资金不足，调整数量
            available_amount = self.portfolio.cash - commission
            buy_volume = int(available_amount / current_price / 100) * 100
            if buy_volume <= 0:
                return False
            total_cost = buy_volume * current_price

        # 创建订单
        order = Order(
            id=f"{self.current_date.strftime('%Y%m%d')}_{ts_code}_buy_{len(self.orders)}",
            ts_code=ts_code,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            price=current_price,
            volume=buy_volume,
            create_time=self.current_date,
            filled_volume=buy_volume,
            filled_price=current_price,
            status="filled"
        )
        self.orders.append(order)

        # 创建成交记录
        trade = Trade(
            order_id=order.id,
            ts_code=ts_code,
            side=OrderSide.BUY,
            price=current_price,
            volume=buy_volume,
            trade_time=self.current_date,
            commission=commission
        )
        self.trades.append(trade)

        # 更新持仓
        if ts_code not in self.portfolio.positions:
            self.portfolio.positions[ts_code] = Position(ts_code=ts_code)

        position = self.portfolio.positions[ts_code]
        total_volume = position.volume + buy_volume
        position.avg_cost = (position.volume * position.avg_cost + total_cost) / total_volume
        position.volume = total_volume

        # 更新现金
        self.portfolio.cash -= (total_cost + commission)

        logger.debug(f"买入 {ts_code}: {buy_volume}股 @ {current_price:.2f}")
        return True

    def sell(
        self,
        ts_code: str,
        volume: Optional[int] = None,
        percent: Optional[float] = None,
        price: Optional[float] = None
    ) -> bool:
        """卖出"""
        if self.current_date is None or self.price_data is None:
            return False

        if ts_code not in self.portfolio.positions:
            return False

        position = self.portfolio.positions[ts_code]
        if position.volume <= 0:
            return False

        # 获取当前价格
        current_data = self.price_data[self.price_data['trade_date'] == self.current_date]
        if current_data.empty:
            return False

        current_price = price or current_data['close'].values[0]
        current_price = current_price * (1 - self.slippage)  # 考虑滑点

        # 计算卖出数量
        if volume is not None:
            sell_volume = min(volume, position.volume)
        elif percent is not None:
            sell_volume = int(position.volume * percent / 100) * 100
        else:
            sell_volume = position.volume

        if sell_volume <= 0:
            return False

        # 计算费用
        total_value = sell_volume * current_price
        commission = max(total_value * self.commission_rate, self.min_commission)

        # 创建订单
        order = Order(
            id=f"{self.current_date.strftime('%Y%m%d')}_{ts_code}_sell_{len(self.orders)}",
            ts_code=ts_code,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            price=current_price,
            volume=sell_volume,
            create_time=self.current_date,
            filled_volume=sell_volume,
            filled_price=current_price,
            status="filled"
        )
        self.orders.append(order)

        # 计算盈亏
        cost = sell_volume * position.avg_cost
        realized_pnl = sell_volume * current_price - cost - commission

        # 创建成交记录
        trade = Trade(
            order_id=order.id,
            ts_code=ts_code,
            side=OrderSide.SELL,
            price=current_price,
            volume=sell_volume,
            trade_time=self.current_date,
            commission=commission
        )
        self.trades.append(trade)

        # 更新持仓
        position.volume -= sell_volume
        position.realized_pnl += realized_pnl

        if position.volume == 0:
            position.avg_cost = 0

        # 更新现金
        self.portfolio.cash += (total_value - commission)

        logger.debug(f"卖出 {ts_code}: {sell_volume}股 @ {current_price:.2f}, 盈亏: {realized_pnl:.2f}")
        return True

    def update_portfolio(self):
        """更新组合状态"""
        if self.current_date is None or self.price_data is None:
            return

        current_data = self.price_data[self.price_data['trade_date'] == self.current_date]
        if current_data.empty:
            return

        price_dict = dict(zip(current_data['ts_code'], current_data['close']))

        # 更新持仓市值
        for ts_code, position in self.portfolio.positions.items():
            if position.volume > 0 and ts_code in price_dict:
                position.update_price(price_dict[ts_code])

        # 计算总价值
        prev_value = self.portfolio.total_value if self.portfolio.total_value > 0 else self.initial_capital
        self.portfolio.update_value()
        self.portfolio.daily_pnl = self.portfolio.total_value - prev_value
        self.portfolio.total_pnl = self.portfolio.total_value - self.initial_capital

        # 记录每日状态
        self.daily_values.append({
            'date': self.current_date,
            'total_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'position_value': self.portfolio.total_value - self.portfolio.cash,
            'daily_pnl': self.portfolio.daily_pnl,
            'total_pnl': self.portfolio.total_pnl,
            'return': (self.portfolio.total_value - self.initial_capital) / self.initial_capital
        })

        # 记录持仓
        positions_snapshot = {
            'date': self.current_date,
            'positions': {
                code: {
                    'volume': pos.volume,
                    'avg_cost': pos.avg_cost,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl
                }
                for code, pos in self.portfolio.positions.items()
                if pos.volume > 0
            }
        }
        self.daily_positions.append(positions_snapshot)

    def run(
        self,
        strategy: 'BaseStrategy',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """运行回测"""
        if self.price_data is None:
            raise ValueError("请先设置回测数据")

        self.reset()

        # 过滤日期范围
        df = self.price_data.copy()
        if start_date:
            df = df[df['trade_date'] >= start_date]
        if end_date:
            df = df[df['trade_date'] <= end_date]

        if df.empty:
            raise ValueError("回测数据为空")

        # 绑定引擎到策略
        strategy.bind_engine(self)

        logger.info(f"开始回测: {df['trade_date'].min()} 到 {df['trade_date'].max()}")

        # 逐日回测
        for date in df['trade_date'].unique():
            self.current_date = date

            # 更新组合状态
            self.update_portfolio()

            # 执行策略
            daily_data = df[df['trade_date'] == date]
            strategy.on_bar(daily_data)

        # 计算绩效指标
        results = self.calculate_metrics()

        logger.info(f"回测完成: 总收益率 {results['total_return']:.2%}")

        return results

    def calculate_metrics(self) -> Dict:
        """计算绩效指标"""
        if not self.daily_values:
            return {}

        values_df = pd.DataFrame(self.daily_values)
        values_df['date'] = pd.to_datetime(values_df['date'])
        values_df = values_df.sort_values('date')

        # 计算收益率序列
        values_df['daily_return'] = values_df['total_value'].pct_change()

        # 总收益率
        total_return = (values_df['total_value'].iloc[-1] - self.initial_capital) / self.initial_capital

        # 年化收益率
        trading_days = len(values_df)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1

        # 波动率
        volatility = values_df['daily_return'].std() * np.sqrt(252)

        # 夏普比率 (假设无风险利率 3%)
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

        # 最大回撤
        values_df['cummax'] = values_df['total_value'].cummax()
        values_df['drawdown'] = (values_df['total_value'] - values_df['cummax']) / values_df['cummax']
        max_drawdown = values_df['drawdown'].min()

        # 卡玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 胜率
        winning_days = len(values_df[values_df['daily_return'] > 0])
        win_rate = winning_days / len(values_df)

        # 盈亏比
        avg_gain = values_df[values_df['daily_return'] > 0]['daily_return'].mean()
        avg_loss = abs(values_df[values_df['daily_return'] < 0]['daily_return'].mean())
        profit_loss_ratio = avg_gain / avg_loss if avg_loss > 0 else 0

        # 交易统计
        winning_trades = len([t for t in self.trades if t.side == OrderSide.SELL])
        total_trades = len([t for t in self.trades])

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'trading_days': trading_days,
            'final_value': values_df['total_value'].iloc[-1],
            'daily_values': self.daily_values,
            'trades': [
                {
                    'ts_code': t.ts_code,
                    'side': t.side.value,
                    'price': t.price,
                    'volume': t.volume,
                    'time': t.trade_time,
                    'commission': t.commission
                }
                for t in self.trades
            ]
        }


class BaseStrategy(ABC):
    """策略基类"""

    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.engine: Optional[BacktestEngine] = None

    def bind_engine(self, engine: BacktestEngine):
        """绑定回测引擎"""
        self.engine = engine

    @abstractmethod
    def on_bar(self, data: pd.DataFrame):
        """每个交易 bar 触发的逻辑"""
        pass

    def buy(self, ts_code: str, **kwargs) -> bool:
        """买入"""
        if self.engine:
            return self.engine.buy(ts_code, **kwargs)
        return False

    def sell(self, ts_code: str, **kwargs) -> bool:
        """卖出"""
        if self.engine:
            return self.engine.sell(ts_code, **kwargs)
        return False


class MACDStrategy(BaseStrategy):
    """MACD 金叉死叉策略"""

    def __init__(
        self,
        ts_code: str,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ):
        super().__init__("MACDStrategy")
        self.ts_code = ts_code
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.position = 0

    def on_bar(self, data: pd.DataFrame):
        """根据 MACD 信号交易"""
        if data.empty:
            return

        row = data.iloc[0]

        # 获取 MACD 指标
        macd_bar = row.get('macd_bar', 0)
        macd_bar_prev = row.get('macd_bar', 0)  # 简化，实际需要前一日数据

        # MACD 金叉 (柱状图由负转正)
        if macd_bar > 0 and self.position <= 0:
            if self.position < 0:
                self.sell(self.ts_code, percent=1.0)  # 平空仓
            self.buy(self.ts_code, percent=0.95)      # 买入
            self.position = 1

        # MACD 死叉 (柱状图由正转负)
        elif macd_bar < 0 and self.position >= 0:
            if self.position > 0:
                self.sell(self.ts_code, percent=1.0)  # 平多仓
            self.position = -1


class MAStrategy(BaseStrategy):
    """均线突破策略"""

    def __init__(
        self,
        ts_code: str,
        short_window: int = 5,
        long_window: int = 20
    ):
        super().__init__("MAStrategy")
        self.ts_code = ts_code
        self.short_window = short_window
        self.long_window = long_window

    def on_bar(self, data: pd.DataFrame):
        """根据均线交叉交易"""
        if data.empty:
            return

        row = data.iloc[0]

        ma_short = row.get(f'ma{self.short_window}', 0)
        ma_long = row.get(f'ma{self.long_window}', 0)

        if ma_short == 0 or ma_long == 0:
            return

        # 金叉买入
        if ma_short > ma_long:
            self.buy(self.ts_code, percent=0.95)

        # 死叉卖出
        elif ma_short < ma_long:
            self.sell(self.ts_code, percent=1.0)


class RSIStrategy(BaseStrategy):
    """RSI 超买超卖策略"""

    def __init__(
        self,
        ts_code: str,
        oversold: int = 30,
        overbought: int = 70
    ):
        super().__init__("RSIStrategy")
        self.ts_code = ts_code
        self.oversold = oversold
        self.overbought = overbought

    def on_bar(self, data: pd.DataFrame):
        """根据 RSI 交易"""
        if data.empty:
            return

        row = data.iloc[0]
        rsi = row.get('rsi6', 50)

        # 超卖买入
        if rsi < self.oversold:
            self.buy(self.ts_code, percent=0.5)

        # 超买卖出
        elif rsi > self.overbought:
            self.sell(self.ts_code, percent=1.0)


class MultiFactorStrategy(BaseStrategy):
    """多因子组合策略"""

    def __init__(
        self,
        ts_codes: List[str],
        lookback_days: int = 20,
        top_n: int = 5
    ):
        super().__init__("MultiFactorStrategy")
        self.ts_codes = ts_codes
        self.lookback_days = lookback_days
        self.top_n = top_n
        self.holdings: Dict[str, float] = {}

    def on_bar(self, data: pd.DataFrame):
        """多因子选股 + 定期调仓"""
        if data.empty:
            return

        # 简化版本：基于技术指标评分
        scores = []

        for _, row in data.iterrows():
            ts_code = row.get('ts_code')
            score = 0

            # MACD 加分
            if row.get('macd_bar', 0) > 0:
                score += 1

            # 均线多头排列加分
            if row.get('ma5', 0) > row.get('ma20', 0) > row.get('ma60', 0):
                score += 1

            # RSI 适中加分
            rsi = row.get('rsi6', 50)
            if 30 < rsi < 70:
                score += 1

            # 上涨趋势加分
            if row.get('close', 0) > row.get('ma20', 0):
                score += 1

            scores.append((ts_code, score))

        # 排序选择前 N
        scores.sort(key=lambda x: x[1], reverse=True)
        top_stocks = [code for code, _ in scores[:self.top_n]]

        # 卖出不在持仓的股票
        for ts_code in list(self.holdings.keys()):
            if ts_code not in top_stocks:
                self.sell(ts_code, percent=1.0)
                del self.holdings[ts_code]

        # 买入新的股票
        for ts_code in top_stocks:
            if ts_code not in self.holdings:
                self.buy(ts_code, percent=0.9 / self.top_n)
                self.holdings[ts_code] = 0.9 / self.top_n


def run_backtest_example():
    """回测示例"""
    # 创建模拟数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
    np.random.seed(42)

    data = []
    price = 10.0
    for date in dates:
        change = np.random.normal(0, 0.02)
        price = price * (1 + change)

        data.append({
            'trade_date': date,
            'ts_code': '000001.SZ',
            'open': price * (1 + np.random.normal(0, 0.005)),
            'high': price * (1 + abs(np.random.normal(0, 0.01))),
            'low': price * (1 - abs(np.random.normal(0, 0.01))),
            'close': price,
            'vol': int(np.random.randint(1000000, 10000000)),
            'macd_bar': np.sin(len(data) * 0.1) + np.random.normal(0, 0.1),
            'ma5': price * (1 + np.random.normal(0, 0.01)),
            'ma20': price * (1 + np.random.normal(0, 0.02)),
            'rsi6': 50 + np.random.normal(0, 20)
        })

    df = pd.DataFrame(data)

    # 运行回测
    engine = BacktestEngine(initial_capital=1000000)
    engine.set_data(df)

    strategy = MACDStrategy('000001.SZ')
    results = engine.run(strategy)

    print(f"总收益率: {results['total_return']:.2%}")
    print(f"年化收益率: {results['annual_return']:.2%}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    run_backtest_example()
