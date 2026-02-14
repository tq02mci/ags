"""
A股数据采集模块
支持多数据源: AKShare(免费), Tushare(付费), Baostock(免费)
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import akshare as ak
import pandas as pd
import tushare as ts
from loguru import logger

from src.config import settings
from src.database.connection import get_supabase_client


class DataCollector:
    """基础数据采集器"""

    def __init__(self):
        self.supabase = get_supabase_client()
        self.batch_size = settings.DATA_BATCH_SIZE

    async def save_to_supabase(self, table: str, data: pd.DataFrame, conflict_cols: Optional[List[str]] = None) -> int:
        """保存数据到 Supabase"""
        if data.empty:
            return 0

        try:
            # 处理所有可能的非 JSON 序列化类型
            for col in data.columns:
                if pd.api.types.is_datetime64_any_dtype(data[col]):
                    # 转换为字符串
                    data[col] = data[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                elif data[col].dtype == 'object':
                    # 处理 object 类型，可能是 Timestamp
                    try:
                        # 尝试检测是否包含 Timestamp 对象
                        sample = data[col].dropna().iloc[0] if not data[col].dropna().empty else None
                        if hasattr(sample, 'strftime'):
                            data[col] = pd.to_datetime(data[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass

            # 处理 NaN 值
            data = data.replace({pd.NaT: None, float('nan'): None})

            # 转换为字典列表，确保所有值都可以 JSON 序列化
            records = data.to_dict('records')

            # 递归处理所有值，确保可以 JSON 序列化
            def serialize_value(v):
                if hasattr(v, 'strftime'):
                    return v.strftime('%Y-%m-%d %H:%M:%S')
                elif pd.isna(v):
                    return None
                return v

            records = [
                {k: serialize_value(v) for k, v in record.items()}
                for record in records
            ]

            # 分批插入
            total_inserted = 0
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]

                if conflict_cols:
                    # UPSERT 模式
                    result = self.supabase.table(table).upsert(batch, on_conflict=','.join(conflict_cols)).execute()
                else:
                    # 普通插入
                    result = self.supabase.table(table).insert(batch).execute()

                total_inserted += len(batch)

            logger.info(f"成功保存 {total_inserted} 条记录到 {table}")
            return total_inserted

        except Exception as e:
            logger.error(f"保存到 {table} 失败: {e}")
            raise


class AKShareCollector(DataCollector):
    """AKShare 数据采集器 (免费)"""

    def __init__(self):
        super().__init__()

    def get_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表"""
        logger.info("正在从 AKShare 获取股票列表...")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 上海A股
                sh_stocks = ak.stock_sh_a_spot_em()
                sh_stocks['exchange'] = 'SH'

                # 深圳A股
                sz_stocks = ak.stock_sz_a_spot_em()
                sz_stocks['exchange'] = 'SZ'

                # 北京A股
                bj_stocks = ak.stock_bj_a_spot_em()
                bj_stocks['exchange'] = 'BJ'

                # 合并
                all_stocks = pd.concat([sh_stocks, sz_stocks, bj_stocks], ignore_index=True)

                # 标准化列名
                column_mapping = {
                    '代码': 'symbol',
                    '名称': 'name',
                    '最新价': 'close',
                    '涨跌幅': 'pct_change',
                    '涨跌额': 'change',
                    '成交量': 'vol',
                    '成交额': 'amount',
                    '振幅': 'amplitude',
                    '最高': 'high',
                    '最低': 'low',
                    '今开': 'open',
                    '昨收': 'pre_close',
                    '量比': 'volume_ratio',
                    '换手率': 'turnover',
                    '市盈率-动态': 'pe',
                    '市净率': 'pb',
                    '总市值': 'total_mv',
                    '流通市值': 'circ_mv',
                    '涨速': 'rise_speed',
                    '5分钟涨跌': '5min_change',
                    '60日涨跌幅': '60d_change',
                    '年初至今涨跌幅': 'ytd_change',
                }

                all_stocks = all_stocks.rename(columns=column_mapping)
                all_stocks['ts_code'] = all_stocks['symbol'] + '.' + all_stocks['exchange']

                return all_stocks

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"第 {attempt + 1} 次尝试失败: {e}，{2 ** attempt}秒后重试...")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"获取股票列表失败，已重试 {max_retries} 次: {e}")
                    raise

    def get_daily_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据"""
        try:
            # 判断交易所
            if symbol.startswith('6'):
                exchange = 'sh'
            elif symbol.startswith('0') or symbol.startswith('3'):
                exchange = 'sz'
            elif symbol.startswith('8') or symbol.startswith('4'):
                exchange = 'bj'
            else:
                exchange = 'sh'

            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="qfq"  # 前复权
            )

            if df.empty:
                return pd.DataFrame()

            # 标准化列名
            df = df.rename(columns={
                '日期': 'trade_date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'vol',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change',
                '换手率': 'turnover',
            })

            df['ts_code'] = f"{symbol}.{exchange.upper()}"
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')

            # 计算昨收
            df['pre_close'] = df['close'].shift(1)

            # 只保留数据库中存在的列
            db_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_change', 'vol', 'amount']
            df = df[[col for col in db_columns if col in df.columns]]

            return df

        except Exception as e:
            logger.error(f"获取 {symbol} 日线数据失败: {e}")
            return pd.DataFrame()

    def get_stock_info(self) -> pd.DataFrame:
        """获取股票基础信息"""
        logger.info("正在获取股票基础信息...")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 获取所有股票列表
                df = ak.stock_info_a_code_name()
                df = df.rename(columns={'code': 'symbol', 'name': 'name'})

                # 获取详细信息（带重试）
                try:
                    info_df = ak.stock_individual_info_em()
                    info_df = info_df.rename(columns={
                        '股票代码': 'symbol',
                        '股票简称': 'name',
                        '总股本': 'total_shares',
                        '流通股': 'float_shares',
                        '行业': 'industry',
                        '上市时间': 'list_date',
                    })
                    # 合并数据
                    df = df.merge(info_df, on='symbol', how='left', suffixes=('', '_info'))
                except Exception as e:
                    logger.warning(f"获取详细信息失败: {e}，使用基础信息")

                # 添加交易所
                df['exchange'] = df['symbol'].apply(self._get_exchange)
                df['ts_code'] = df['symbol'] + '.' + df['exchange']
                df['market'] = df['symbol'].apply(self._get_market)

                return df

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"第 {attempt + 1} 次尝试失败: {e}，{2 ** attempt}秒后重试...")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"获取股票信息失败，已重试 {max_retries} 次: {e}")
                    raise

    @staticmethod
    def _get_exchange(symbol: str) -> str:
        """根据代码判断交易所"""
        if symbol.startswith('6'):
            return 'SH'
        elif symbol.startswith('0') or symbol.startswith('3'):
            return 'SZ'
        elif symbol.startswith('8') or symbol.startswith('4'):
            return 'BJ'
        return 'SH'

    @staticmethod
    def _get_market(symbol: str) -> str:
        """根据代码判断市场板块"""
        if symbol.startswith('60'):
            return '主板'
        elif symbol.startswith('68'):
            return '科创板'
        elif symbol.startswith('00'):
            return '主板'
        elif symbol.startswith('30'):
            return '创业板'
        elif symbol.startswith('8') or symbol.startswith('4'):
            return '北交所'
        return '其他'

    async def sync_stock_list(self):
        """同步股票列表"""
        df = self.get_stock_info()

        # 标准化列
        df['list_status'] = 'L'
        df['area'] = None
        df['fullname'] = df['name']
        df['enname'] = None
        df['cnspell'] = None
        df['curr_type'] = 'CNY'
        df['is_hs'] = 'N'

        columns = ['ts_code', 'symbol', 'name', 'area', 'industry', 'fullname',
                   'enname', 'cnspell', 'market', 'exchange', 'curr_type',
                   'list_status', 'list_date', 'is_hs']

        df = df[[col for col in columns if col in df.columns]]

        await self.save_to_supabase('stocks_info', df, conflict_cols=['ts_code'])
        return len(df)

    async def sync_daily_data(self, trade_date: Optional[str] = None):
        """同步日线数据"""
        if trade_date is None:
            trade_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"同步 {trade_date} 日线数据...")

        # 获取当天所有股票行情
        df = ak.stock_zh_a_spot_em()

        # 标准化
        df['ts_code'] = df['代码'] + '.' + df['代码'].apply(
            lambda x: 'SH' if x.startswith('6') else 'SZ' if x.startswith(('0', '3')) else 'BJ'
        )
        df['trade_date'] = pd.to_datetime(trade_date)

        df = df.rename(columns={
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'vol',
            '成交额': 'amount',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '昨收': 'pre_close',
        })

        # 选择需要的列
        columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                   'pre_close', 'change', 'pct_change', 'vol', 'amount']
        df = df[[col for col in columns if col in df.columns]]

        await self.save_to_supabase('stock_daily', df, conflict_cols=['ts_code', 'trade_date'])
        return len(df)


class TushareCollector(DataCollector):
    """Tushare 数据采集器 (需要Token)"""

    def __init__(self):
        super().__init__()
        if settings.TUSHARE_TOKEN:
            self.pro = ts.pro_api(settings.TUSHARE_TOKEN)
        else:
            self.pro = None
            logger.warning("TUSHARE_TOKEN 未配置，Tushare 功能不可用")

    def is_available(self) -> bool:
        """检查是否可用"""
        return self.pro is not None

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        if not self.is_available():
            return pd.DataFrame()

        df = self.pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,curr_type,list_status,list_date,delist_date,is_hs'
        )
        return df

    def get_daily_data(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据"""
        if not self.is_available():
            return pd.DataFrame()

        df = self.pro.daily(
            ts_code=ts_code,
            start_date=start_date.replace('-', ''),
            end_date=end_date.replace('-', '')
        )

        if not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'])

        return df

    async def sync_stock_list(self):
        """同步股票列表"""
        if not self.is_available():
            logger.warning("Tushare 未配置，跳过")
            return 0

        df = self.get_stock_list()
        df['list_date'] = pd.to_datetime(df['list_date'], errors='coerce')
        df['delist_date'] = pd.to_datetime(df['delist_date'], errors='coerce')

        await self.save_to_supabase('stocks_info', df, conflict_cols=['ts_code'])
        return len(df)

    async def sync_daily_data(self, trade_date: Optional[str] = None):
        """同步日线数据"""
        if not self.is_available():
            return 0

        if trade_date is None:
            trade_date = datetime.now().strftime('%Y%m%d')
        else:
            trade_date = trade_date.replace('-', '')

        df = self.pro.daily(trade_date=trade_date)

        if not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            await self.save_to_supabase('stock_daily', df, conflict_cols=['ts_code', 'trade_date'])

        return len(df)

    def get_fundamental_data(self, ts_code: str) -> Dict[str, pd.DataFrame]:
        """获取基本面数据"""
        if not self.is_available():
            return {}

        data = {}

        # 利润表
        try:
            data['income'] = self.pro.income(ts_code=ts_code, period='20231231')
        except:
            pass

        # 资产负债表
        try:
            data['balance'] = self.pro.balancesheet(ts_code=ts_code, period='20231231')
        except:
            pass

        # 现金流量表
        try:
            data['cashflow'] = self.pro.cashflow(ts_code=ts_code, period='20231231')
        except:
            pass

        # 财务指标
        try:
            data['fina_indicator'] = self.pro.fina_indicator(ts_code=ts_code)
        except:
            pass

        return data


class DataSyncManager:
    """数据同步管理器"""

    def __init__(self):
        self.akshare = AKShareCollector()
        self.tushare = TushareCollector()
        self.supabase = get_supabase_client()

    async def full_sync(self):
        """全量同步"""
        logger.info("开始全量数据同步...")

        # 1. 同步股票列表
        logger.info("步骤 1/5: 同步股票列表")
        if self.tushare.is_available():
            await self.tushare.sync_stock_list()
        else:
            await self.akshare.sync_stock_list()

        # 2. 同步日线数据 (最近3年)
        logger.info("步骤 2/5: 同步历史日线数据")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 3)

        # 获取股票列表
        stocks = self.supabase.table('stocks_info').select('ts_code,list_status').eq('list_status', 'L').execute()

        for i, stock in enumerate(stocks.data):
            ts_code = stock['ts_code']
            symbol = ts_code.split('.')[0]

            try:
                # 尝试使用 AKShare 获取
                df = self.akshare.get_daily_data(
                    symbol,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )

                if not df.empty:
                    await self.akshare.save_to_supabase(
                        'stock_daily',
                        df,
                        conflict_cols=['ts_code', 'trade_date']
                    )

                if i % 100 == 0:
                    logger.info(f"已处理 {i}/{len(stocks.data)} 只股票")

                # 限速
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"同步 {ts_code} 失败: {e}")
                continue

        logger.info("全量同步完成")

    async def incremental_sync(self, days: int = 1):
        """增量同步"""
        logger.info(f"开始增量同步 (最近 {days} 天)...")

        for i in range(days):
            trade_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')

            # 检查是否为交易日
            cal = self.supabase.table('trade_calendar').select('*').eq('cal_date', trade_date).eq('is_open', True).execute()
            if not cal.data:
                logger.info(f"{trade_date} 非交易日，跳过")
                continue

            try:
                await self.akshare.sync_daily_data(trade_date)
                logger.info(f"同步 {trade_date} 完成")
            except Exception as e:
                logger.error(f"同步 {trade_date} 失败: {e}")

        logger.info("增量同步完成")

    async def sync_single_stock(self, ts_code: str, days: int = 365):
        """同步单只股票数据"""
        symbol = ts_code.split('.')[0]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = self.akshare.get_daily_data(
            symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        if not df.empty:
            await self.akshare.save_to_supabase(
                'stock_daily',
                df,
                conflict_cols=['ts_code', 'trade_date']
            )

        return len(df)
