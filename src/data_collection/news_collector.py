"""
A股资讯数据采集模块
支持: 新闻、公告、研报、龙虎榜、大宗交易等
"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import akshare as ak
import pandas as pd
from loguru import logger

from src.config import settings
from src.database.connection import get_supabase_client


class NewsCollector:
    """资讯数据采集器 (基于 AKShare 免费接口)"""

    def __init__(self):
        self.supabase = get_supabase_client()

    # ==================== 新闻资讯 ====================

    def get_stock_news(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        获取个股新闻
        来源: 东方财富
        """
        try:
            df = ak.stock_news_em(symbol=symbol)

            if df.empty:
                return pd.DataFrame()

            # 标准化
            df = df.rename(columns={
                '关键词': 'keywords',
                '新闻标题': 'title',
                '新闻内容': 'content',
                '发布时间': 'publish_time',
                '文章来源': 'source',
            })

            df['symbol'] = symbol
            df['ts_code'] = self._symbol_to_ts_code(symbol)
            df['news_type'] = 'stock'
            df['created_at'] = datetime.now()

            # 限制字段长度
            df['title'] = df['title'].str[:200]
            df['content'] = df['content'].str[:4000]

            return df

        except Exception as e:
            logger.error(f"获取 {symbol} 新闻失败: {e}")
            return pd.DataFrame()

    def get_major_news(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        获取重大新闻/财经要闻
        """
        try:
            df = ak.stock_zh_a_alerts_cls()

            if df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                '时间': 'publish_time',
                '内容': 'content',
            })

            df['news_type'] = 'major'
            df['title'] = df['content'].str[:100] + '...'
            df['source'] = '财联社'
            df['created_at'] = datetime.now()

            return df

        except Exception as e:
            logger.error(f"获取重大新闻失败: {e}")
            return pd.DataFrame()

    # ==================== 公司公告 ====================

    def get_stock_notice(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取公司公告
        类型: 业绩公告、分红、股权激励、增减持等
        """
        try:
            df = ak.stock_notice_report(symbol=symbol, date="")

            if df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                '公告日期': 'publish_date',
                '公告标题': 'title',
                '公告内容': 'content',
                '公告类型': 'notice_type',
            })

            df['symbol'] = symbol
            df['ts_code'] = self._symbol_to_ts_code(symbol)
            df['created_at'] = datetime.now()

            return df

        except Exception as e:
            logger.error(f"获取 {symbol} 公告失败: {e}")
            return pd.DataFrame()

    def get_all_notices(self, date: str) -> pd.DataFrame:
        """
        获取某日期所有公告
        """
        try:
            df = ak.stock_notice_report(symbol="全部", date=date.replace('-', ''))

            if df.empty:
                return pd.DataFrame()

            df['created_at'] = datetime.now()
            return df

        except Exception as e:
            logger.error(f"获取 {date} 公告失败: {e}")
            return pd.DataFrame()

    # ==================== 研究报告 ====================

    def get_stock_research(self, symbol: str) -> pd.DataFrame:
        """
        获取个股研报
        """
        try:
            # 东方财富研报
            df = ak.stock_research_report_em(symbol=symbol)

            if df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                '研报日期': 'report_date',
                '机构': 'institution',
                '评级': 'rating',
                '目标价': 'target_price',
                '标题': 'title',
            })

            df['symbol'] = symbol
            df['ts_code'] = self._symbol_to_ts_code(symbol)
            df['report_type'] = 'research'
            df['created_at'] = datetime.now()

            return df

        except Exception as e:
            logger.error(f"获取 {symbol} 研报失败: {e}")
            return pd.DataFrame()

    def get_research_report(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取全市场研报
        """
        try:
            df = ak.stock_research_report_em(symbol="")

            if df.empty:
                return pd.DataFrame()

            df['created_at'] = datetime.now()
            return df

        except Exception as e:
            logger.error(f"获取研报失败: {e}")
            return pd.DataFrame()

    # ==================== 龙虎榜 ====================

    def get_longhu_bang(self, date: str) -> pd.DataFrame:
        """
        获取龙虎榜数据
        包含: 营业部买卖情况、机构席位
        """
        try:
            df = ak.stock_lhb_detail_daily_sina(start_date=date, end_date=date)

            if df.empty:
                return pd.DataFrame()

            df['trade_date'] = date
            df['created_at'] = datetime.now()

            return df

        except Exception as e:
            logger.error(f"获取 {date} 龙虎榜失败: {e}")
            return pd.DataFrame()

    def get_longhu_detail(self, symbol: str, date: str) -> pd.DataFrame:
        """
        获取个股龙虎榜详细数据
        """
        try:
            df = ak.stock_lhb_stock_detail_em(symbol=symbol, date=date.replace('-', ''))

            if df.empty:
                return pd.DataFrame()

            df['symbol'] = symbol
            df['ts_code'] = self._symbol_to_ts_code(symbol)
            df['trade_date'] = date
            df['created_at'] = datetime.now()

            return df

        except Exception as e:
            logger.error(f"获取 {symbol} 龙虎榜详情失败: {e}")
            return pd.DataFrame()

    # ==================== 大宗交易 ====================

    def get_dzjy(self, date: str) -> pd.DataFrame:
        """
        获取大宗交易数据
        """
        try:
            df = ak.stock_dzjy_mrmx(symbol="", start_date=date.replace('-', ''), end_date=date.replace('-', ''))

            if df.empty:
                return pd.DataFrame()

            df['trade_date'] = date
            df['created_at'] = datetime.now()

            return df

        except Exception as e:
            logger.error(f"获取 {date} 大宗交易失败: {e}")
            return pd.DataFrame()

    # ==================== 资金流向 ====================

    def get_money_flow(self, symbol: str) -> pd.DataFrame:
        """
        获取个股资金流向
        """
        try:
            df = ak.stock_individual_fund_flow(symbol=symbol)

            if df.empty:
                return pd.DataFrame()

            df['symbol'] = symbol
            df['ts_code'] = self._symbol_to_ts_code(symbol)
            df['created_at'] = datetime.now()

            return df

        except Exception as e:
            logger.error(f"获取 {symbol} 资金流向失败: {e}")
            return pd.DataFrame()

    def get_sector_money_flow(self) -> pd.DataFrame:
        """
        获取行业资金流向
        """
        try:
            df = ak.stock_sector_fund_flow_rank()

            if df.empty:
                return pd.DataFrame()

            df['created_at'] = datetime.now()
            return df

        except Exception as e:
            logger.error(f"获取行业资金流向失败: {e}")
            return pd.DataFrame()

    # ==================== 融资融券 ====================

    def get_margin_trading(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取融资融券数据
        """
        try:
            df = ak.stock_margin_detail_szse(symbol=symbol)

            if df.empty:
                return pd.DataFrame()

            df['symbol'] = symbol
            df['ts_code'] = self._symbol_to_ts_code(symbol)
            df['created_at'] = datetime.now()

            return df

        except Exception as e:
            logger.error(f"获取 {symbol} 融资融券失败: {e}")
            return pd.DataFrame()

    # ==================== 同步方法 ====================

    async def sync_stock_news(self, symbols: List[str] = None):
        """同步个股新闻"""
        if symbols is None:
            # 默认同步热门股票
            symbols = ['000001', '000858', '600519', '300750']  # 平安、五粮液、茅台、宁德时代

        total = 0
        for symbol in symbols:
            try:
                df = self.get_stock_news(symbol)
                if not df.empty:
                    # 保存到数据库
                    records = df.to_dict('records')
                    self.supabase.table("stock_news").upsert(
                        records,
                        on_conflict="ts_code,title"
                    ).execute()
                    total += len(df)
                    logger.info(f"同步 {symbol} 新闻: {len(df)} 条")

                await asyncio.sleep(0.5)  # 限速

            except Exception as e:
                logger.error(f"同步 {symbol} 新闻失败: {e}")

        return total

    async def sync_daily_news(self, date: Optional[str] = None):
        """同步每日重大新闻"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        try:
            df = self.get_major_news(date)
            if not df.empty:
                records = df.to_dict('records')
                self.supabase.table("market_news").insert(records).execute()
                logger.info(f"同步 {date} 重大新闻: {len(df)} 条")
                return len(df)
        except Exception as e:
            logger.error(f"同步重大新闻失败: {e}")

        return 0

    async def sync_longhu_bang(self, date: Optional[str] = None):
        """同步龙虎榜数据"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        try:
            df = self.get_longhu_bang(date)
            if not df.empty:
                records = df.to_dict('records')
                self.supabase.table("longhu_bang").upsert(
                    records,
                    on_conflict="symbol,trade_date"
                ).execute()
                logger.info(f"同步 {date} 龙虎榜: {len(df)} 条")
                return len(df)
        except Exception as e:
            logger.error(f"同步龙虎榜失败: {e}")

        return 0

    @staticmethod
    def _symbol_to_ts_code(symbol: str) -> str:
        """转换为 Tushare 格式"""
        if symbol.startswith('6'):
            return f"{symbol}.SH"
        elif symbol.startswith(('0', '3')):
            return f"{symbol}.SZ"
        elif symbol.startswith(('8', '4')):
            return f"{symbol}.BJ"
        return symbol


class TushareNewsCollector:
    """
    Tushare 资讯采集器 (付费，数据更全面)
    需要 TUSHARE_TOKEN
    """

    def __init__(self):
        import tushare as ts
        if settings.TUSHARE_TOKEN:
            self.pro = ts.pro_api(settings.TUSHARE_TOKEN)
        else:
            self.pro = None
            logger.warning("TUSHARE_TOKEN 未配置")

    def is_available(self) -> bool:
        return self.pro is not None

    def get_major_news(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取重大新闻"""
        if not self.is_available():
            return pd.DataFrame()

        try:
            df = self.pro.major_news(
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                src='sina'  # 或 'wallstreetcn'
            )
            return df
        except Exception as e:
            logger.error(f"获取重大新闻失败: {e}")
            return pd.DataFrame()

    def get_news(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取新闻快讯"""
        if not self.is_available():
            return pd.DataFrame()

        try:
            df = self.pro.news(
                start_date=start_date,
                end_date=end_date
            )
            return df
        except Exception as e:
            logger.error(f"获取新闻失败: {e}")
            return pd.DataFrame()

    def get_cctv_news(self, date: str) -> pd.DataFrame:
        """获取新闻联播"""
        if not self.is_available():
            return pd.DataFrame()

        try:
            df = self.pro.cctv_news(date=date.replace('-', ''))
            return df
        except Exception as e:
            logger.error(f"获取新闻联播失败: {e}")
            return pd.DataFrame()

    def get_disclosure_notice(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取上市公司公告"""
        if not self.is_available():
            return pd.DataFrame()

        try:
            df = self.pro.anns(
                ts_code=ts_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )
            return df
        except Exception as e:
            logger.error(f"获取公告失败: {e}")
            return pd.DataFrame()

    def get_research_report(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取券商研报"""
        if not self.is_available():
            return pd.DataFrame()

        try:
            df = self.pro.report(
                ts_code=ts_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )
            return df
        except Exception as e:
            logger.error(f"获取研报失败: {e}")
            return pd.DataFrame()
