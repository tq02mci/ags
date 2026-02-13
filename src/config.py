"""
A股量化交易系统 - 配置管理
"""
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """系统配置"""

    # 项目路径
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # Supabase 配置
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_SERVICE_KEY: str = ""

    # 数据库直接连接 (可选)
    DATABASE_URL: Optional[str] = None

    # Tushare API Token
    TUSHARE_TOKEN: str = ""

    # Redis 配置 (可选)
    REDIS_URL: Optional[str] = None

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    # API 配置
    API_BASE_URL: str = "http://localhost:8000"

    # 开发环境
    DEBUG: bool = False

    # 数据采集配置
    DATA_BATCH_SIZE: int = 1000
    MAX_WORKERS: int = 4
    REQUEST_TIMEOUT: int = 30

    # 模型配置
    MODEL_DEFAULT_TRAIN_DAYS: int = 252 * 3  # 3年
    MODEL_DEFAULT_TEST_DAYS: int = 60  # 60天
    MODEL_CHECKPOINT_DIR: Path = PROJECT_ROOT / "models" / "checkpoints"

    # 回测配置
    BACKTEST_INITIAL_CAPITAL: float = 1000000.0  # 100万初始资金
    BACKTEST_COMMISSION_RATE: float = 0.0003  # 手续费
    BACKTEST_SLIPPAGE: float = 0.001  # 滑点

    class Config:
        env_file = Path(__file__).parent.parent / ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保目录存在
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# 全局配置实例
settings = Settings()


# 股票市场配置
MARKET_CONFIG = {
    "SSE": {  # 上海证券交易所
        "open_time": "09:30",
        "close_time": "15:00",
        "lunch_break": ("11:30", "13:00"),
        "trade_days": "Mon-Fri",
    },
    "SZSE": {  # 深圳证券交易所
        "open_time": "09:30",
        "close_time": "15:00",
        "lunch_break": ("11:30", "13:00"),
        "trade_days": "Mon-Fri",
    },
    "BJSE": {  # 北京证券交易所
        "open_time": "09:30",
        "close_time": "15:00",
        "lunch_break": ("11:30", "13:00"),
        "trade_days": "Mon-Fri",
    },
}

# 数据表配置
TABLE_SCHEMA = {
    "stocks_info": "股票基础信息",
    "trade_calendar": "交易日历",
    "stock_daily": "日线行情",
    "stock_minute": "分钟线行情",
    "adj_factor": "复权因子",
    "income_statement": "利润表",
    "balance_sheet": "资产负债表",
    "cash_flow": "现金流量表",
    "financial_indicators": "财务指标",
    "technical_indicators": "技术指标",
    "model_configs": "模型配置",
    "predictions": "预测结果",
    "backtest_results": "回测结果",
    "data_sync_log": "数据同步日志",
}
