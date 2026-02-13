"""
Supabase 数据库连接管理
"""
from typing import Optional

from supabase import Client, create_client
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import settings


class DatabaseManager:
    """数据库管理器"""

    _instance: Optional["DatabaseManager"] = None
    _client: Optional[Client] = None
    _engine = None
    _session_maker = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def client(self) -> Client:
        """获取 Supabase 客户端"""
        if self._client is None:
            if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
                raise ValueError("SUPABASE_URL 和 SUPABASE_KEY 必须配置")
            self._client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_SERVICE_KEY or settings.SUPABASE_KEY
            )
        return self._client

    @property
    def engine(self):
        """获取 SQLAlchemy 引擎"""
        if self._engine is None:
            if settings.DATABASE_URL:
                self._engine = create_engine(settings.DATABASE_URL)
            else:
                # 从 Supabase URL 构建连接字符串
                # postgresql://postgres:[YOUR-PASSWORD]@db.xxx.supabase.co:5432/postgres
                raise ValueError("需要配置 DATABASE_URL 或 Supabase 连接信息")
        return self._engine

    @property
    def session(self):
        """获取数据库会话"""
        if self._session_maker is None:
            self._session_maker = sessionmaker(bind=self.engine)
        return self._session_maker()

    def reset(self):
        """重置连接"""
        self._client = None
        self._engine = None
        self._session_maker = None


# 全局数据库管理器实例
db = DatabaseManager()


# 便捷函数
def get_supabase_client() -> Client:
    """获取 Supabase 客户端"""
    return db.client


def get_db_session():
    """获取数据库会话 (用于依赖注入)"""
    session = db.session
    try:
        yield session
    finally:
        session.close()
