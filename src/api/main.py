"""
A股量化交易系统 - FastAPI 服务
"""
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import settings
from src.database.connection import get_supabase_client


# 数据模型
class StockInfo(BaseModel):
    ts_code: str
    name: str
    industry: Optional[str] = None
    exchange: Optional[str] = None
    list_date: Optional[str] = None


class StockDaily(BaseModel):
    ts_code: str
    trade_date: str
    open: float
    high: float
    low: float
    close: float
    vol: int
    amount: float
    pct_change: Optional[float] = None


class PredictionRequest(BaseModel):
    ts_code: str
    days: int = 5


class PredictionResponse(BaseModel):
    ts_code: str
    pred_date: str
    predictions: List[dict]


class BacktestRequest(BaseModel):
    ts_code: str
    start_date: str
    end_date: str
    strategy: str = "macd"
    initial_capital: float = 1000000.0


# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动
    print(f"🚀 A股量化交易系统 API 启动于 {datetime.now()}")
    yield
    # 关闭
    print(f"👋 API 服务关闭于 {datetime.now()}")


# 创建 FastAPI 应用
app = FastAPI(
    title="A股量化交易系统 API",
    description="提供股票数据查询、技术分析、预测和回测功能",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 获取 Supabase 客户端
def get_db():
    return get_supabase_client()


@app.get("/")
async def root():
    return {
        "message": "A股量化交易系统 API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    try:
        db = get_db()
        db.table("stocks_info").select("count", count="exact").limit(1).execute()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# ===== 股票信息接口 =====

@app.get("/api/stocks", response_model=List[StockInfo])
async def get_stocks(
    exchange: Optional[str] = None,
    industry: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """获取股票列表"""
    try:
        db = get_db()
        query = db.table("stocks_info").select("*")

        if exchange:
            query = query.eq("exchange", exchange)
        if industry:
            query = query.eq("industry", industry)

        result = query.eq("list_status", "L").limit(limit).offset(offset).execute()

        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/{ts_code}")
async def get_stock_detail(ts_code: str):
    """获取股票详情"""
    try:
        db = get_db()
        result = db.table("stocks_info").select("*").eq("ts_code", ts_code).single().execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="股票不存在")

        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/{ts_code}/daily")
async def get_stock_daily(
    ts_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = Query(252, ge=1, le=1000)
):
    """获取股票日线数据"""
    try:
        db = get_db()
        query = db.table("stock_daily").select("*").eq("ts_code", ts_code)

        if start_date:
            query = query.gte("trade_date", start_date)
        else:
            # 默认最近一年
            default_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            query = query.gte("trade_date", default_start)

        if end_date:
            query = query.lte("trade_date", end_date)

        result = query.order("trade_date", desc=True).limit(limit).execute()

        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/{ts_code}/latest")
async def get_stock_latest(ts_code: str):
    """获取股票最新行情"""
    try:
        db = get_db()

        # 使用视图查询
        result = db.table("v_stock_latest").select("*").eq("ts_code", ts_code).single().execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="股票数据不存在")

        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== 技术指标接口 =====

@app.get("/api/stocks/{ts_code}/indicators")
async def get_technical_indicators(
    ts_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = Query(60, ge=1, le=500)
):
    """获取技术指标"""
    try:
        db = get_db()
        query = db.table("technical_indicators").select("*").eq("ts_code", ts_code)

        if start_date:
            query = query.gte("trade_date", start_date)
        if end_date:
            query = query.lte("trade_date", end_date)

        result = query.order("trade_date", desc=True).limit(limit).execute()

        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/{ts_code}/signals")
async def get_trading_signals(ts_code: str):
    """获取交易信号"""
    try:
        db = get_db()

        # 获取最新技术指标
        result = db.table("technical_indicators").select("*").eq("ts_code", ts_code).order("trade_date", desc=True).limit(5).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="技术指标数据不存在")

        latest = result.data[0]

        # 生成交易信号
        signals = []

        # MA 信号
        if latest.get("ma5") and latest.get("ma20"):
            if latest["ma5"] > latest["ma20"]:
                signals.append({"type": "MA", "signal": "BUY", "description": "MA5上穿MA20"})
            else:
                signals.append({"type": "MA", "signal": "SELL", "description": "MA5下穿MA20"})

        # MACD 信号
        if latest.get("macd_bar"):
            if latest["macd_bar"] > 0:
                signals.append({"type": "MACD", "signal": "BUY", "description": "MACD红柱"})
            else:
                signals.append({"type": "MACD", "signal": "SELL", "description": "MACD绿柱"})

        # RSI 信号
        if latest.get("rsi6"):
            if latest["rsi6"] < 30:
                signals.append({"type": "RSI", "signal": "BUY", "description": "RSI超卖"})
            elif latest["rsi6"] > 70:
                signals.append({"type": "RSI", "signal": "SELL", "description": "RSI超买"})

        return {
            "ts_code": ts_code,
            "trade_date": latest.get("trade_date"),
            "signals": signals,
            "indicators": latest
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== 预测接口 =====

@app.post("/api/predictions")
async def create_prediction(request: PredictionRequest):
    """创建预测"""
    try:
        # TODO: 实现预测逻辑
        return {
            "ts_code": request.ts_code,
            "status": "pending",
            "message": "预测任务已提交"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/{ts_code}")
async def get_predictions(
    ts_code: str,
    limit: int = Query(30, ge=1, le=100)
):
    """获取预测结果"""
    try:
        db = get_db()
        result = db.table("predictions").select("*").eq("ts_code", ts_code).order("pred_date", desc=True).limit(limit).execute()

        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== 回测接口 =====

@app.post("/api/backtests")
async def create_backtest(request: BacktestRequest):
    """创建回测任务"""
    try:
        # TODO: 实现回测逻辑
        return {
            "ts_code": request.ts_code,
            "status": "pending",
            "message": "回测任务已提交"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtests")
async def get_backtests(limit: int = Query(50, ge=1, le=100)):
    """获取回测结果列表"""
    try:
        db = get_db()
        result = db.table("backtest_results").select("*").order("created_at", desc=True).limit(limit).execute()

        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtests/{backtest_id}")
async def get_backtest_detail(backtest_id: str):
    """获取回测详情"""
    try:
        db = get_db()
        result = db.table("backtest_results").select("*").eq("id", backtest_id).single().execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="回测记录不存在")

        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== 市场概览接口 =====

@app.get("/api/market/overview")
async def get_market_overview():
    """获取市场概览"""
    try:
        db = get_db()

        # 获取统计数据
        stocks_count = db.table("stocks_info").select("count", count="exact").eq("list_status", "L").execute()

        # 获取最新交易日的涨跌统计
        latest_date_result = db.table("stock_daily").select("trade_date").order("trade_date", desc=True).limit(1).execute()
        latest_date = latest_date_result.data[0]["trade_date"] if latest_date_result.data else None

        stats = {
            "total_stocks": stocks_count.count if hasattr(stocks_count, 'count') else 0,
            "latest_trade_date": latest_date,
        }

        if latest_date:
            # 计算涨跌家数
            daily_data = db.table("stock_daily").select("pct_change").eq("trade_date", latest_date).execute()

            if daily_data.data:
                up_count = sum(1 for d in daily_data.data if d.get("pct_change", 0) > 0)
                down_count = sum(1 for d in daily_data.data if d.get("pct_change", 0) < 0)

                stats["up_stocks"] = up_count
                stats["down_stocks"] = down_count
                stats["flat_stocks"] = len(daily_data.data) - up_count - down_count

        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/hot")
async def get_hot_stocks(limit: int = Query(20, ge=1, le=100)):
    """获取热门股票 (涨幅排行)"""
    try:
        db = get_db()

        # 获取最新交易日涨幅前N的股票
        result = db.table("v_stock_latest").select("*").order("pct_change", desc=True).limit(limit).execute()

        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
