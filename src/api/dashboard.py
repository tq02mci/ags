"""
A股量化交易系统 - Streamlit 可视化界面
"""
import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# API 基础 URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# 页面配置
st.set_page_config(
    page_title="A股量化交易系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .up { color: #ff4d4f; }
    .down { color: #52c41a; }
</style>
""", unsafe_allow_html=True)


def fetch_api(endpoint, params=None):
    """调用 API"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API 调用失败: {e}")
        return None


def format_change(value):
    """格式化涨跌幅"""
    if value is None:
        return "-"
    color = "up" if value > 0 else "down" if value < 0 else ""
    symbol = "+" if value > 0 else ""
    return f'<span class="{color}">{symbol}{value:.2f}%</span>'


# 侧边栏导航
st.sidebar.title("📊 导航")
page = st.sidebar.radio(
    "选择页面",
    ["🏠 首页", "🔍 股票查询", "📈 技术分析", "🤖 预测模型", "📋 回测", "⚙️ 数据管理"]
)

# ===== 首页 =====
if page == "🏠 首页":
    st.markdown("<h1 class='main-header'>📈 A股量化交易系统</h1>", unsafe_allow_html=True)

    # 市场概览
    st.subheader("市场概览")

    overview = fetch_api("/api/market/overview")
    if overview:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("上市公司", overview.get("total_stocks", 0))
        with col2:
            st.metric("上涨家数", overview.get("up_stocks", 0), delta_color="normal")
        with col3:
            st.metric("下跌家数", overview.get("down_stocks", 0), delta_color="inverse")
        with col4:
            st.metric("平盘家数", overview.get("flat_stocks", 0))

    # 热门股票
    st.subheader("🔥 涨幅榜")
    hot_stocks = fetch_api("/api/market/hot", {"limit": 20})
    if hot_stocks:
        df = pd.DataFrame(hot_stocks)
        df = df[["ts_code", "name", "industry", "close", "pct_change", "vol"]]
        df.columns = ["代码", "名称", "行业", "最新价", "涨跌幅", "成交量"]

        # 格式化涨跌幅
        def color_change(val):
            color = "red" if val > 0 else "green" if val < 0 else "gray"
            return f"color: {color}"

        styled_df = df.style.applymap(color_change, subset=["涨跌幅"])
        st.dataframe(styled_df, use_container_width=True)

    # 快速搜索
    st.subheader("🔍 快速搜索")
    search_code = st.text_input("输入股票代码", placeholder="如: 000001.SZ")
    if search_code:
        st.session_state["selected_stock"] = search_code
        st.switch_page("pages/1_stock_query.py")

# ===== 股票查询 =====
elif page == "🔍 股票查询":
    st.title("🔍 股票查询")

    # 搜索框
    col1, col2 = st.columns([3, 1])
    with col1:
        stock_code = st.text_input("股票代码", "000001.SZ", key="stock_search")
    with col2:
        st.write("")
        st.write("")
        search_btn = st.button("🔍 查询", type="primary")

    if stock_code and search_btn:
        # 获取股票详情
        stock_info = fetch_api(f"/api/stocks/{stock_code}")

        if stock_info:
            st.success(f"📌 {stock_info.get('name', '')} ({stock_code})")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("行业", stock_info.get("industry", "-"))
            with col2:
                st.metric("交易所", stock_info.get("exchange", "-"))
            with col3:
                st.metric("上市日期", stock_info.get("list_date", "-"))
            with col4:
                st.metric("市场", stock_info.get("market", "-"))

            # 获取日线数据
            daily_data = fetch_api(f"/api/stocks/{stock_code}/daily", {"limit": 252})

            if daily_data:
                df = pd.DataFrame(daily_data)
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date")

                # K线图
                st.subheader("📊 K线图")

                fig = go.Figure(data=[go.Candlestick(
                    x=df["trade_date"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="K线"
                )])

                fig.update_layout(
                    title=f"{stock_code} K线图",
                    xaxis_title="日期",
                    yaxis_title="价格",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # 成交量图
                st.subheader("📊 成交量")

                fig_vol = go.Figure(data=[go.Bar(
                    x=df["trade_date"],
                    y=df["vol"],
                    name="成交量",
                    marker_color="blue"
                )])

                fig_vol.update_layout(
                    xaxis_title="日期",
                    yaxis_title="成交量",
                    height=300
                )

                st.plotly_chart(fig_vol, use_container_width=True)

                # 数据表格
                st.subheader("📋 历史数据")
                st.dataframe(df[["trade_date", "open", "high", "low", "close", "vol", "pct_change"]].tail(20))

# ===== 技术分析 =====
elif page == "📈 技术分析":
    st.title("📈 技术分析")

    stock_code = st.text_input("股票代码", "000001.SZ")

    if stock_code:
        indicators = fetch_api(f"/api/stocks/{stock_code}/indicators", {"limit": 120})

        if indicators:
            df = pd.DataFrame(indicators)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date")

            # 技术指标选择
            st.subheader("选择指标")
            cols = st.columns(4)
            with cols[0]:
                show_ma = st.checkbox("移动平均线", True)
            with cols[1]:
                show_macd = st.checkbox("MACD", True)
            with cols[2]:
                show_rsi = st.checkbox("RSI", True)
            with cols[3]:
                show_boll = st.checkbox("布林带", True)

            # MA 图表
            if show_ma and "ma5" in df.columns:
                st.subheader("移动平均线")
                fig = go.Figure()

                fig.add_trace(go.Scatter(x=df["trade_date"], y=df["close"], name="收盘价", line=dict(color="black")))

                for ma, color in [("ma5", "orange"), ("ma10", "blue"), ("ma20", "red"), ("ma60", "green")]:
                    if ma in df.columns:
                        fig.add_trace(go.Scatter(x=df["trade_date"], y=df[ma], name=ma.upper(), line=dict(color=color)))

                fig.update_layout(height=400, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            # MACD 图表
            if show_macd and "macd_bar" in df.columns:
                st.subheader("MACD")

                fig = go.Figure()
                colors = ["red" if v > 0 else "green" for v in df["macd_bar"]]

                fig.add_trace(go.Bar(x=df["trade_date"], y=df["macd_bar"], name="MACD柱状", marker_color=colors))
                fig.add_trace(go.Scatter(x=df["trade_date"], y=df["macd_dif"], name="DIF", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=df["trade_date"], y=df["macd_dea"], name="DEA", line=dict(color="orange")))

                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            # RSI 图表
            if show_rsi and "rsi6" in df.columns:
                st.subheader("RSI")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["trade_date"], y=df["rsi6"], name="RSI6", line=dict(color="blue")))

                if "rsi12" in df.columns:
                    fig.add_trace(go.Scatter(x=df["trade_date"], y=df["rsi12"], name="RSI12", line=dict(color="orange")))

                if "rsi24" in df.columns:
                    fig.add_trace(go.Scatter(x=df["trade_date"], y=df["rsi24"], name="RSI24", line=dict(color="green")))

                # 添加超买超卖线
                fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="超买")
                fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="超卖")

                fig.update_layout(height=300, yaxis_range=[0, 100])
                st.plotly_chart(fig, use_container_width=True)

            # 布林带图表
            if show_boll and "boll_upper" in df.columns:
                st.subheader("布林带")

                fig = go.Figure()

                fig.add_trace(go.Scatter(x=df["trade_date"], y=df["close"], name="收盘价", line=dict(color="black")))
                fig.add_trace(go.Scatter(x=df["trade_date"], y=df["boll_upper"], name="上轨", line=dict(color="red", dash="dash")))
                fig.add_trace(go.Scatter(x=df["trade_date"], y=df["boll_mid"], name="中轨", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=df["trade_date"], y=df["boll_lower"], name="下轨", line=dict(color="green", dash="dash")))

                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

# ===== 预测模型 =====
elif page == "🤖 预测模型":
    st.title("🤖 预测模型")

    stock_code = st.text_input("股票代码", "000001.SZ")

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("模型类型", ["LSTM", "XGBoost", "LightGBM", "集成模型"])
    with col2:
        pred_days = st.slider("预测天数", 1, 30, 5)

    if st.button("🚀 开始预测", type="primary"):
        with st.spinner("模型预测中..."):
            # TODO: 调用预测 API
            st.info("预测功能开发中...")

            # 模拟结果
            st.success("预测完成!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("预测方向", "上涨", "+65% 概率")
            with col2:
                st.metric("预测收益率", "3.2%", "±1.5%")
            with col3:
                st.metric("置信度", "72%", "高")

# ===== 回测 =====
elif page == "📋 回测":
    st.title("📋 策略回测")

    col1, col2 = st.columns(2)
    with col1:
        stock_code = st.text_input("股票代码", "000001.SZ")
    with col2:
        strategy = st.selectbox("策略", ["MACD金叉死叉", "均线突破", "RSI超买卖", "布林带突破", "多因子组合"])

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("结束日期", datetime.now())

    initial_capital = st.slider("初始资金", 100000, 10000000, 1000000, 100000)

    if st.button("▶️ 开始回测", type="primary"):
        with st.spinner("回测运行中..."):
            # TODO: 调用回测 API
            st.info("回测功能开发中...")

            # 模拟结果
            st.success("回测完成!")

            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            with metrics_col1:
                st.metric("总收益率", "32.5%", "+15.2% vs 沪深300")
            with metrics_col2:
                st.metric("年化收益率", "28.3%", "+12.1% vs 基准")
            with metrics_col3:
                st.metric("最大回撤", "-12.8%", "中等风险")
            with metrics_col4:
                st.metric("夏普比率", "1.85", "优秀")

            # 收益曲线
            st.subheader("📈 收益曲线")
            st.line_chart([1.0, 1.05, 1.12, 1.08, 1.15, 1.22, 1.18, 1.28, 1.35, 1.32, 1.45])

            # 交易记录
            st.subheader("📋 交易记录")
            trades = pd.DataFrame({
                "日期": ["2024-01-15", "2024-02-01", "2024-03-10"],
                "操作": ["买入", "卖出", "买入"],
                "价格": [10.5, 12.3, 11.8],
                "数量": [1000, 1000, 1000],
                "盈亏": ["-", "+1800", "-"]
            })
            st.dataframe(trades)

# ===== 数据管理 =====
elif page == "⚙️ 数据管理":
    st.title("⚙️ 数据管理")

    # 系统状态
    st.subheader("系统状态")

    health = fetch_api("/health")
    if health:
        if health.get("status") == "healthy":
            st.success(f"✅ 系统运行正常 | 数据库: {health.get('database', 'unknown')}")
        else:
            st.error(f"❌ 系统异常: {health.get('error', 'unknown')}")

    # 数据同步
    st.subheader("📥 数据同步")

    sync_col1, sync_col2, sync_col3 = st.columns(3)

    with sync_col1:
        if st.button("🔄 同步股票列表"):
            st.info("正在同步股票列表...")

    with sync_col2:
        if st.button("🔄 同步日线数据"):
            st.info("正在同步日线数据...")

    with sync_col3:
        if st.button("🔄 同步财务数据"):
            st.info("正在同步财务数据...")

    # 数据概览
    st.subheader("📊 数据概览")

    overview = fetch_api("/api/market/overview")
    if overview:
        st.json(overview)
