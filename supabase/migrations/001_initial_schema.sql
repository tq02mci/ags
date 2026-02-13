-- A股量化交易系统 - Supabase 数据库初始Schema
-- 执行: supabase db push

-- 启用必要的扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- 用于模糊搜索
-- CREATE EXTENSION IF NOT EXISTS "timescaledb";  -- 时序数据扩展（Supabase付费版支持）

-- ============================================
-- 1. 元数据表
-- ============================================

-- 股票基础信息表
CREATE TABLE stocks_info (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) UNIQUE NOT NULL,           -- Tushare代码 (如: 000001.SZ)
    symbol VARCHAR(10) NOT NULL,                    -- 股票代码 (如: 000001)
    name VARCHAR(100) NOT NULL,                     -- 股票名称
    area VARCHAR(50),                               -- 地区
    industry VARCHAR(100),                          -- 行业
    fullname VARCHAR(200),                          -- 全称
    enname VARCHAR(200),                            -- 英文名
    cnspell VARCHAR(50),                            -- 拼音
    market VARCHAR(20),                             -- 市场 (主板/创业板/科创板)
    exchange VARCHAR(10),                           -- 交易所 (SZ/SH/BJ)
    curr_type VARCHAR(10),                          -- 货币
    list_status VARCHAR(5),                         -- 上市状态 (L上市/D退市/P暂停)
    list_date DATE,                                 -- 上市日期
    delist_date DATE,                               -- 退市日期
    is_hs VARCHAR(5),                               -- 是否沪深港通 (N/H/S)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE stocks_info IS 'A股股票基础信息表';

-- 交易日历表
CREATE TABLE trade_calendar (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exchange VARCHAR(10) NOT NULL,                  -- 交易所
    cal_date DATE NOT NULL,                         -- 日历日期
    is_open BOOLEAN DEFAULT FALSE,                  -- 是否开盘
    pretrade_date DATE,                             -- 上一交易日
    UNIQUE(exchange, cal_date)
);

COMMENT ON TABLE trade_calendar IS 'A股交易日历';

-- ============================================
-- 2. 行情数据表
-- ============================================

-- 日线行情表 (时序数据)
CREATE TABLE stock_daily (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,                   -- 股票代码
    trade_date DATE NOT NULL,                       -- 交易日期
    open DECIMAL(12,4),                             -- 开盘价
    high DECIMAL(12,4),                             -- 最高价
    low DECIMAL(12,4),                              -- 最低价
    close DECIMAL(12,4),                            -- 收盘价
    pre_close DECIMAL(12,4),                        -- 昨收价
    change DECIMAL(12,4),                           -- 涨跌额
    pct_change DECIMAL(8,4),                        -- 涨跌幅(%)
    vol BIGINT,                                     -- 成交量(手)
    amount DECIMAL(18,4),                           -- 成交额(千元)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ts_code, trade_date)
);

-- 创建索引
CREATE INDEX idx_stock_daily_code_date ON stock_daily(ts_code, trade_date DESC);
CREATE INDEX idx_stock_daily_date ON stock_daily(trade_date DESC);

COMMENT ON TABLE stock_daily IS 'A股日线行情';

-- 分钟线行情表 (可选，大数据量)
CREATE TABLE stock_minute (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    trade_time TIMESTAMPTZ NOT NULL,                -- 交易时间
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    vol BIGINT,
    amount DECIMAL(18,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ts_code, trade_time)
);

CREATE INDEX idx_stock_minute_code_time ON stock_minute(ts_code, trade_time DESC);

COMMENT ON TABLE stock_minute IS 'A股分钟线行情';

-- 复权因子表
CREATE TABLE adj_factor (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    adj_factor DECIMAL(15,8),                       -- 复权因子
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ts_code, trade_date)
);

CREATE INDEX idx_adj_factor_code_date ON adj_factor(ts_code, trade_date DESC);

-- ============================================
-- 3. 财务数据表
-- ============================================

-- 利润表
CREATE TABLE income_statement (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    ann_date DATE,                                  -- 公告日期
    f_ann_date DATE,                                -- 实际公告日期
    end_date DATE NOT NULL,                         -- 报告期
    comp_type VARCHAR(5),                           -- 公司类型
    basic_eps DECIMAL(15,4),                        -- 基本每股收益
    diluted_eps DECIMAL(15,4),                      -- 稀释每股收益
    total_revenue DECIMAL(20,4),                    -- 营业总收入
    revenue DECIMAL(20,4),                          -- 营业收入
    total_cogs DECIMAL(20,4),                       -- 营业总成本
    gross_profit DECIMAL(20,4),                     -- 毛利润
    oper_exp DECIMAL(20,4),                         -- 营业支出
    oper_profit DECIMAL(20,4),                      -- 营业利润
    total_profit DECIMAL(20,4),                     -- 利润总额
    net_income DECIMAL(20,4),                       -- 净利润
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ts_code, end_date)
);

CREATE INDEX idx_income_code_date ON income_statement(ts_code, end_date DESC);

-- 资产负债表
CREATE TABLE balance_sheet (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    ann_date DATE,
    f_ann_date DATE,
    end_date DATE NOT NULL,
    total_assets DECIMAL(20,4),                     -- 资产总计
    total_liab DECIMAL(20,4),                       -- 负债合计
    total_hldr_eqy_exc_min_int DECIMAL(20,4),       -- 股东权益合计
    total_hldr_eqy_inc_min_int DECIMAL(20,4),       -- 股东权益合计(含少数股东)
    trad_asset DECIMAL(20,4),                       -- 流动资产
    notes_receiv DECIMAL(20,4),                     -- 应收票据
    accounts_receiv DECIMAL(20,4),                  -- 应收账款
    inventories DECIMAL(20,4),                      -- 存货
    fix_assets DECIMAL(20,4),                       -- 固定资产
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ts_code, end_date)
);

CREATE INDEX idx_balance_code_date ON balance_sheet(ts_code, end_date DESC);

-- 现金流量表
CREATE TABLE cash_flow (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    ann_date DATE,
    f_ann_date DATE,
    end_date DATE NOT NULL,
    n_cashflow_act DECIMAL(20,4),                   -- 经营活动现金流净额
    n_cashflow_inv_act DECIMAL(20,4),               -- 投资活动现金流净额
    n_cash_finc_act DECIMAL(20,4),                  -- 筹资活动现金流净额
    free_cashflow DECIMAL(20,4),                    -- 自由现金流
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ts_code, end_date)
);

-- 财务指标表
CREATE TABLE financial_indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    ann_date DATE,
    end_date DATE NOT NULL,
    eps DECIMAL(15,4),                              -- 每股收益
    dt_eps DECIMAL(15,4),                           -- 稀释每股收益
    total_revenue_ps DECIMAL(15,4),                 -- 每股营业收入
    revenue_ps DECIMAL(15,4),                       -- 每股营业收入
    capital_rese_ps DECIMAL(15,4),                  -- 每股资本公积
    surplus_rese_ps DECIMAL(15,4),                  -- 每股盈余公积
    undist_profit_ps DECIMAL(15,4),                 -- 每股未分配利润
    extra_item DECIMAL(15,4),                       -- 非经常性损益
    profit_dedt DECIMAL(20,4),                      -- 扣非净利润
    gross_margin DECIMAL(8,4),                      -- 毛利率
    current_ratio DECIMAL(8,4),                     -- 流动比率
    quick_ratio DECIMAL(8,4),                       -- 速动比率
    cash_ratio DECIMAL(8,4),                        -- 现金比率
    invturn_days DECIMAL(8,2),                      -- 存货周转天数
    arturn_days DECIMAL(8,2),                       -- 应收账款周转天数
    inv_turn DECIMAL(8,4),                          -- 存货周转率
    ar_turn DECIMAL(8,4),                           -- 应收账款周转率
    assets_turn DECIMAL(8,4),                       -- 总资产周转率
    bps DECIMAL(15,4),                              -- 每股净资产
    ocfps DECIMAL(15,4),                            -- 每股经营现金流
    gr_ps DECIMAL(15,4),                            -- 每股营业总收入
    or_ps DECIMAL(15,4),                            -- 每股营业收入
    net_cfps DECIMAL(15,4),                         -- 每股现金流量净额
    debt_to_assets DECIMAL(8,4),                    -- 资产负债率
    roe DECIMAL(8,4),                               -- 净资产收益率
    roe_waa DECIMAL(8,4),                           -- 加权平均净资产收益率
    roe_dt DECIMAL(8,4),                            -- 扣非净资产收益率
    roa DECIMAL(8,4),                               -- 总资产报酬率
    roic DECIMAL(8,4),                              -- 投入资本回报率
    grossprofit_margin DECIMAL(8,4),                -- 销售毛利率
    op_of_gr DECIMAL(8,4),                          -- 营业利润率
    profit_to_gr DECIMAL(8,4),                      -- 净利润率
    salescash_to_or DECIMAL(8,4),                   -- 销售现金比率
    ocf_to_or DECIMAL(8,4),                         -- 经营活动产生的现金流量净额/营业收入
    ocf_to_opincome DECIMAL(8,4),                   -- 经营活动产生的现金流量净额/经营活动净收益
    basic_eps_yoy DECIMAL(8,4),                     -- 每股收益同比增长率
    dt_eps_yoy DECIMAL(8,4),                        -- 稀释每股收益同比增长率
    cfps_yoy DECIMAL(8,4),                          -- 每股现金流量净额同比增长率
    op_yoy DECIMAL(8,4),                            -- 营业利润同比增长率
    ebt_yoy DECIMAL(8,4),                           -- 利润总额同比增长率
    netprofit_yoy DECIMAL(8,4),                     -- 净利润同比增长率
    dt_netprofit_yoy DECIMAL(8,4),                  -- 扣非净利润同比增长率
    roe_yoy DECIMAL(8,4),                           -- 净资产收益率同比增长率
    bps_yoy DECIMAL(8,4),                           -- 每股净资产同比增长率
    assets_yoy DECIMAL(8,4),                        -- 总资产同比增长率
    eqt_yoy DECIMAL(8,4),                           -- 股东权益同比增长率
    tr_yoy DECIMAL(8,4),                            -- 营业总收入同比增长率
    or_yoy DECIMAL(8,4),                            -- 营业收入同比增长率
    q_sales_yoy DECIMAL(8,4),                       -- 营业收入季度同比增长率
    q_op_qoq DECIMAL(8,4),                          -- 营业利润季度环比增长率
    equity_yoy DECIMAL(8,4),                        -- 股东权益同比增长率
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ts_code, end_date)
);

CREATE INDEX idx_fin_indicators_code_date ON financial_indicators(ts_code, end_date DESC);

-- ============================================
-- 4. 技术指标表
-- ============================================

CREATE TABLE technical_indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,

    -- 移动平均线
    ma5 DECIMAL(12,4),
    ma10 DECIMAL(12,4),
    ma20 DECIMAL(12,4),
    ma30 DECIMAL(12,4),
    ma60 DECIMAL(12,4),
    ma120 DECIMAL(12,4),
    ma250 DECIMAL(12,4),

    -- MACD
    macd_dif DECIMAL(12,4),
    macd_dea DECIMAL(12,4),
    macd_bar DECIMAL(12,4),

    -- KDJ
    kdj_k DECIMAL(8,4),
    kdj_d DECIMAL(8,4),
    kdj_j DECIMAL(8,4),

    -- RSI
    rsi6 DECIMAL(8,4),
    rsi12 DECIMAL(8,4),
    rsi24 DECIMAL(8,4),

    -- BOLL
    boll_upper DECIMAL(12,4),
    boll_mid DECIMAL(12,4),
    boll_lower DECIMAL(12,4),

    -- 其他指标
    cci DECIMAL(12,4),
    wr DECIMAL(8,4),
    obv BIGINT,
    atr DECIMAL(12,4),

    -- 成交量指标
    vol_ma5 BIGINT,
    vol_ma10 BIGINT,
    vol_ma20 BIGINT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ts_code, trade_date)
);

CREATE INDEX idx_tech_indicators_code_date ON technical_indicators(ts_code, trade_date DESC);
CREATE INDEX idx_tech_indicators_date ON technical_indicators(trade_date DESC);

COMMENT ON TABLE technical_indicators IS '股票技术指标表';

-- ============================================
-- 5. 预测和回测表
-- ============================================

-- 模型配置表
CREATE TABLE model_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,                -- lstm, xgboost, etc.
    version VARCHAR(20) NOT NULL,
    params JSONB,                                   -- 模型参数
    features JSONB,                                 -- 特征列表
    description TEXT,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(model_name, version)
);

-- 预测结果表
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    model_id UUID REFERENCES model_configs(id),
    pred_date DATE NOT NULL,                        -- 预测日期
    target_date DATE NOT NULL,                      -- 目标日期

    -- 预测值
    pred_direction INT,                             -- 预测方向 (-1, 0, 1)
    pred_return DECIMAL(8,4),                       -- 预测收益率
    pred_price DECIMAL(12,4),                       -- 预测价格
    confidence DECIMAL(5,4),                        -- 置信度

    -- 实际值（后续回填）
    actual_direction INT,
    actual_return DECIMAL(8,4),
    actual_price DECIMAL(12,4),

    -- 评估
    is_correct BOOLEAN,
    error DECIMAL(8,4),

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_predictions_code_date ON predictions(ts_code, pred_date DESC);
CREATE INDEX idx_predictions_model ON predictions(model_id, pred_date DESC);

-- 回测结果表
CREATE TABLE backtest_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    backtest_name VARCHAR(200) NOT NULL,
    model_id UUID REFERENCES model_configs(id),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,

    -- 回测参数
    initial_capital DECIMAL(18,4),
    strategy_params JSONB,

    -- 结果指标
    total_return DECIMAL(10,4),                     -- 总收益率
    annual_return DECIMAL(10,4),                    -- 年化收益率
    max_drawdown DECIMAL(10,4),                     -- 最大回撤
    sharpe_ratio DECIMAL(10,4),                     -- 夏普比率
    sortino_ratio DECIMAL(10,4),                    -- 索提诺比率
    calmar_ratio DECIMAL(10,4),                     -- 卡玛比率
    win_rate DECIMAL(6,4),                          -- 胜率
    profit_factor DECIMAL(10,4),                    -- 盈亏比

    -- 交易统计
    total_trades INT,
    winning_trades INT,
    losing_trades INT,
    avg_profit DECIMAL(12,4),
    avg_loss DECIMAL(12,4),

    -- 详细结果 (JSON)
    trades_detail JSONB,
    daily_nav JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_backtest_model ON backtest_results(model_id);
CREATE INDEX idx_backtest_date ON backtest_results(created_at DESC);

-- ============================================
-- 6. 数据同步日志表
-- ============================================

CREATE TABLE data_sync_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sync_type VARCHAR(50) NOT NULL,                 -- daily, minute, financial, etc.
    sync_date DATE,                                 -- 同步日期
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    status VARCHAR(20),                             -- success, failed, running
    records_count INT,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_sync_log_type_date ON data_sync_log(sync_type, sync_date DESC);

-- ============================================
-- 7. 创建触发器函数 (自动更新 updated_at)
-- ============================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 应用触发器
CREATE TRIGGER update_stocks_info_updated_at
    BEFORE UPDATE ON stocks_info
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_configs_updated_at
    BEFORE UPDATE ON model_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_predictions_updated_at
    BEFORE UPDATE ON predictions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 8. 创建 RLS 策略 (行级安全)
-- ============================================

-- 为所有表启用 RLS
ALTER TABLE stocks_info ENABLE ROW LEVEL SECURITY;
ALTER TABLE stock_daily ENABLE ROW LEVEL SECURITY;
ALTER TABLE technical_indicators ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE backtest_results ENABLE ROW LEVEL SECURITY;

-- 允许所有用户读取
CREATE POLICY "Allow all read" ON stocks_info FOR SELECT USING (true);
CREATE POLICY "Allow all read" ON stock_daily FOR SELECT USING (true);
CREATE POLICY "Allow all read" ON technical_indicators FOR SELECT USING (true);
CREATE POLICY "Allow all read" ON predictions FOR SELECT USING (true);
CREATE POLICY "Allow all read" ON backtest_results FOR SELECT USING (true);

-- 只允许服务角色写入
CREATE POLICY "Allow service write" ON stocks_info
    FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');

-- ============================================
-- 9. 创建视图
-- ============================================

-- 股票最新行情视图
CREATE VIEW v_stock_latest AS
SELECT
    s.ts_code,
    s.name,
    s.industry,
    d.trade_date,
    d.open,
    d.high,
    d.low,
    d.close,
    d.pct_change,
    d.vol,
    d.amount,
    t.ma5,
    t.ma20,
    t.macd_bar,
    t.rsi6
FROM stocks_info s
LEFT JOIN LATERAL (
    SELECT * FROM stock_daily
    WHERE ts_code = s.ts_code
    ORDER BY trade_date DESC
    LIMIT 1
) d ON true
LEFT JOIN LATERAL (
    SELECT * FROM technical_indicators
    WHERE ts_code = s.ts_code
    ORDER BY trade_date DESC
    LIMIT 1
) t ON true
WHERE s.list_status = 'L';

-- 股票综合视图 (行情+指标+估值)
CREATE VIEW v_stock_overview AS
SELECT
    s.ts_code,
    s.name,
    s.industry,
    d.close,
    d.pct_change,
    d.vol,
    f.eps,
    f.bps,
    f.roe,
    f.gross_margin,
    t.ma5,
    t.ma20,
    t.macd_bar,
    t.rsi6
FROM stocks_info s
LEFT JOIN LATERAL (
    SELECT * FROM stock_daily
    WHERE ts_code = s.ts_code
    ORDER BY trade_date DESC
    LIMIT 1
) d ON true
LEFT JOIN LATERAL (
    SELECT * FROM financial_indicators
    WHERE ts_code = s.ts_code
    ORDER BY end_date DESC
    LIMIT 1
) f ON true
LEFT JOIN LATERAL (
    SELECT * FROM technical_indicators
    WHERE ts_code = s.ts_code
    ORDER BY trade_date DESC
    LIMIT 1
) t ON true
WHERE s.list_status = 'L';

-- ============================================
-- 10. 初始化数据
-- ============================================

-- 插入交易所交易日历示例数据
INSERT INTO trade_calendar (exchange, cal_date, is_open) VALUES
('SSE', '2024-01-02', true),
('SSE', '2024-01-03', true),
('SSE', '2024-01-04', true),
('SZSE', '2024-01-02', true),
('SZSE', '2024-01-03', true),
('SZSE', '2024-01-04', true);

-- 插入示例模型配置
INSERT INTO model_configs (model_name, model_type, version, params, features, description, is_active) VALUES
('LSTM_Baseline', 'lstm', '1.0.0',
 '{"hidden_size": 128, "num_layers": 2, "dropout": 0.2, "learning_rate": 0.001}'::jsonb,
 '["close", "vol", "ma5", "ma20", "rsi6", "macd_bar"]'::jsonb,
 '基础LSTM预测模型', true),
('XGBoost_Trend', 'xgboost', '1.0.0',
 '{"max_depth": 6, "n_estimators": 100, "learning_rate": 0.1}'::jsonb,
 '["close", "vol", "ma5", "ma20", "rsi6", "macd_bar", "pe", "pb"]'::jsonb,
 'XGBoost趋势预测模型', false);
