-- A股量化交易系统 - 资讯数据表
-- 新闻、公告、研报、龙虎榜等

-- ============================================
-- 1. 新闻资讯表
-- ============================================

-- 个股新闻表
CREATE TABLE stock_news (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    news_type VARCHAR(20) DEFAULT 'stock',  -- stock, market, industry
    title VARCHAR(500) NOT NULL,
    content TEXT,
    summary TEXT,                           -- 摘要
    keywords VARCHAR(300),
    source VARCHAR(100),
    publish_time TIMESTAMPTZ,
    url TEXT,
    sentiment_score DECIMAL(4,3),           -- 情感分析得分 (-1 到 1)
    sentiment_label VARCHAR(10),            -- positive, negative, neutral
    importance_score INT DEFAULT 0,         -- 重要度评分
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ts_code, title)
);

CREATE INDEX idx_stock_news_code ON stock_news(ts_code, publish_time DESC);
CREATE INDEX idx_stock_news_time ON stock_news(publish_time DESC);
CREATE INDEX idx_stock_news_sentiment ON stock_news(sentiment_label);

COMMENT ON TABLE stock_news IS '个股新闻资讯';

-- 市场重大新闻表
CREATE TABLE market_news (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    news_type VARCHAR(20) DEFAULT 'major',  -- major, policy, global
    title VARCHAR(500) NOT NULL,
    content TEXT,
    summary TEXT,
    source VARCHAR(100),
    publish_time TIMESTAMPTZ,
    url TEXT,
    affect_scope VARCHAR(50),               -- 影响范围: market, industry, stock
    affected_industries JSONB,              -- 影响的行业列表
    sentiment_score DECIMAL(4,3),
    sentiment_label VARCHAR(10),
    importance_score INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_market_news_time ON market_news(publish_time DESC);
CREATE INDEX idx_market_news_type ON market_news(news_type);

COMMENT ON TABLE market_news IS '市场重大新闻/财经要闻';

-- ============================================
-- 2. 公司公告表
-- ============================================

CREATE TABLE stock_notices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    notice_type VARCHAR(100),               -- 公告类型
    title VARCHAR(500) NOT NULL,
    content TEXT,
    publish_date DATE NOT NULL,
    url TEXT,
    attachment_url TEXT,                    -- 附件链接
    is_important BOOLEAN DEFAULT FALSE,     -- 是否重要公告
    affect_type VARCHAR(50),                -- 影响类型: positive, negative, neutral
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ts_code, title, publish_date)
);

CREATE INDEX idx_notices_code ON stock_notices(ts_code, publish_date DESC);
CREATE INDEX idx_notices_type ON stock_notices(notice_type);
CREATE INDEX idx_notices_date ON stock_notices(publish_date DESC);

COMMENT ON TABLE stock_notices IS '上市公司公告';

-- ============================================
-- 3. 研究报告表
-- ============================================

CREATE TABLE research_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20),
    symbol VARCHAR(10),
    report_date DATE,
    institution VARCHAR(200),               -- 研究机构
    author VARCHAR(100),                    -- 分析师
    title VARCHAR(500),
    summary TEXT,
    rating VARCHAR(50),                     -- 评级: 买入,增持,中性,减持
    rating_change VARCHAR(50),              -- 评级变动: 上调,维持,下调
    target_price DECIMAL(10,2),             -- 目标价
    current_price DECIMAL(10,2),            -- 当前价
    upside DECIMAL(6,2),                    -- 上涨空间(%)
    industry VARCHAR(100),
    report_type VARCHAR(50),                -- 研报类型: 首次覆盖,业绩点评,行业深度
    url TEXT,
    pages INT,                              -- 页数
    importance_score INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_reports_code ON research_reports(ts_code, report_date DESC);
CREATE INDEX idx_reports_inst ON research_reports(institution);
CREATE INDEX idx_reports_rating ON research_reports(rating);

COMMENT ON TABLE research_reports IS '券商研究报告';

-- ============================================
-- 4. 龙虎榜数据表
-- ============================================

CREATE TABLE longhu_bang (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    name VARCHAR(100),
    close DECIMAL(10,2),
    pct_change DECIMAL(8,2),                -- 涨跌幅
    turnover DECIMAL(10,2),                 -- 换手率
    total_amount DECIMAL(18,4),             -- 总成交额
    net_amount DECIMAL(18,4),               -- 龙虎榜净买入额
    buy_amount DECIMAL(18,4),               -- 买入额
    sell_amount DECIMAL(18,4),              -- 卖出额
    reason VARCHAR(200),                    -- 上榜原因
    dept_names JSONB,                       -- 营业部名称列表
    is_continuous BOOLEAN DEFAULT FALSE,    -- 是否连续上榜
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ts_code, trade_date)
);

CREATE INDEX idx_longhu_code ON longhu_bang(ts_code, trade_date DESC);
CREATE INDEX idx_longhu_date ON longhu_bang(trade_date DESC);
CREATE INDEX idx_longhu_net ON longhu_bang(net_amount DESC);

COMMENT ON TABLE longhu_bang IS '龙虎榜数据';

-- 龙虎榜详情表 (营业部明细)
CREATE TABLE longhu_detail (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    dept_name VARCHAR(200),                 -- 营业部名称
    dept_code VARCHAR(50),                  -- 营业部代码
    buy_amount DECIMAL(18,4),
    buy_ratio DECIMAL(6,4),                 -- 买入占比
    sell_amount DECIMAL(18,4),
    sell_ratio DECIMAL(6,4),                -- 卖出占比
    net_amount DECIMAL(18,4),
    is_institution BOOLEAN DEFAULT FALSE,   -- 是否机构席位
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_longhu_detail_code ON longhu_detail(ts_code, trade_date DESC);
CREATE INDEX idx_longhu_detail_dept ON longhu_detail(dept_name);

COMMENT ON TABLE longhu_detail IS '龙虎榜营业部明细';

-- ============================================
-- 5. 大宗交易表
-- ============================================

CREATE TABLE dzjy (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    price DECIMAL(10,2),
    volume BIGINT,
    amount DECIMAL(18,4),
    buyer VARCHAR(200),                     -- 买方营业部
    seller VARCHAR(200),                    -- 卖方营业部
    close DECIMAL(10,2),                    -- 当日收盘价
    discount_rate DECIMAL(6,2),             -- 折价率
    is_premium BOOLEAN DEFAULT FALSE,       -- 是否溢价
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_dzjy_code ON dzjy(ts_code, trade_date DESC);
CREATE INDEX idx_dzjy_date ON dzjy(trade_date DESC);

COMMENT ON TABLE dzjy IS '大宗交易数据';

-- ============================================
-- 6. 资金流向表
-- ============================================

CREATE TABLE money_flow (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    close DECIMAL(10,2),
    pct_change DECIMAL(8,2),

    -- 主力净流入
    main_net_amount DECIMAL(18,4),
    main_net_ratio DECIMAL(8,4),            -- 主力净流入占比

    -- 超大单
    huge_buy DECIMAL(18,4),
    huge_sell DECIMAL(18,4),
    huge_net DECIMAL(18,4),

    -- 大单
    large_buy DECIMAL(18,4),
    large_sell DECIMAL(18,4),
    large_net DECIMAL(18,4),

    -- 中单
    medium_buy DECIMAL(18,4),
    medium_sell DECIMAL(18,4),
    medium_net DECIMAL(18,4),

    -- 小单
    small_buy DECIMAL(18,4),
    small_sell DECIMAL(18,4),
    small_net DECIMAL(18,4),

    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ts_code, trade_date)
);

CREATE INDEX idx_moneyflow_code ON money_flow(ts_code, trade_date DESC);
CREATE INDEX idx_moneyflow_date ON money_flow(trade_date DESC);

COMMENT ON TABLE money_flow IS '个股资金流向';

-- ============================================
-- 7. 融资融券表
-- ============================================

CREATE TABLE margin_trading (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,

    -- 融资
    rzye DECIMAL(18,4),                     -- 融资余额
    rzmre DECIMAL(18,4),                    -- 融资买入额
    rzche DECIMAL(18,4),                    -- 融资偿还额
    rzjmre DECIMAL(18,4),                   -- 融资净买入

    -- 融券
    rqye DECIMAL(18,4),                     -- 融券余额
    rqmcl DECIMAL(18,4),                    -- 融券卖出量
    rqchl DECIMAL(18,4),                    -- 融券偿还量
    rqjmcl DECIMAL(18,4),                   -- 融券净卖出

    -- 汇总
    rqye_sum DECIMAL(18,4),                 -- 融资融券余额
    rzrqye DECIMAL(18,4),                   -- 融资融券余额差值

    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ts_code, trade_date)
);

CREATE INDEX idx_margin_code ON margin_trading(ts_code, trade_date DESC);
CREATE INDEX idx_margin_date ON margin_trading(trade_date DESC);

COMMENT ON TABLE margin_trading IS '融资融券数据';

-- ============================================
-- 8. 创建视图
-- ============================================

-- 个股综合资讯视图
CREATE VIEW v_stock_news_summary AS
SELECT
    s.ts_code,
    s.name,
    s.industry,
    COUNT(DISTINCT n.id) AS news_count_7d,
    COUNT(DISTINCT nt.id) AS notice_count_7d,
    COUNT(DISTINCT r.id) AS report_count_7d,
    MAX(n.publish_time) AS latest_news_time,
    AVG(n.sentiment_score) AS avg_sentiment
FROM stocks_info s
LEFT JOIN stock_news n ON s.ts_code = n.ts_code AND n.publish_time > NOW() - INTERVAL '7 days'
LEFT JOIN stock_notices nt ON s.ts_code = nt.ts_code AND nt.publish_date > CURRENT_DATE - 7
LEFT JOIN research_reports r ON s.ts_code = r.ts_code AND r.report_date > CURRENT_DATE - 7
WHERE s.list_status = 'L'
GROUP BY s.ts_code, s.name, s.industry;

-- ============================================
-- 9. 启用 RLS
-- ============================================

ALTER TABLE stock_news ENABLE ROW LEVEL SECURITY;
ALTER TABLE market_news ENABLE ROW LEVEL SECURITY;
ALTER TABLE stock_notices ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE longhu_bang ENABLE ROW LEVEL SECURITY;
ALTER TABLE longhu_detail ENABLE ROW LEVEL SECURITY;
ALTER TABLE dzjy ENABLE ROW LEVEL SECURITY;
ALTER TABLE money_flow ENABLE ROW LEVEL SECURITY;
ALTER TABLE margin_trading ENABLE ROW LEVEL SECURITY;

-- 允许所有用户读取
CREATE POLICY "Allow all read" ON stock_news FOR SELECT USING (true);
CREATE POLICY "Allow all read" ON market_news FOR SELECT USING (true);
CREATE POLICY "Allow all read" ON stock_notices FOR SELECT USING (true);
CREATE POLICY "Allow all read" ON research_reports FOR SELECT USING (true);
CREATE POLICY "Allow all read" ON longhu_bang FOR SELECT USING (true);
CREATE POLICY "Allow all read" ON longhu_detail FOR SELECT USING (true);
CREATE POLICY "Allow all read" ON dzjy FOR SELECT USING (true);
CREATE POLICY "Allow all read" ON money_flow FOR SELECT USING (true);
CREATE POLICY "Allow all read" ON margin_trading FOR SELECT USING (true);
