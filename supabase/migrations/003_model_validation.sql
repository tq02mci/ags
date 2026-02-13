-- 模型验证和校准相关表

-- ============================================
-- 1. 模型验证结果表
-- ============================================

CREATE TABLE model_validation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_type VARCHAR(50) NOT NULL,
    ts_code VARCHAR(20),
    validation_date DATE NOT NULL,
    end_date DATE,

    -- 性能指标
    accuracy DECIMAL(6,4),
    precision DECIMAL(6,4),
    recall DECIMAL(6,4),
    f1_score DECIMAL(6,4),
    auc DECIMAL(6,4),

    -- 混淆矩阵
    true_positive INT DEFAULT 0,
    false_positive INT DEFAULT 0,
    true_negative INT DEFAULT 0,
    false_negative INT DEFAULT 0,

    -- 验证配置
    train_days INT,
    test_days INT,
    step_days INT,

    -- 校准信息
    optimal_threshold DECIMAL(4,3) DEFAULT 0.5,
    calibration_error DECIMAL(6,4),
    calibration_report JSONB,

    -- 特征重要性
    feature_importance JSONB,

    -- 元数据
    run_id VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_validation_model ON model_validation_results(model_type, validation_date DESC);
CREATE INDEX idx_validation_code ON model_validation_results(ts_code, validation_date DESC);

COMMENT ON TABLE model_validation_results IS '模型验证结果记录';

-- ============================================
-- 2. 滚动预测历史表
-- ============================================

CREATE TABLE rolling_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts_code VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,

    -- 预测信息
    predict_date DATE NOT NULL,           -- 预测生成日期
    model_type VARCHAR(50),
    model_version VARCHAR(20),

    -- 预测结果
    predicted_direction INT,              -- 1=涨, 0=跌
    probability DECIMAL(6,4),             -- 上涨概率
    probability_calibrated DECIMAL(6,4),  -- 校准后概率
    confidence DECIMAL(6,4),              -- 置信度

    -- 实际结果（回填）
    actual_direction INT,
    actual_close DECIMAL(12,4),
    actual_return DECIMAL(8,4),

    -- 验证结果
    is_correct BOOLEAN,
    error_type VARCHAR(20),               -- FP, FN, TP, TN

    -- 使用的特征
    features_used JSONB,

    -- 交易信号
    signal_strength VARCHAR(10),          -- strong, moderate, weak
    suggested_action VARCHAR(10),         -- buy, sell, hold

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ts_code, trade_date, predict_date)
);

CREATE INDEX idx_predictions_code ON rolling_predictions(ts_code, trade_date DESC);
CREATE INDEX idx_predictions_date ON rolling_predictions(predict_date DESC);
CREATE INDEX idx_predictions_correct ON rolling_predictions(is_correct);

COMMENT ON TABLE rolling_predictions IS '滚动预测历史记录';

-- ============================================
-- 3. 模型性能监控表
-- ============================================

CREATE TABLE model_performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_date DATE NOT NULL,
    model_type VARCHAR(50) NOT NULL,

    -- 滚动窗口性能 (最近N天)
    window_days INT DEFAULT 30,

    -- 准确率相关
    accuracy DECIMAL(6,4),
    accuracy_trend VARCHAR(20),           -- improving, stable, declining

    -- 收益相关
    cumulative_return DECIMAL(10,4),      -- 累计收益
    annualized_return DECIMAL(10,4),      -- 年化收益
    max_drawdown DECIMAL(8,4),            -- 最大回撤
    sharpe_ratio DECIMAL(8,4),

    -- 预测分布
    prediction_distribution JSONB,        -- 预测分布统计
    calibration_score DECIMAL(6,4),       -- 校准度

    -- 异常检测
    is_anomaly BOOLEAN DEFAULT FALSE,
    anomaly_reason TEXT,

    -- 建议
    recommendation VARCHAR(50),           -- 模型建议: retrain, adjust_threshold, etc.

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_performance_date ON model_performance_metrics(metric_date DESC);
CREATE INDEX idx_performance_model ON model_performance_metrics(model_type, metric_date DESC);

COMMENT ON TABLE model_performance_metrics IS '模型性能监控指标';

-- ============================================
-- 4. 模型参数优化历史表
-- ============================================

CREATE TABLE model_optimization_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_type VARCHAR(50) NOT NULL,
    ts_code VARCHAR(20),

    -- 优化前后参数
    params_before JSONB,
    params_after JSONB,

    -- 优化效果
    improvement_accuracy DECIMAL(6,4),
    improvement_f1 DECIMAL(6,4),

    -- 优化方法
    optimization_method VARCHAR(50),      -- grid_search, bayesian, walk_forward

    -- 验证结果
    validation_results JSONB,

    optimized_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_optimization_model ON model_optimization_history(model_type, optimized_at DESC);

COMMENT ON TABLE model_optimization_history IS '模型参数优化历史';

-- ============================================
-- 5. 创建视图
-- ============================================

-- 模型最新性能视图
CREATE VIEW v_model_latest_performance AS
SELECT DISTINCT ON (model_type, ts_code)
    model_type,
    ts_code,
    validation_date,
    accuracy,
    f1_score,
    auc,
    optimal_threshold,
    (true_positive + true_negative)::FLOAT /
        NULLIF(true_positive + true_negative + false_positive + false_negative, 0) as win_rate
FROM model_validation_results
ORDER BY model_type, ts_code, validation_date DESC;

-- 预测准确率趋势视图
CREATE VIEW v_prediction_accuracy_trend AS
SELECT
    ts_code,
    DATE_TRUNC('week', predict_date) as week,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_predictions,
    AVG(CASE WHEN is_correct THEN 1 ELSE 0 END) as accuracy,
    AVG(probability) as avg_probability,
    AVG(probability_calibrated) as avg_probability_calibrated
FROM rolling_predictions
WHERE predict_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY ts_code, DATE_TRUNC('week', predict_date)
ORDER BY ts_code, week DESC;

-- ============================================
-- 6. 创建触发器
-- ============================================

-- 自动更新 rolling_predictions 的更新时间
CREATE OR REPLACE FUNCTION update_rolling_predictions_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_rolling_predictions_updated_at
    BEFORE UPDATE ON rolling_predictions
    FOR EACH ROW EXECUTE FUNCTION update_rolling_predictions_timestamp();

-- ============================================
-- 7. 启用 RLS
-- ============================================

ALTER TABLE model_validation_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE rolling_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_performance_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_optimization_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all read" ON model_validation_results FOR SELECT USING (true);
CREATE POLICY "Allow all read" ON rolling_predictions FOR SELECT USING (true);
CREATE POLICY "Allow all read" ON model_performance_metrics FOR SELECT USING (true);
CREATE POLICY "Allow all read" ON model_optimization_history FOR SELECT USING (true);

-- ============================================
-- 8. 插入示例数据
-- ============================================

-- 示例：插入一条验证结果
INSERT INTO model_validation_results (
    model_type, ts_code, validation_date, end_date,
    accuracy, precision, recall, f1_score, auc,
    true_positive, false_positive, true_negative, false_negative,
    train_days, test_days, step_days,
    optimal_threshold, calibration_error
) VALUES (
    'xgboost', '000001.SZ', '2025-01-15', '2025-02-01',
    0.6234, 0.6123, 0.5987, 0.6054, 0.6456,
    45, 28, 52, 30,
    252, 1, 1,
    0.52, 0.0234
);
