# 🚀 快速开始指南

## 5分钟启动系统

### 1. 创建 Supabase 数据库 (2分钟)

1. 访问 [supabase.com](https://supabase.com)
2. 点击 "New Project"
3. 选择地区 (建议: 新加坡/东京)
4. 复制项目 URL 和 Service Role Key

### 2. 配置环境变量 (1分钟)

```bash
cp .env.example .env
```

编辑 `.env`:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-role-key
TUSHARE_TOKEN=your-token  # 可选，用于更多数据
```

### 3. 初始化数据库 (2分钟)

**在 Supabase Dashboard → SQL Editor 按顺序执行:**

```sql
-- 1. 基础数据表 (股票、行情、财务、技术指标)
-- 复制 supabase/migrations/001_initial_schema.sql 全部内容 → Run

-- 2. 资讯数据表 (新闻、公告、研报、龙虎榜)
-- 复制 supabase/migrations/002_news_tables.sql 全部内容 → Run

-- 3. 模型验证表 (滚动验证、预测历史、性能监控)
-- 复制 supabase/migrations/003_model_validation.sql 全部内容 → Run
```

> ⚠️ **必须按顺序执行 3 个 SQL 文件**，否则会有依赖错误

### 4. 启动服务 (1分钟)

**在 GitHub Codespaces:**

```bash
# 自动安装依赖后，运行:
python start.py

# 选择菜单选项:
# 1. 启动 API 服务      → http://localhost:8000/docs
# 2. 启动可视化界面     → http://localhost:8501
```

**本地运行:**

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 API
python src/api/main.py

# 启动可视化 (新终端)
streamlit run src/api/dashboard.py
```

---

## 📊 初始化数据流程

### 首次部署（按顺序执行）

```bash
# 1. 同步股票列表 (约 5000+ 只)
python scripts/sync_data.py --type stock_list

# 2. 同步历史行情数据 (最近3年，约 15-30 分钟)
python scripts/sync_data.py --type full

# 3. 计算技术指标 (约 10-20 分钟)
python scripts/calc_indicators.py

# 4. 检查数据质量
python scripts/data_quality.py
```

### 启动自动数据同步

数据同步已配置 GitHub Actions，会自动运行：

- **每天 16:00** (收盘后): 同步日线数据
- **每天 22:00**: 补充数据、计算指标
- **每周日 22:00**: 训练模型

**启用方法:**
1. 将代码推送到 GitHub
2. 在仓库 Settings → Secrets 添加:
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `SUPABASE_SERVICE_KEY`
   - `TUSHARE_TOKEN` (可选)
3. GitHub Actions 会自动按计划执行

---

## 🎯 滚动验证与模型校准（核心功能）

### 从 2025-01-01 开始验证

```bash
# 单只股票滚动验证
python scripts/rolling_validation.py \
  --ts-code 000001.SZ \
  --start-date 2025-01-01 \
  --model-type xgboost

# 验证所有股票 (100只)
python scripts/rolling_validation.py \
  --all-stocks \
  --start-date 2025-01-01 \
  --max-stocks 100
```

### 查看验证结果

```bash
# 分析历史验证结果
python scripts/rolling_validation.py \
  --analyze \
  --results-dir models/validation/

# 查看输出文件
ls models/validation/
# - validation_metrics_*.csv    # 验证指标
# - predictions_history_*.csv   # 预测历史
# - calibration_report_*.json   # 校准报告
# - validation_results.png      # 可视化图表
```

### 自动化验证

已配置 GitHub Actions (`.github/workflows/rolling-validation.yml`):
- **每天凌晨 01:00** 自动运行滚动验证
- 自动校准模型参数
- 自动保存验证结果到数据库

**手动触发:**
1. GitHub 仓库 → Actions → "滚动验证与模型校准"
2. 点击 "Run workflow"
3. 可指定股票代码、日期范围

---

## 🔧 常用命令速查

### 数据管理
```bash
# 快速启动菜单
python start.py

# 同步当日数据
python scripts/sync_data.py --type daily

# 同步单只股票
python scripts/sync_data.py --type incremental --stock 000001.SZ

# 检查数据质量
python scripts/data_quality.py --check all
```

### 模型训练与验证
```bash
# 训练多因子模型
python scripts/train_model.py --type xgboost --stock 000001.SZ

# 滚动验证 (带校准)
python scripts/rolling_validation.py --ts-code 000001.SZ --start-date 2025-01-01

# 分析验证结果
python scripts/rolling_validation.py --analyze
```

### 数据质量
```bash
# 检查所有数据
python scripts/data_quality.py

# 检查股票列表
python scripts/data_quality.py --check stock_list

# 检查日线数据
python scripts/data_quality.py --check daily
```

### 测试
```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/unit/test_technical.py -v
```

---

## 🐳 Docker 方式 (可选)

```bash
cd docker

# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f api

# 执行数据同步
docker-compose --profile sync run --rm sync

# 停止服务
docker-compose down
```

访问:
- API: http://localhost:8000
- 可视化: http://localhost:8501
- Jupyter: http://localhost:8888

---

## 📁 项目结构

```
├── src/
│   ├── data_collection/     # 数据采集 (行情+资讯)
│   ├── analysis/            # 技术指标 + 回测 + 多因子
│   ├── prediction/          # ML模型 + 滚动验证
│   │   ├── multi_factor_model.py      # 多因子预测
│   │   └── walk_forward_validation.py # 滚动验证
│   ├── api/                 # FastAPI + Streamlit
│   └── database/            # 数据库连接
├── scripts/                 # 工具脚本
│   ├── sync_data.py         # 数据同步
│   ├── calc_indicators.py   # 计算指标
│   ├── rolling_validation.py # 滚动验证 ⭐
│   └── data_quality.py      # 数据质量检查
├── supabase/migrations/     # 数据库Schema
│   ├── 001_initial_schema.sql    # 基础表
│   ├── 002_news_tables.sql       # 资讯表
│   └── 003_model_validation.sql  # 验证表 ⭐
├── .github/workflows/       # 自动化工作流
│   ├── data-sync.yml        # 定时数据同步
│   ├── model-training.yml   # 模型训练
│   └── rolling-validation.yml # 滚动验证 ⭐
└── tests/                   # 测试用例
```

---

## 🎯 核心功能

### API 接口

| 功能 | 命令/API |
|------|----------|
| 股票查询 | GET /api/stocks |
| 日线数据 | GET /api/stocks/{code}/daily |
| 技术指标 | GET /api/stocks/{code}/indicators |
| 交易信号 | GET /api/stocks/{code}/signals |
| 策略回测 | POST /api/backtests |
| 模型预测 | POST /api/predictions |

### 多因子预测模型

**整合因子:**
- 📈 价格动量 (5/10/20日收益率)
- 📊 技术指标 (MACD、RSI、布林带)
- 📰 新闻情感 (情感均值、新闻数量)
- 💰 资金流向 (主力净流入、大单占比)
- 🐉 龙虎榜 (机构买卖、上榜次数)
- 🌐 市场情绪 (涨跌比、平均涨跌幅)

### 滚动验证流程

```
Day N:  用 [N-252, N-1] 数据训练 → 预测 Day N → 记录预测
Day N+1: 对比真实数据 → 计算准确率 → 校准概率 → 优化阈值
         ↓
Day N+1: 用 [N-251, N] 数据训练 → 预测 Day N+1 → ...
```

**持续优化，越用越准！**

---

## 🚀 部署后运行流程

### 第 1 天 (今天)
```bash
# 1. 初始化数据
python scripts/sync_data.py --type stock_list
python scripts/sync_data.py --type full
python scripts/calc_indicators.py

# 2. 启动服务
python start.py  # 选择 1 和 2
```

### 第 2 天起 (2025-01-01)
```bash
# 运行滚动验证 (用真实数据校验预测)
python scripts/rolling_validation.py \
  --ts-code 000001.SZ \
  --start-date 2025-01-01 \
  --end-date 2025-01-02

# 查看验证结果
ls models/validation/
```

### 每天自动运行 (GitHub Actions)
- 数据同步 ✅
- 滚动验证 ✅
- 模型校准 ✅
- 结果保存 ✅

---

## ❓ 常见问题

**Q: 数据从哪来?**
A: 默认使用 AKShare (免费)，可选 Tushare (需Token)

**Q: 免费额度够用吗?**
A:
- Supabase: 500MB 免费额度可存 100只股票×3年
- GitHub Actions: 2000分钟/月，足够日常同步

**Q: 如何更新数据?**
A:
- 自动: GitHub Actions 每天自动同步
- 手动: `python scripts/sync_data.py --type daily`

**Q: 滚动验证是什么?**
A: 每天用过去1年数据训练模型，预测明天涨跌，等明天收盘后用真实数据校验，不断校准模型参数

**Q: 模型多久更新一次?**
A:
- 每天自动验证并校准
- 每周日重新训练完整模型
- 可随时手动运行 `scripts/rolling_validation.py`

**Q: 支持实时数据吗?**
A: 当前支持日线/分钟线，实时推送需额外接入 WebSocket

**Q: 部署后用户怎么访问?**
A: 浏览器直接访问 Streamlit Cloud 或 Vercel 部署的地址，无需安装任何软件

---

## 📞 遇到问题?

1. 查看 [README.md](README.md) 完整文档
2. 查看 [ARCHITECTURE.md](ARCHITECTURE.md) 架构说明
3. 检查日志: `logs/` 目录
4. 提交 [GitHub Issue](https://github.com/yourusername/a-stock-quant/issues)

---

**🎉 祝你部署顺利！记得从 2025-01-01 开始跑验证！**
