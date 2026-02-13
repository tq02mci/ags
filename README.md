# 📈 A股量化交易系统

基于 GitHub Codespaces + Supabase 的 A股数据分析与预测平台。

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    GitHub Codespaces (开发环境)                   │
│                   VS Code + Python 3.11                          │
├─────────────────────────────────────────────────────────────────┤
│                          FastAPI                                  │
│                    RESTful API 服务                              │
├─────────────────────────────────────────────────────────────────┤
│                       Supabase (PostgreSQL)                       │
│  股票数据 · 技术指标 · 财务数据 · 预测结果 · 回测结果              │
├─────────────────────────────────────────────────────────────────┤
│                      GitHub Actions                               │
│  定时数据同步 · 模型训练 · 自动部署                              │
└─────────────────────────────────────────────────────────────────┘
```

## 成本优势

| 服务 | 免费额度 | 预估月费 |
|------|----------|----------|
| GitHub Codespaces | 120小时/月 | ¥0 |
| Supabase Database | 500MB-8GB | ¥0-50 |
| Supabase Storage | 1GB | ¥0-20 |
| GitHub Actions | 2000分钟/月 | ¥0-100 |
| **总计** | - | **¥0-170** |

## 快速开始

### 1. 创建 GitHub 仓库并启用 Codespaces

```bash
# 克隆仓库
git clone https://github.com/yourusername/a-stock-quant.git
cd a-stock-quant

# 或在 GitHub 上点击 "Code" → "Codespaces" → "Create codespace"
```

### 2. 配置 Supabase

1. 访问 [Supabase](https://supabase.com) 创建项目
2. 在 SQL Editor 中执行 [数据库初始化脚本](supabase/migrations/001_initial_schema.sql)
3. 复制项目 URL 和 API Key

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入 Supabase 配置
```

### 4. 启动开发环境

Codespaces 会自动安装依赖。手动启动服务：

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 API 服务
python src/api/main.py
# 服务运行在 http://localhost:8000

# 启动 Streamlit 前端 (新终端)
streamlit run src/api/dashboard.py
# 服务运行在 http://localhost:8501
```

## 项目结构

```
a-stock-quant/
├── .devcontainer/           # GitHub Codespaces 配置
├── .github/workflows/       # GitHub Actions 工作流
│   ├── data-sync.yml        # 定时数据同步
│   ├── model-training.yml   # 模型训练
│   └── deploy.yml           # 部署
├── scripts/                 # 工具脚本
│   ├── sync_data.py         # 数据同步
│   ├── calc_indicators.py   # 计算技术指标
│   └── train_model.py       # 模型训练
├── src/                     # 源代码
│   ├── config.py            # 配置管理
│   ├── data_collection/     # 数据采集
│   ├── analysis/            # 技术分析
│   ├── prediction/          # 预测模型
│   ├── api/                 # API服务
│   └── database/            # 数据库操作
├── supabase/migrations/     # 数据库迁移
├── ARCHITECTURE.md          # 架构文档
└── README.md                # 本文件
```

## 核心功能

### 📊 数据采集
- **数据源**: AKShare(免费), Tushare(付费), Baostock(免费)
- **数据类型**: 日线行情、分钟线、财务报表、复权因子
- **自动同步**: GitHub Actions 定时同步

### 📈 技术分析
- **趋势指标**: MA, EMA, MACD
- **震荡指标**: RSI, KDJ, CCI, WR
- **波动指标**: BOLL, ATR
- **量能指标**: OBV, VOL_MA

### 🤖 预测模型
- **LSTM**: 长短期记忆网络，适合时序预测
- **XGBoost**: 梯度提升树，适合特征工程
- **LightGBM**: 快速训练，适合大规模数据
- **集成模型**: 多模型投票/加权

### 📋 回测系统
- 支持多种策略回测
- 计算收益率、夏普比率、最大回撤等指标
- 可视化收益曲线和交易记录

## API 文档

启动服务后访问: `http://localhost:8000/docs`

### 主要接口

| 接口 | 描述 |
|------|------|
| `GET /api/stocks` | 获取股票列表 |
| `GET /api/stocks/{code}` | 股票详情 |
| `GET /api/stocks/{code}/daily` | 日线数据 |
| `GET /api/stocks/{code}/indicators` | 技术指标 |
| `GET /api/stocks/{code}/signals` | 交易信号 |
| `POST /api/predictions` | 创建预测 |
| `POST /api/backtests` | 创建回测 |

## 工作流说明

### 数据同步工作流 (data-sync.yml)

**触发条件:**
- 定时: 每天 16:00 (收盘后), 22:00 (补充数据)
- 手动: 支持 workflow_dispatch

**执行步骤:**
1. 同步股票列表
2. 同步日线数据
3. 计算技术指标
4. 通知结果

### 模型训练工作流 (model-training.yml)

**触发条件:**
- 定时: 每周日晚上
- 手动: 可指定模型类型和股票

**执行步骤:**
1. 训练模型
2. 上传模型文件
3. 生成预测

## 部署

### 前端部署 (Vercel)

```bash
# 安装 Vercel CLI
npm i -g vercel

# 部署
vercel --prod
```

### 数据库

Supabase 自动托管，无需额外部署。

## 环境变量

| 变量 | 说明 | 必需 |
|------|------|------|
| `SUPABASE_URL` | Supabase 项目 URL | ✅ |
| `SUPABASE_KEY` | Supabase anon key | ✅ |
| `SUPABASE_SERVICE_KEY` | Supabase service role key | ✅ |
| `TUSHARE_TOKEN` | Tushare Pro Token | ❌ |
| `LOG_LEVEL` | 日志级别 | ❌ |

## 开发指南

### 添加新的数据源

在 `src/data_collection/collectors.py` 中创建新的采集器类，继承 `DataCollector`。

### 添加新的技术指标

在 `src/analysis/technical.py` 的 `TechnicalAnalyzer` 类中添加计算方法。

### 添加新的预测模型

在 `src/prediction/models.py` 中继承 `BaseModel` 实现新的模型。

## 注意事项

1. **数据合规**: 仅用于个人研究，勿用于商业用途
2. **风险控制**: 预测结果仅供参考，不构成投资建议
3. **成本控制**: 注意 GitHub Actions 使用时间和 Supabase 存储容量

## 技术栈

- **后端**: FastAPI, SQLAlchemy, Pandas
- **数据库**: PostgreSQL (Supabase)
- **前端**: Streamlit, Plotly
- **ML**: PyTorch, XGBoost, scikit-learn
- **部署**: GitHub Actions, Vercel

## 许可证

MIT License - 详见 LICENSE 文件

## 贡献

欢迎提交 Issue 和 Pull Request!

---

⚠️ **风险提示**: 股市有风险，投资需谨慎。本系统仅供学习研究使用。
