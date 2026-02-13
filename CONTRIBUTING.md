# 贡献指南

感谢您对 A股量化交易系统 的兴趣！我们欢迎各种形式的贡献。

## 开发环境搭建

### 使用 GitHub Codespaces (推荐)

1. 点击仓库首页的 "Code" → "Codespaces" → "Create codespace"
2. 等待环境自动配置完成
3. 开始开发！

### 本地开发

```bash
# 克隆仓库
git clone https://github.com/yourusername/a-stock-quant.git
cd a-stock-quant

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件
```

## 项目结构

```
src/
├── data_collection/     # 数据采集模块
├── analysis/           # 分析模块
│   ├── technical.py    # 技术指标
│   ├── backtest.py     # 回测系统
│   └── factors.py      # 多因子模型
├── prediction/         # 预测模型
│   └── models.py       # LSTM/XGBoost等
├── api/                # API服务
│   ├── main.py         # FastAPI
│   └── dashboard.py    # Streamlit
└── database/           # 数据库操作
```

## 开发流程

1. **Fork 仓库** 并创建您的分支
   ```bash
   git checkout -b feature/my-feature
   ```

2. **编写代码**
   - 遵循 PEP 8 规范
   - 添加必要的注释
   - 编写单元测试

3. **运行测试**
   ```bash
   pytest tests/ -v
   ```

4. **提交代码**
   ```bash
   git add .
   git commit -m "feat: 添加新功能"
   git push origin feature/my-feature
   ```

5. **创建 Pull Request**

## 代码规范

### Python 代码风格

- 使用 `black` 格式化代码
- 使用 `isort` 排序导入
- 使用类型提示

```bash
# 格式化代码
black src/ tests/
isort src/ tests/

# 类型检查
mypy src/
```

### 提交信息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/):

- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具

示例:
```
feat: 添加MACD策略回测功能
fix: 修复RSI计算错误
docs: 更新API文档
```

## 添加新功能

### 添加新的技术指标

在 `src/analysis/technical.py` 中添加:

```python
@staticmethod
def calculate_new_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """计算新指标"""
    df['new_indicator'] = ...  # 你的计算逻辑
    return df
```

然后在 `calculate_all` 方法中调用。

### 添加新的预测模型

在 `src/prediction/models.py` 中继承 `BaseModel`:

```python
class MyModel(BaseModel):
    def __init__(self, name: str = "MyModel", params: Optional[Dict] = None):
        super().__init__(name, params)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # 特征工程
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # 训练逻辑
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 预测逻辑
        pass
```

### 添加新的数据源

在 `src/data_collection/collectors.py` 中继承 `DataCollector`:

```python
class MyDataSource(DataCollector):
    def get_stock_list(self) -> pd.DataFrame:
        # 获取股票列表
        pass

    def get_daily_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        # 获取日线数据
        pass
```

## 测试

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定模块
pytest tests/unit/test_technical.py

# 运行并生成覆盖率报告
pytest --cov=src --cov-report=html
```

### 编写测试

```python
def test_my_feature():
    """测试新功能"""
    # 准备数据
    df = pd.DataFrame(...)

    # 执行
    result = my_function(df)

    # 验证
    assert result == expected
```

## 文档

- 更新 `README.md` 说明新功能
- 在代码中添加 docstring
- 更新 API 文档 (自动生成)

## 问题反馈

如果您发现了 bug 或有新功能建议:

1. 先搜索 [Issues](https://github.com/yourusername/a-stock-quant/issues) 看是否已存在
2. 如果没有，创建新的 Issue
3. 提供详细描述、复现步骤、期望行为

## 安全提醒

- 不要将 API Token 提交到代码仓库
- 敏感信息使用 GitHub Secrets
- 定期轮换 API Token

## 许可证

通过提交代码，您同意将代码以 MIT 许可证授权。

## 联系方式

- 问题反馈: [GitHub Issues](https://github.com/yourusername/a-stock-quant/issues)
- 讨论交流: [GitHub Discussions](https://github.com/yourusername/a-stock-quant/discussions)

感谢贡献！🎉
