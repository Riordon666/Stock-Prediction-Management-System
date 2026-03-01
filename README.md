![Python](https://img.shields.io/badge/Python-3.10.11-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1--GPU-orange?logo=tensorflow&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.2-76b900?logo=nvidia&logoColor=white)

# Stock Prediction Management System (专业增强版)

基于 **Flask + TensorFlow** 的全栈股票分析与预测系统。本系统已升级为 **22维多特征预测架构**，集成了深度学习（GRU/LSTM）、技术指标计算与实时新闻情感分析。

## 🌟 核心升级特性 (v1.1.0)

- **双模型支持**: 提供 **GRU** 与 **LSTM** 两种深度学习模型，支持在前端动态切换。
- **22维特征引擎**: 预测维度从单一价格升级为 22 维增强向量，包含：
  - **基础行情**: 开/高/低/收、成交量、成交额。
  - **15项技术指标**: MA(5/10/20)、EMA(12/26)、MACD(Diff/Dea/Hist)、RSI(14)、KDJ(K/D/J)、成交量均线。
  - **新闻情感分析**: 集成 HuggingFace Transformer 模型，对财联社快讯进行实时情感量化。
- **GPU 硬件加速**: 深度适配 Windows CUDA 11.2，支持 NVIDIA 显卡加速训练与推理。
- **训练保护机制**: 训练脚本支持 `SIGINT/SIGTERM` 信号捕捉，通过 Ctrl+C 或 PyCharm 停止时自动保存权重与断点。

## 🛠️ 功能模块

| 模块 | 入口 | 技术要点 |
| --- | --- | --- |
| **实时快讯** | `/` | 财联社异步抓取、关键词权重过滤、无缝滚动通知栏、今日热榜缓存。 |
| **多维分析** | `/analysis` | ApexCharts 交互式 K 线图、MACD/RSI 联动、自动支撑压力位算法、AI 自动生成的分析报告副本。 |
| **AI 深度预测** | `/predict` | **22维特征输入**、GRU/LSTM 模型切换、自回归滑动预测（未来10/20/30天）、多指标对齐算法。 |

## 📂 目录结构 (精简)

```text
Stock Prediction Management System/
├─ app/
│  ├─ web/
│  │  ├─ web_server.py          # 核心 API 与 Web 服务
│  │  └─ templates/             # 现代化 UI 模板
│  ├─ core/
│  │  ├─ data_provider.py       # 万能数据适配层
│  │  └─ news_fetcher.py        # 快讯异步采集
├─ forecasting/
│  ├─ core/                     # 特征工程 (FeatureEngine)
│  ├─ utils/                    # 技术指标 (Indicators) 与 情感分析 (Sentiment)
│  ├─ models/                   # 存储 GRU/LSTM 权重与训练断点 (state.json)
│  ├─ train_gru.py              # GRU 22维多特征训练脚本
│  └─ train_lstm.py             # LSTM 22维多特征训练脚本
├─ .env                         # 环境配置 (端口、日志等)
└─ run.py                       # 一键启动入口
```

## 🚀 快速开始

### 1. 环境准备
推荐使用 Python 3.10。确保已安装 CUDA 11.2 和 cuDNN 8.1（若需使用 GPU 加速）。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 模型训练 (必须执行)
由于系统已升级至 22 维输入，需先重训模型生成最新的权重文件：
```powershell
python -m forecasting.train_gru
python -m forecasting.train_lstm
```
*注：训练过程中可随时中止，系统会自动保存已完成的进度。*

### 3. 运行系统
```powershell
python run.py
```
访问：`http://localhost:8888`

## 📊 预测 API 说明

- **接口**: `GET /api/predict_stock`
- **参数**:
  - `stock_code`: 6位代码（如 `600519`）
  - `market_type`: `A` / `HK`
  - `predict_days`: `10` / `20` / `30`
  - `model_type`: `gru` / `lstm`
- **特征工程**: 自动从 `ohlcv` 数据计算 15 个指标并融合当前新闻情感得分进行预测。

## ⚠️ 免责声明
本系统仅供学术研究使用。股市有风险，AI 预测仅作为多维度参考之一，不构成任何投资建议。
