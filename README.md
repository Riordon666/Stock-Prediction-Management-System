![Python](https://img.shields.io/badge/Python-3.10.11-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)


# Stock Prediction Management System
（股票预测管理系统）


基于 **Flask** 的股票分析/预测管理系统，提供股票历史行情获取、技术指标可视化、综合评分与分析报告展示等功能。

## 功能特性

| 模块       | 路由/入口 | 功能说明 | 状态 |
|----------| --- | --- | --- |
| 首页(实时快讯) | `/` | 实时快讯时间线（默认近2天，最多500条）<br>只看重要（关键词过滤）<br>手动刷新（触发后端抓取并刷新列表）<br>自动刷新（5分钟一次）<br>底部滚动快讯（取最新3条，无缝循环）<br>今日热榜（TopHub，失败回退并缓存） | 已实现 |
| 股票分析     | `/analysis` | 价格趋势（收盘价折线 + MA5/MA20/MA60）<br>技术指标（RSI / MACD：含 Signal、Histogram）<br>成交量（柱状图 + 均量线 MA20）<br>支撑/压力位展示<br>雷达图多维度评分<br>AI 分析报告展示 | 已实现 |
| 股票预测     | `/predict` | 正在开发中 | 开发中 |


## 技术栈

- **后端**：Flask
- **数据处理**：pandas、numpy
- **前端图表**：ApexCharts（通过 CDN 引入）
- **数据源适配器**：
  - akshare（优先）
  - baostock（可选，作为备用数据源）

## 目录结构

```
Stock Prediction Management System/
├─ app/
│  ├─ web/
│  │  ├─ web_server.py          # Flask Web 服务与 API
│  │  ├─ templates/             # HTML 模板（index/analysis/predict）
│  │  └─ static/                # 静态资源（css、js、图片等）
│  ├─ core/
│  │  ├─ database.py            # 数据库管理
│  │  ├─ data_provider.py       # 统一数据提供层（多数据源故障转移）
│  │  └─ fallback_manager.py    # 故障转移管理器
│  └─ adapters/
│     ├─ akshare_adapter.py     # akshare 适配器
│     ├─ baostock_adapter.py    # baostock 适配器
│     └─ base_adapter.py        # 适配器基类
├─ data/
│  └─ logs/                     # 服务日志（默认输出到这里）
└─ run.py                       # 入口：启动 Flask 服务
```

## 安装与运行


### 1. 创建并激活虚拟环境（可选但推荐）

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 安装依赖

```powershell
pip install flask python-dotenv pandas numpy akshare
```

可选（备用数据源）：

```powershell
pip install baostock
```

### 3. 启动服务

```powershell
python run.py
```

默认访问：

- http://127.0.0.1:8888

## 配置说明（环境变量）

项目使用 `python-dotenv` 读取环境变量（可自行创建 `.env` 文件）：

- `PORT`：服务端口（默认 8888）
- `LOG_LEVEL`：日志级别（默认 INFO）
- `LOG_FILE`：日志文件路径（默认 `data/logs/server.log`）
- `TOPHUBDATA_ACCESS_KEY`：可选。用于访问 TopHubData 官方 API 获取“今日热榜”（更稳定）。未配置时会自动回退到 `tophub.today` 抓取解析。

示例（.env）：

```env
PORT=8888
LOG_LEVEL=INFO
LOG_FILE=data/logs/server.log
TOPHUBDATA_ACCESS_KEY=your_key_here
```

## 首页实时快讯相关 API

- `GET /api/latest_news`
  - 参数：`days`（1-7）、`limit`（1-500）、`important`（0/1）
  - 说明：返回实时快讯列表；若本地无数据会尝试触发一次抓取。
- `POST /api/fetch_news`
  - 说明：触发后台抓取财联社电报数据（异步执行）。
- `GET /api/hotspots`
  - 参数：`limit`（1-30）
  - 说明：返回“今日热榜”；会缓存最近一次非空结果，抓取/解析失败时回退到上次非空数据，避免频繁出现“暂无热点”。

## 数据源说明

- **akshare**：默认优先使用。
- **baostock**：可选备用。

当未安装 `baostock` 时，日志会提示 `BaostockAdapter`不可用，这是正常的（只要 akshare 可用即可完成数据获取）。

## 常见问题

### 1) 提示“未检测到可用数据源适配器”

- 需要至少安装一个数据源：
  - `pip install akshare`（推荐）
  - 或 `pip install baostock`

### 2) 技术指标/图表显示异常

- 前端图表使用 ApexCharts（CDN 引入），请确认网络可访问 CDN。
- 建议浏览器强制刷新（Ctrl+F5）避免缓存影响。

## 许可证

MIT License

## 免责声明
本系统为个人设计与研究项目，仅用于学习与学术研究用途，不构成任何投资建议。股票数据来源于公开市场，预测结果可能存在误差，仅供参考。投资有风险，入市需谨慎！
