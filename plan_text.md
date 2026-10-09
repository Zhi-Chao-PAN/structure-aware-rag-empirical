LangGraph Financial Swarm
落地应用改进完善计划 v2.0
Multi-Agent Financial Analysis System — From Research to Production
潘志超 | 2026年2月
当前状态：v0.9.0 | 5家公司 | 90题Benchmark 100%准确率 | 273项测试 |
本地Ollama推理
目标：从固定数据集的研究原型 → 任意上市公司实时可查的落地产品

改进计划总览
本计划分为 6 大模块，共 18 项改进，预计 2-3
周完成。每项改进都有明确的交付物和验收标准。
模块
改进项数
核心价值
优先级
Phase 1: 实时数据引擎
3
从固定数据→全球任意公司
P0 最高
Phase 2: 智能分析增强
4
深度分析能力，超越简单问答
P0
Phase 3: 用户体验革新
3
专业级交互界面
P1
Phase 4: 可靠性与性能
3
生产级稳定性
P1
Phase 5: 部署与分发
3
一键部署，开箱即用
P2
Phase 6: 创新亮点
2
差异化竞争力
P2
Phase 1: 实时数据引擎（核心突破）
这是整个改进的基石。将系统从「固定5家公司」升级为「全球任意上市公司实时可查」。
1.1 Yahoo Finance 实时数据采集模块
• 技术方案：基于 yfinance 库，自动拉取任意 ticker
的财务报表（损益表、资产负债表、现金流量表）
• 覆盖范围：美股、港股、A股、日股、欧股等全球主要市场
• 数据格式：自动转换为结构化 Markdown（与现有 RAG 管道兼容）
• 缓存策略：本地 SQLite 缓存 + TTL 过期机制，避免重复请求
• 交付物：src/data/finance_fetcher.py，CLI 命令 /fetch AAPL NVDA
• 创新点：动态 RAG — 用户提问时自动检测公司名，按需拉取数据并实时构建索引
1.2 动态索引热更新
• 技术方案：新数据拉取后，增量更新向量索引（不重建全量），BM25 索引同步刷新
• 性能目标：新增一家公司数据 < 3秒完成索引更新
• 交付物：RAGAdapter.add_documents() 增量接口
1.3 多数据源聚合
• 技术方案：yfinance（财报）+ 本地 PDF/MD（用户自有文档）+
未来可扩展（新闻、研报）
• 数据源注册机制：插件式架构，新增数据源只需实现 DataSource 接口
• 交付物：src/data/sources/ 目录，DataSource 抽象基类

Phase 2: 智能分析增强（深度能力）
从「查数据」升级为「做分析」，让系统具备真正的金融分析能力。
2.1 财务比率自动计算引擎
• 功能：自动计算 ROE、ROA、流动比率、速动比率、负债率等 20+ 财务指标
• 技术：Quant Agent 增强，内置财务公式库，支持跨年对比
• 亮点：用户问「NVIDIA 的盈利能力如何」→ 自动计算并对比行业平均
2.2 同行业对比分析（Peer Comparison）
• 功能：自动识别同行业公司，生成对比报告
• 技术：GICS 行业分类 + 自动拉取同行数据 + 雷达图可视化
• 示例：「分析 NVIDIA 在半导体行业的竞争地位」→ 自动对比
AMD、Intel、Qualcomm
• 创新点：Multi-Agent 协作 — Researcher 拉数据，Quant 算指标，Supervisor
综合判断
2.3 趋势预测与预警
• 功能：基于历史数据的简单趋势外推 + 异常检测
• 技术：线性回归/移动平均 + Z-score 异常检测（不依赖深度学习，轻量可靠）
• 示例：「NVIDIA 毛利率是否在下降？」→ 趋势分析 + 置信区间 + 预警
2.4 自然语言财报摘要生成
• 功能：一键生成公司财务健康报告（中英文）
• 格式：结构化报告（概览→收入分析→盈利能力→风险提示→投资建议）
• 交付物：/report AAPL 命令，输出 PDF/Markdown 报告

Phase 3: 用户体验革新（专业界面）
3.1 Streamlit Pro Dashboard
• 功能：专业级金融仪表盘，多 tab 布局
• 页面：实时查询 | 公司对比 | 行业分析 | 数据浏览器 | 报告生成
• 可视化：ECharts/Plotly 交互式图表（K线图、财务趋势、雷达图、热力图）
• 响应式设计：桌面端和移动端自适应
3.2 智能搜索建议
• 功能：输入时自动补全公司名、推荐相关问题
• 技术：Trie 前缀树 + 热门查询统计 + 上下文感知推荐
• 示例：输入「NV」→ 推荐「NVIDIA revenue」「NVIDIA vs AMD」「NVIDIA gross
margin trend」
3.3 多语言支持
• 功能：中英文双语查询和回答
• 技术：语言检测 + prompt 模板切换 + 中文财务术语映射
• 价值：真正「惠及老百姓」— 中国用户可以用中文查询全球公司财报
• 创新点：跨语言金融 RAG — 中文提问，英文数据源，中文回答
Phase 4: 可靠性与性能（生产级）
4.1 多 LLM 后端支持
• 功能：支持 Ollama / OpenAI API / DeepSeek API / 通义千问 等多种后端
• 价值：用户可根据硬件条件选择最优方案（本地 GPU / 云端 API）
• 自动降级：本地模型不可用时自动切换到云端 API
4.2 查询结果可信度评估
• 功能：每个回答附带可信度评分 + 数据来源追溯
• 技术：Source Attribution（标注每个数字来自哪个文档/表格）
• 价值：金融数据必须可追溯，这是专业系统的基本要求
4.3 异步流式响应
• 功能：查询结果流式输出（打字机效果），而非等待完整响应
• 技术：SSE (Server-Sent Events) + Streamlit streaming
• 体验：用户感知延迟从 12s → 2s（首 token 时间）

Phase 5: 部署与分发（开箱即用）
5.1 一键安装脚本
• pip install langgraph-financial-swarm 或 Docker 一键启动
• 自动检测 GPU，选择最优模型配置
• 首次运行自动下载模型 + 拉取示例数据
5.2 云端部署方案
• Streamlit Cloud 免费部署（演示用）
• Docker Compose 生产部署（Ollama + App + Nginx）
• 部署文档 + 一键脚本
5.3 API Server
• FastAPI REST API，支持第三方集成
• 端点：/query, /report, /compare, /fetch
• OpenAPI 文档自动生成
Phase 6: 创新亮点（差异化）
6.1 Agent 推理过程可视化
• 功能：实时展示 Multi-Agent 协作过程（哪个 Agent 在工作、查了什么、推理了什么）
• 技术：LangGraph 状态图 + D3.js 动态可视化
• 价值：让用户理解 AI 的推理过程，增强信任感
• 创新点：这是目前市面上几乎没有的功能 — 大多数 RAG 系统是黑盒
6.2 个人投资组合追踪
• 功能：用户输入持仓，系统自动追踪并生成分析报告
• 示例：「我持有 NVDA 100股、AAPL 50股，帮我分析组合风险」
• 技术：Portfolio 数据模型 + 风险指标计算（Beta、Sharpe Ratio、最大回撤）

执行时间线
阶段
时间
关键交付物
测试目标
Phase 1
第 1 周
yfinance 采集 + 动态索引 + 数据源架构
300+ tests
Phase 2
第 1-2 周
财务计算引擎 + 同行对比 + 趋势预测 + 报告生成350+ tests
Phase 3
第 2 周
Pro Dashboard + 搜索建议 + 中英双语
380+ tests
Phase 4
第 2-3 周
多 LLM 后端 + 可信度评估 + 流式响应
400+ tests
Phase 5
第 3 周
pip 包 + Docker 部署 + FastAPI
420+ tests
Phase 6
第 3 周
Agent 可视化 + 投资组合
450+ tests
核心原则
1. 每个改进都必须有测试覆盖，不允许裸提交
2. 每个 Phase 完成后做一次集成测试 + 性能回归测试
3. 用户体验优先 — 功能再强，不好用等于没用
4. 渐进式交付 — 每天都有可演示的进展
预期成果
完成全部 6 个 Phase 后，LangGraph Financial Swarm 将成为：
• 全球任意上市公司实时财务分析系统
• 支持中英文双语的智能金融助手
• 具备专业级 UI 和可视化的落地产品
• 450+ 测试覆盖的生产级代码质量
• 一键部署、开箱即用的开源项目
这不仅是一个课程项目，而是一个真正可用的金融分析工具。

