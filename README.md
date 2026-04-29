# 📊 金融研报智能问答助手 (Financial RAG Assistant)

## 💡 项目简介
本项目是一个针对金融投研场景设计的私有化 RAG（检索增强生成）问答系统。
旨在解决传统大模型在处理数十页超长金融研报时存在的“数据遗漏”、“逻辑幻觉”和“页码溯源困难”等痛点。

## 🛠️ 技术栈
* **核心框架**: LlamaIndex, Python
* **大语言模型**: DeepSeek-V3 (API)
* **向量模型**: BAAI/bge-small-zh-v1.5 (本地私有化部署)
* **前端展示**: Streamlit

## 🚀 核心亮点
1. **子问题查询引擎 (Sub-Question Engine)**：利用大模型作为 Planner，将复杂的“多公司横向对比”指令动态拆解为并行子任务，彻底解决多文档召回挤压问题，对比数据 100% 召回。
2. **数据隐私保护**：采用 BAAI 本地微型 Embedding 模型，确保绝密研报资料不上传第三方云端，仅在本地进行向量计算。
3. **强制防幻觉溯源**：通过 Prompt 注入底层约束，实现 100% 回答附带【研报来源名 + 原文摘录】。

## 💻 快速运行
1. 克隆本项目，并在 `data` 文件夹下放入您的 PDF 研报。
2. 安装依赖：`pip install -r requirements.txt`
3. 配置环境：将 `.env.example` 改名为 `.env` 并填入您的 API Key。
4. 启动网页端：`streamlit run app.py`
