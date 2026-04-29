import streamlit as st
import os
import re
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="金融研报 AI 助手", layout="wide", page_icon="📊")

# --- 2. 初始化引擎 (使用缓存确保只加载一次) ---
@st.cache_resource
def init_engine():
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    # 配置模型
    Settings.llm = DeepSeek(model="deepseek-chat", api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
    
    # 构建多文档工具集
    data_dir = "./data"
    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    tools = []
    
    for i, file in enumerate(files):
        documents = SimpleDirectoryReader(input_files=[os.path.join(data_dir, file)]).load_data()
        index = VectorStoreIndex.from_documents(documents)
        engine = index.as_query_engine(similarity_top_k=5)
        
        tool_id = f"report_{i}"
        tools.append(QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(
                name=tool_id,
                description=f"关于 {file.replace('.pdf','')} 的研报内容"
            )
        ))
    
    # 创建子问题引擎
    return SubQuestionQueryEngine.from_defaults(query_engine_tools=tools, use_async=True)

# --- 3. 侧边栏设计 ---
with st.sidebar:
    st.title("⚙️ 系统配置")
    if st.button("🔄 刷新知识库"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    st.subheader("📂 已加载研报")
    pdf_files = [f for f in os.listdir("./data") if f.endswith(".pdf")]
    for f in pdf_files:
        st.caption(f"📄 {f}")
    
    st.markdown("---")
    st.info("本助手采用 LlamaIndex 子问题拆解技术，支持跨文档精准对比。")

# --- 4. 主界面设计 ---
st.title("📊 金融研报智能问答助手")
st.caption("基于 DeepSeek-V3 & LlamaIndex 构建的专业投研工具")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. 问答逻辑 ---
if prompt := st.chat_input("例如：请对比这三家公司2025年的营收和风险点..."):
    # 显示用户提问
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 AI 引擎
    with st.chat_message("assistant"):
        with st.spinner("🚀 正在拆解意图并检索多份研报..."):
            try:
                engine = init_engine()
                # 强制要求中文和表格
                final_query = f"{prompt} (请务必使用中文回答，数据请尽量使用表格呈现)"
                response = engine.query(final_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": str(response)})
            except Exception as e:
                st.error(f"发生错误：{e}")