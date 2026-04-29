import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

# ==========================================
# 第一步：基础环境配置
# ==========================================
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    print("❌ 错误：未在 .env 中找到 DEEPSEEK_API_KEY")
    exit()

# 配置大模型大脑 (DeepSeek)
Settings.llm = DeepSeek(model="deepseek-chat", api_key=api_key)

# 配置本地向量模型 (节省 Token，处理速度快)
print("👉 正在加载本地 Embedding 模型 (BAAI/bge-small-zh-v1.5)...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# ==========================================
# 第二步：构建“分治”工具集
# ==========================================
def create_query_tools(data_dir="./data"):
    # 获取 data 文件夹下所有 PDF
    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not files:
        print(f"❌ 错误：在 {data_dir} 文件夹下没找到 PDF 文件")
        return []

    tools = []
    print(f"👉 正在为 {len(files)} 份研报建立专属索引...")

    for i, file in enumerate(files):
        # 1. 加载单份文档
        file_path = os.path.join(data_dir, file)
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        
        # 2. 建立独立索引与查询引擎
        index = VectorStoreIndex.from_documents(documents)
        engine = index.as_query_engine(similarity_top_k=5)
        
        # 3. 设置唯一的工具 ID 和详细描述 (这是解决漏检的关键)
        tool_id = f"report_{i}" # 比如 report_0, report_1
        company_name = file.replace(".pdf", "")
        
        # 在描述里明确告知大模型这个工具对应的公司主体
        tool_description = f"这是一份关于【{company_name}】的深度研究报告。如果问题涉及该公司的财务、业务或风险，必须调用此工具。"
        
        tool = QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(
                name=tool_id,
                description=tool_description
            )
        )
        tools.append(tool)
        print(f"✅ 已完成索引: {file} (分配ID: {tool_id})")
    
    return tools

# ==========================================
# 第三步：运行子问题查询引擎
# ==========================================
def run_pro_rag():
    # 1. 准备工具集
    query_engine_tools = create_query_tools()
    if not query_engine_tools:
        return

    # 2. 初始化子问题查询引擎
    # 它会自动根据你的问题生成 N 个子问题，分头去问上面的工具
    s_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        use_async=True  # 开启并行检索
    )

    print("\n" + "="*50)
    print("🚀 专家级金融研报多文档对比系统已就绪！")
    print("系统将自动拆解任务，确保每一家公司的数据都能被精准提取。")
    print("="*50)

    while True:
        user_q = input("\n👤 请输入您的提问 (输入 'quit' 退出): ")
        if user_q.lower() in ['quit', 'exit', '退出']:
            print("👋 再见！")
            break

        print("\n🤖 AI 正在智能拆解意图并进行多文档检索...")
        
        try:
            # 在 Query 结尾增加强制要求，确保输出质量
            final_query = f"{user_q} (请使用中文回答，并输出详细的数据对比表格)"
            response = s_engine.query(final_query)
            
            print("\n" + "-"*30)
            print(f"📄 最终分析报告：\n\n{response}")
            print("-" * 30)
            
        except Exception as e:
            print(f"❌ 运行出错: {e}")

if __name__ == "__main__":
    run_pro_rag()