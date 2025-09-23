import os

from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from dotenv import load_dotenv
load_dotenv(verbose=True)

# ✅ 覆盖默认 embedding 模型
Settings.embed_model = DashScopeEmbedding(
    model="text-embedding-v4",
    dimension=1024
)
# ✅ 设置全局 LLM
Settings.llm = DashScope(
    model="qwen-plus",
    api_key=os.environ["DASHSCOPE_API_KEY"]  # 可省略，如果已设置环境变量
)  # 你也可以换成 qwen-max, qwen-turbo 等


# 加载文档
documents = SimpleDirectoryReader("./notion_md").load_data()

# 构建索引（此时用的是 DashScopeEmbedding，不会再走 OpenAI）
index = VectorStoreIndex.from_documents(documents)

# 提问
query_engine = index.as_query_engine()
response = query_engine.query("帮我总结一下k6的主要功能？")
print(response)