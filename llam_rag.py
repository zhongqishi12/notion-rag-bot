import os

from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from dotenv import load_dotenv
from build_vector_db import DashScopeEmbeddingWrapper
load_dotenv(verbose=True)

# ✅ 设置全局 LLM
Settings.llm = DashScope(
    model="qwen-plus",
    api_key=os.environ["DASHSCOPE_API_KEY"]  # 可省略，如果已设置环境变量
)  # 你也可以换成 qwen-max, qwen-turbo 等

# ✅ 设置全局 Embedding
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v4",
    dimension=1024,
    api_key=os.environ["DASHSCOPE_API_KEY"]
)

PERSIST_DIR = "./faiss_storage"
vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
print("FAISS 索引维度:", vector_store._faiss_index.d)

# 2. 构造 storage_context（显式传入 vector_store）
storage_context = StorageContext.from_defaults(
    persist_dir=PERSIST_DIR,
    vector_store=vector_store
)
# 3. 加载索引
index = load_index_from_storage(storage_context)

# 4. 查询
query_engine = index.as_query_engine(
    similarity_top_k=5, embedding_model=Settings.embed_model
)
response = query_engine.query("帮我总结一下k6的主要功能？")
print(response)