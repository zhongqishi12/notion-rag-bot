import os
from typing import List

import faiss
import dashscope
from http import HTTPStatus
from dotenv import load_dotenv
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.faiss import FaissVectorStore

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


class DashScopeEmbeddingWrapper(BaseEmbedding):
    # 声明为 Pydantic 字段
    model: str = "text-embedding-v4"
    dimension: int = 1024

    def __init__(self, model: str = "text-embedding-v4", dimension: int = 1024):
        # 调用父类以创建 Pydantic 模型
        super().__init__(model=model, dimension=dimension)

    def _embed(self, text: str):
        resp = dashscope.TextEmbedding.call(
            model=self.model,
            input=text,
            dimension=self.dimension,
            output_type="dense"
        )
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"DashScope Error: {resp.message}")
        return resp.output["embeddings"][0]["embedding"]

    # -------- 同步接口（BaseEmbedding 要求） --------
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    # -------- 异步接口（新版抽象类新增） --------
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_embeddings(self, texts: List[str]) -> List[List[float]]:
        # 简单串行；如需提速可改为并发
        return [self._get_text_embedding(t) for t in texts]


def build_index(
        data_dir: str = "notion_md",
        persist_dir: str = "faiss_storage",
        model: str = "text-embedding-v4",
        dim: int = 1024,
        chunk_size: int = 800,
        chunk_overlap: int = 120
):
    # 1. 加载原始文档 (仅 md)
    documents = SimpleDirectoryReader(
        input_dir=data_dir,
        required_exts=[".md"]
    ).load_data()
    print(f"加载文档数: {len(documents)}")

    # 2. 分块
    parser = SimpleNodeParser.from_defaults(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    nodes = parser.get_nodes_from_documents(documents)
    print(f"切分后节点数: {len(nodes)}")

    # 3. 初始化 FAISS
    faiss_index = faiss.IndexFlatL2(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. 自定义 DashScope Embedding
    embed_model = DashScopeEmbeddingWrapper(model=model, dimension=dim)

    # 5. 构建索引
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model
    )

    # 6. 持久化
    storage_context.persist(persist_dir=persist_dir)
    print(f"✅ 向量索引已保存到: {persist_dir}")

    return index


if __name__ == "__main__":
    build_index()
