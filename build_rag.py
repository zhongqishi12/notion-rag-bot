import os
import dashscope
from http import HTTPStatus
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


# 简单的封装函数
def embed_text(text: str, model="text-embedding-v4", dimension=1024):
    resp = dashscope.TextEmbedding.call(
        model=model,
        input=text,
        dimension=dimension,
        output_type="dense"
    )
    if resp.status_code == HTTPStatus.OK:
        return resp.output["embeddings"][0]["embedding"]
    else:
        raise RuntimeError(f"DashScope Error: {resp.message}")


# 读取前 10 个 md 文件
def load_md_files(md_dir="notion_md", limit=10):
    docs = []
    for i, fname in enumerate(os.listdir(md_dir)):
        if i >= limit:
            break
        if fname.endswith(".md"):
            print(f"Loading {fname}...")
            with open(os.path.join(md_dir, fname), "r", encoding="utf-8") as f:
                text = f.read()
            docs.append(Document(page_content=text, metadata={"source": fname}))
    return docs


# 定义 DashScope Embeddings 包装类（可复用）
class DashScopeEmbeddings(Embeddings):
    def __init__(self, model="text-embedding-v4", dimension=1024):
        self.model = model
        self.dimension = dimension

    def embed_documents(self, texts):
        return [embed_text(t, model=self.model, dimension=self.dimension) for t in texts]

    def embed_query(self, text):
        return embed_text(text, model=self.model, dimension=self.dimension)


if __name__ == "__main__":
    docs = load_md_files("notion_md", limit=10)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embeddings = DashScopeEmbeddings()

    # 存储到 FAISS
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local("faiss_index")
    print("✅ 已保存到 FAISS 向量库")

    # 测试查询
    vs = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    query = "审批流程是怎么写的？"
    results = vs.similarity_search(query, k=3)

    for r in results:
        print("\n==== 检索结果 ====")
        print(r.page_content[:200])
        print("来源:", r.metadata.get("source"))