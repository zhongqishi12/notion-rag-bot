import os
from dotenv import load_dotenv
from langchain_community.document_loaders import NotionDBLoader, NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./faiss_index")


def load_from_notion():
    # 方案 A：从 Notion 数据库加载（推荐结构化知识库）
    #db_id = os.getenv("NOTION_DB_ID")
    db_id = 'fb0fa29a80364b139c98422638b866b6'
    if db_id:
        loader = NotionDBLoader(integration_token=NOTION_TOKEN, database_id=db_id)
        docs = loader.load()
        print(docs)
    else:
        # 方案 B：从 Notion 导出 zip 解压后的目录（离线、无需 API）
        loader = NotionDirectoryLoader(path="./notion_export")
        docs = loader.load()

    # 追加一些元数据清洗/补齐在此处（可选）
    return docs


def main():
    print("⏳ Loading data from Notion...")
    docs = load_from_notion()

    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    # )
    # chunks = splitter.split_documents(docs)
    #
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="BAAI/bge-m3",
    #     encode_kwargs={"normalize_embeddings": True}
    # )
    #
    # vs = FAISS.from_documents(chunks, embedding=embeddings)
    # vs.save_local(VECTOR_STORE_PATH)
    # print(f"✅ Indexed {len(chunks)} chunks → {VECTOR_STORE_PATH}")


if __name__ == "__main__":
    main()
