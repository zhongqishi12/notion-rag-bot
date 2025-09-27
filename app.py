import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from openai import OpenAI

# 你写过的 DashScope Embeddings 类
from build_vector_db import DashScopeEmbeddings

load_dotenv()

# 初始化向量库
embeddings = DashScopeEmbeddings(model="text-embedding-v4", dimension=1024)
vs = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# 初始化 LLM (通义 DashScope 兼容模式)
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

st.set_page_config(page_title="📚 Notion RAG Chatbot", layout="wide")
st.title("📚 我的私人知识库 Chatbot")

# 保持对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 展示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户问题
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 向量检索
    docs = vs.similarity_search(prompt, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # 拼接 Prompt
    full_prompt = f"""你是一个知识库助手，请仅基于以下内容回答问题。
    内容:
    {context}

    问题: {prompt}
    """

    # 调用 LLM
    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": full_prompt}],
    )

    answer = resp.choices[0].message.content
    sources = [d.metadata.get("source") for d in docs]

    # 展示助手回答
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.markdown(f"**来源:** {sources}")
    st.session_state.messages.append({"role": "assistant", "content": answer})