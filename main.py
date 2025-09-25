import streamlit as st
import os
import re
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.readers.github import GithubRepositoryReader, GithubClient


def parse_github_url(url: str):
    pattern = r"https?://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+))?"
    match = re.match(pattern, url)
    if not match:
        raise ValueError("Invalid GitHub repository URL")
    owner, repo, branch = match.groups()
    return owner, repo, branch if branch else "main"


@st.cache_resource
def load_github_data(github_token: str, owner: str, repo: str, branch: str = "main"):
    github_client = GithubClient(github_token)
    loader = GithubRepositoryReader(
        github_client,
        owner=owner,
        repo=repo,
        filter_file_extensions=(
            [".py", ".ipynb", ".js", ".ts", ".md"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        verbose=False,
        concurrent_requests=8,
    )
    docs = loader.load_data(branch=branch)
    processed = []
    for d in docs:
        text = d.get_text() if hasattr(d, "get_text") else getattr(d, "text", str(d))
        meta = getattr(d, "extra_info", {}) or {}
        processed.append({"page_content": text, "metadata": meta})
    return processed


@st.cache_resource
def build_vectorstore(docs, hf_token: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"use_auth_token": hf_token},
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts, metadatas = [], []
    for d in docs:
        splits = splitter.split_text(d["page_content"])
        texts.extend(splits)
        metadatas.extend([d.get("metadata", {})] * len(splits))

    vectordb = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectordb


def create_chain(_vectordb, groq_api_key: str, temperature: float = 0.0, top_k: int = 3):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=temperature,
        streaming=True,
        groq_api_key=groq_api_key,
    )
    retriever = _vectordb.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, return_source_documents=True
    )
    return chain


# ---------------- Main App ----------------
def chat_page():
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; padding:10px 0;">
            <div style="background:linear-gradient(90deg, #ff4b4b, #ffbb33); 
                        border-radius:50%; width:36px; height:36px; display:flex; 
                        align-items:center; justify-content:center; font-weight:bold; color:white;">
                🤖
            </div>
            <h2 style="margin:0; font-family:sans-serif;">Chat with Code — Groq + LangChain</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        groq_api_key = st.text_input("🔑 Groq API Key", type="password")
        hf_token = st.text_input("🔑 HuggingFace Token", type="password")
        github_token = st.text_input("🔑 GitHub Token", type="password")

        st.markdown("---")
        st.markdown("### 📂 Repository / Files")
        repo_url = st.text_input("GitHub repository URL", placeholder="https://github.com/owner/repo")
        uploaded_files = st.file_uploader(
            "Or upload your code files",
            type=["py", "ipynb", "js", "ts", "md", "java", "cpp", "c", "cs", "txt"],
            accept_multiple_files=True,
        )

        st.markdown("---")
        temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.05)
        k = st.number_input("Retriever top-k", min_value=1, max_value=10, value=3)

        if st.button("🚀 Build Index"):
            if not (groq_api_key and hf_token):
                st.error("Please enter Groq & HuggingFace API keys")
            else:
                try:
                    docs = []
                    if repo_url and github_token:
                        owner, repo, branch = parse_github_url(repo_url)
                        docs = load_github_data(github_token, owner, repo, branch)
                    if uploaded_files:
                        for f in uploaded_files:
                            text = f.read().decode("utf-8", errors="ignore")
                            docs.append({"page_content": text, "metadata": {"source": f.name}})
                    if not docs:
                        st.error("Please provide a GitHub repo or upload code files.")
                    else:
                        with st.spinner("Building FAISS vectorstore..."):
                            vectordb = build_vectorstore(docs, hf_token)
                            st.session_state["vectordb"] = vectordb
                            st.session_state["docs"] = docs
                            st.success("✅ Index built successfully")
                except Exception as e:
                    st.error(str(e))

    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = None
    if "vectordb" in st.session_state and groq_api_key and hf_token:
        try:
            chain = create_chain(st.session_state["vectordb"], groq_api_key, temperature, int(k))
        except Exception as e:
            st.error(f"Error creating chain: {e}")

    
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Type your question about the repository or uploaded code..."):
        if not chain:
            st.error("Please build an index first from sidebar.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""

                try:
                    with st.spinner("Thinking..."):
                        history = st.session_state.memory.load_memory_variables({})["chat_history"]
                        if len(history) > 5:
                            history = history[-5:]

                        for chunk in chain.stream({"question": prompt, "chat_history": history}):
                            text_chunk = chunk.get("answer") or chunk.get("result") or ""
                            full_response += text_chunk
                            placeholder.markdown(full_response + "▌")

                    placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"Error during query: {e}")


def about_page():
    st.title("📘 About Chat with Code")
    st.markdown(
        """
        ### 🔹 What is this tool?
        **Chat with Code** is a Retrieval-Augmented Generation (RAG) app that allows you to:
        - Chat with any GitHub repository by scraping its code & docs.
        - Upload your own code files (Python, JS, C++, Java, etc.) and query them.
        - Use **Groq LLMs** for fast inference and **FAISS vectorstore** for efficient search.
        - Ask natural language questions and get answers grounded in your code.

        ### 🔹 How it works
        1. Enter your API keys in the sidebar.
        2. Provide a GitHub repository URL *or* upload code files.
        3. The app builds a vectorstore of your code using **HuggingFace embeddings**.
        4. You can then chat with the codebase in real-time using Groq.

        ### 🔹 Required API Keys & Guides
        - **Groq API Key** → [Get it here](https://console.groq.com)  
          Used for running the LLM (Groq free tier gives 8k tokens).  

        - **HuggingFace Token** → [Get it here](https://huggingface.co/settings/tokens)  
          Needed for downloading sentence-transformer embeddings.  

        - **GitHub Token (Optional)** → [Get it here](https://github.com/settings/tokens)  
          Only required if you want to index private repositories.  

        ### 🔹 Deployment
        This app can run locally or be deployed on **Streamlit Cloud**.  
        When deployed, users simply need to paste their keys in the sidebar — no `.env` file required.  

        ### 🔹 Why use it?
        - Debug and understand large repositories faster.  
        - Learn from open-source projects interactively.  
        - Upload your own project and query it like ChatGPT.  

        ---
        💡 **Pro Tip:** You can adjust the **temperature** (creativity) and **retriever top-k** (number of code chunks retrieved) from the sidebar.
        """
    )


def main():
    st.set_page_config(page_title="Chat with Code", layout="wide")

    page = st.sidebar.radio("📍 Navigation", ["💬 Chat", "ℹ️ About the Agent"])

    if page == "💬 Chat":
        chat_page()
    elif page == "ℹ️ About the Agent":
        about_page()


if __name__ == "__main__":
    main()
