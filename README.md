# 📚 Smart RAG Chatbot with Groq & Hugging Face

This is a **Streamlit-based Retrieval-Augmented Generation (RAG) chatbot** that allows you to paste github repo link, upload code files, upload documents, build embeddings, and chat with them using **Groq LLMs** and **Hugging Face embeddings**.  
It uses **FAISS** as the vector database to store and retrieve chunks of your documents.

---

## ✨ Features
- 🔑 Secure API key management via `.env`
- 📂 Upload documents (PDF, TXT, DOCX, etc.)
- 🔍 FAISS vector store for efficient similarity search
- 💬 Conversational memory for smooth multi-turn chats
- ⚡ Powered by **Groq LLMs** (fast inference) + **Hugging Face Embeddings**
- 🌐 Deployed easily with **Streamlit Cloud**

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) – UI Framework  
- [LangChain](https://www.langchain.com/) – RAG Orchestration  
- [Groq](https://groq.com/) – LLM Backend  
- [Hugging Face](https://huggingface.co/) – Embeddings  
- [FAISS](https://github.com/facebookresearch/faiss) – Vector Database  

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
