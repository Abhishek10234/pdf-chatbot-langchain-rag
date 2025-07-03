# 🧠 AI PDF Chatbot using LangChain (RAG)

This is a Retrieval-Augmented Generation (RAG)-based chatbot app built using LangChain, OpenAI, and Streamlit. Upload any PDF and ask questions — the bot will answer based on content from the PDF.

## 🔧 Tech Stack
- LangChain
- FAISS (vector DB)
- OpenAI GPT
- PyPDF2
- Streamlit

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure you create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_key_here
```

## 🧪 Example Questions
- "What is the summary of this document?"
- "List 3 key points from the PDF."
