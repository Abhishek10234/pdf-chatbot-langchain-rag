import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import streamlit as st

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.title("PDF Chatbot with LangChain (RAG)")

# Upload PDF
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf:
    # Read PDF
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content

    # Split into chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    # Store vectors in FAISS
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

    # Final QA system (RAG)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # User query
    query = st.text_input("Ask a question about the PDF:")

    if query:
        answer = qa.run(query)
        st.write(answer)