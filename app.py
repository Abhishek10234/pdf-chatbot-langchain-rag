import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import streamlit as st

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.title("📄 PDF Chatbot with LangChain (RAG)")

pdf = st.file_uploader("Upload your PDF", type="pdf")
if pdf:
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    query = st.text_input("Ask a question about the PDF:")
    if query:
        docs = vectorstore.similarity_search(query, k=3)
        llm = ChatOpenAI(openai_api_key=openai_api_key)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        st.write(response)
