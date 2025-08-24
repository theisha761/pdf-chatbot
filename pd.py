from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
st.title("chat with your pdf 📝")
uploaded_file=st.file_uploader("upload your pdf", type="pdf")
if uploaded_file is not None:
    with open("temp.pdf","wb") as f:
        f.write(uploaded_file.read())
        loader=PyPDFLoader("temp.pdf")
        docs=loader.load()
        spilliter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=10)
        chunks=spilliter.split_documents(docs)
        embed=OpenAIEmbeddings()
        vector_store=Chroma.from_documents(chunks,embed)
        model=ChatOpenAI()
        chain=RetrievalQA.from_llm(model,retriever=vector_store.as_retriever(search_kwargs={"k":2}))
        query=st.text_input("ask question related to pdf")
        if query:
            result=chain.run(query)
            st.write(result)
           
    
