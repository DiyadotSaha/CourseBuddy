
import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings 
import getpass

def extract_info_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        info_text = '\n'.join(p.text for p in paragraphs)
        return info_text
    else:
        print("Failed to retrieve data from URL:", url)
        return None

def save_to_text_file(text, filename, folder='extracted_info'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, filename)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

def returnSplits():
    loader=DirectoryLoader('extracted_info', show_progress=True, use_multithreading=True)
    documents=loader.load()
    return documents

def createVectorStore():
    embeddings = HuggingFaceEmbeddings(model_name='WhereIsAI/UAE-Large-V1')
    store = InMemoryStore()
    vectorstore = FAISS.from_documents(returnSplits(), embedding=embeddings)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=512)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=256)
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    retriever.add_documents(returnSplits())
    final_path='faissDB'
    vectorstore.save_local(final_path)
    return vectorstore
    
def loadVectorStore():
    final_path='faissDB'
    embeddings=HuggingFaceEmbeddings(model_name='WhereIsAI/UAE-Large-V1')
    vectorstore = FAISS.load_local(final_path,embeddings,allow_dangerous_deserialization=True)
    return vectorstore

def creatingChain():
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.5,
        token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
    )
    template = """You are Shakespeare and based on the context, always write one verse.
    Question: {question} 
    Context: {summaries}
    Answer:"""
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={
        "prompt": PromptTemplate(
            template=template,
            input_variables=["summaries", "question"],
            ),
        },
    )
    return chain

def ask_question(question,chain):
    resp = chain.invoke({"question": question})
    return resp['answer']



if __name__ == "__main__":
    vectorstore = loadVectorStore()
    print ("==== LOADED VECTOR DATABASE ===")
    st.title("Course Buddy")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.keys():
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Get response from backend
        chain = creatingChain()
        response = ask_question(prompt, chain)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
