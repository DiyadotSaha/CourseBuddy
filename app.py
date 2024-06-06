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
from langchain.llms import HuggingFaceHub

from langchain_community.embeddings import OpenAIEmbeddings 
import getpass
from langchain.chains import LLMChain
import time
global_token = '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'

 
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
    final_path='/Users/diyasaha/CourseBuddy/faissDB'
    model_id='WhereIsAI/UAE-Large-V1'
    embeddings=HuggingFaceEmbeddings(model_name=model_id)
    #vectorstore = FAISS.load_local(final_path,embeddings,allow_dangerous_deserialization=True)
    vectorstore = FAISS.load_local(final_path,embeddings=embeddings)
    return vectorstore





def ask_question(question,chain):
    resp = chain.invoke({"question": question})
    return resp['answer']


if __name__ == "__main__":
    st.title("CourseBuddy")
    st.subheader("Your friendly AI chat bot for course-related queries")
    st.write("Hello! I am CourseBuddy, your friendly AI chat bot. Feel free to ask me anything related to courses!")
    vectorstore = loadVectorStore()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = global_token
    llm = HuggingFaceHub(repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1', 
                         huggingfacehub_api_token=global_token,
                         model_kwargs={"temperature": 0.7, "max_length": 64, "max_new_tokens": 512})
    llm.client.api_url = 'https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1'
    print("==== LOADED VECTOR DATABASE ===")
    if "messages" not in st.session_state:
        st.session_state.messages= []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt=st.chat_input("Ask something!")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        question=prompt
        docs = vectorstore.similarity_search(question)
        print(docs)
        prompt_template = """ You are a nice and helpful AI chat bot. Do not try to complete the question. Generate an answer based on the context and question.
    QUESTION: {question}
    CONTEXT: {context}
        """ 
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        llm_chain = LLMChain(llm=llm, prompt=PROMPT)
        response=llm_chain.run({'context':docs, 'question':question})
        with st.chat_message("assistant"):
            message_placeholder=st.empty()
            assistant_response = response
            full_response = ""
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

