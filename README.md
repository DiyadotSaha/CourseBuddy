# CourseBuddy

A context-aware chatbot designed for UCI students, offering quick access to course details and tailored responses to their queries. Featuring a robust course information database, Langchain retriever, and prompt template, the chatbot delivers efficient and personalized assistance.

 **Watch the Live Demo:** [https://youtu.be/VWcLdm3qz3I?si=FpXSdaONgbQ9MKQU](https://youtu.be/VWcLdm3qz3I?si=FpXSdaONgbQ9MKQU)

![CourseBuddy Demo](vid_demo.gif)

---
## How It's Made

**Tech used:** Python, Streamlit, LangChain, FAISS, HuggingFace Transformers, BeautifulSoup

CourseBuddy is built as a Retrieval-Augmented Generation (RAG) system to provide accurate and contextually relevant responses about UCI courses. The development process involves several key components:

### Data Collection and Preprocessing

- **HTML Parsing:** Utilized BeautifulSoup to scrape course information from UCI's official course catalog.
- **Text Conversion:** Converted HTML content into plain text files for processing.

### Embedding and Vector Store

- **Embeddings:** Generated vector embeddings using HuggingFace's `WhereIsAI/UAE-Large-V1` model.
- **Vector Database:** Stored embeddings in a FAISS vector store for efficient similarity search.

### Retrieval and Question Answering

- **Language Model:** Integrated HuggingFace's `Mistral-7B-Instruct-v0.2` model via API.
- **Prompt Template:** Designed custom prompts to guide the model's responses based on retrieved context.
- **Retrieval Chain:** Implemented a retrieval chain using LangChain to fetch relevant documents and generate answers.

### User Interface

- **Streamlit App:** Developed an interactive UI with Streamlit, allowing users to input queries and receive responses in a chat-like format.
- **Session Management:** Maintained conversation history using Streamlit's session state for a seamless user experience.

---

## Optimizations

- **Reduced Hallucinations:** By grounding responses in retrieved context, the chatbot minimizes the risk of generating inaccurate information.
- **Efficient Retrieval:** FAISS enables fast and scalable similarity searches over the embedded course data.
- **Modular Design:** The system's components are modular, facilitating easy updates and maintenance.

---

## Lessons Learned

Developing CourseBuddy provided insights into:

- **RAG Systems:** Understanding the integration of retrieval mechanisms with generative models to enhance response accuracy.
- **User Experience:** Designing intuitive interfaces that effectively convey AI-generated information to users.
