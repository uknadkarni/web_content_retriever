#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4


# In[12]:


from langchain_community.document_loaders import WebBaseLoader
import bs4


# In[13]:

loader=WebBaseLoader(web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("post-title", "post-content", "post-header"))),
                     requests_kwargs={"headers": {"User-Agent": "MyWebScraperBot/1.0"}}
)

                         
                     
text_documents=loader.load()


# In[14]:


text_documents


# In[15]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs=text_splitter.split_documents(text_documents)
docs[:5]


# In[16]:


## Convert Data into Vectors and store in AstraDB
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
embeddings=OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=docs,
                                    embedding=embeddings)



# In[17]:


# Create a retriever from the vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# In[18]:


# Setup ChatGroq model
groq_api_key=os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.2-1b-preview",
    temperature=0.0,
    max_retries=2
)


# In[97]:


chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("human", "Context: {context}\n\nQuestion: {question}"),
])


# In[1]:


system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

#Define the query
# query = "What is the capital of France?"
query = "What are the components of an Agent System?"
result = chain.invoke({"input": query})
print(result['answer'])
