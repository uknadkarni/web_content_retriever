import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
import chromadb
import mlflow
import time

# Initialize default settings
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.0
if 'web_path' not in st.session_state:
    st.session_state.web_path = "https://lilianweng.github.io/posts/2023-06-23-agent/"

# Set up Chroma database for persistent storage
# Chroma database is set up for persistent storage to save and load the database from the local machine, 
# allowing data to persist between sessions.
CHROMA_PERSIST_DIR = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# @st.cache_resource caches the output of the decorated function, 
# reusing the result for subsequent calls with the same input, which improves performance.
@st.cache_data
def load_and_process_data(web_path):
    """Fetch web content and split into chunks"""
    try:
        loader = WebBaseLoader(
            web_path=(web_path,),
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-title", "post-content", "post-header"))),
            requests_kwargs={"headers": {"User-Agent": "MyWebScraperBot/1.0"}}
        )
        text_documents = loader.load()
        if not text_documents:
            st.warning("No content found at the URL.")
            return None
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_documents(text_documents)
    except Exception as e:
        st.error(f"Error loading web content: {str(e)}")
        return None

@st.cache_resource
def setup_vectorstore(_docs):
    """Create a searchable database from document chunks"""
    if not _docs:
        return None
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings()
    # Chroma.from_documents creates a searchable vector database from the provided documents, 
    # using the specified embedding fUnction.
    return Chroma.from_documents(
        documents=_docs, 
        embedding=embeddings, 
        client=chroma_client,
        collection_name="my_collection"
    )

def setup_llm(temperature):
    """Initialize the language model"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.2-1b-preview",
        temperature=temperature,
        max_retries=2
    )

def setup_rag_chain(vectorstore, llm):
    """Create a chain for retrieval-augmented generation"""
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    # chain_type is set to "stuff" to use the simplest method of 
    # combining retrieved documents with the query, 
    # which is to stuff all retrieved documents into the prompt.
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        # chain_type_kwargs is a parameter that allows passing additional arguments 
        # to customize the behavior of the chain, 
        # in this case, specifying a custom prompt.
        chain_type_kwargs={"prompt": prompt}
    )

def setup_simple_chain(llm):
    """Create a basic Q&A chain without retrieval"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Provide concise answers."),
        ("human", "{question}")
    ])
    # A RunnableSequence in the setup_simple_chain function is a 
    # chain of operations created using the | operator, 
    # combining the prompt template with the language model.
    return prompt | llm  # Using the | operator to create a RunnableSequence

# Set up the Streamlit UI
st.title("Query System with Optional Retrieval Augmentation")

# Get Minikube IP
minikube_ip = os.popen('minikube ip').read().strip()

# Set MLflow tracking URI using Minikube IP and MLflow service NodePort
# Assuming MLflow service is exposed on NodePort 30500
mlflow.set_tracking_uri(f"http://localhost:5000")

# Initialize MLflow
# Set or create the experiment
experiment_name = f"Groq_API_Interaction_{int(time.time())}"
# experiment_name = "Groq_API_Interaction"
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
except Exception as e:
    st.error(f"Error setting up MLflow experiment: {str(e)}")
    
# Sidebar for user settings
st.sidebar.header("Settings")
new_temperature = st.sidebar.slider("Temperature", 0.0, 2.0, st.session_state.temperature, 0.1)
new_web_path = st.sidebar.text_input("Web Path (Optional)", st.session_state.web_path)

# Update settings if changed
if new_temperature != st.session_state.temperature or new_web_path != st.session_state.web_path:
    st.session_state.temperature = new_temperature
    st.session_state.web_path = new_web_path
    st.rerun()

# Initialize the language model
llm = setup_llm(temperature=st.session_state.temperature)

# Set up the appropriate chain based on web path
if st.session_state.web_path:
    docs = load_and_process_data(st.session_state.web_path)
    if docs:
        vectorstore = setup_vectorstore(_docs=docs)
        if vectorstore:
            chain = setup_rag_chain(vectorstore=vectorstore, llm=llm)
            st.success("Using Retrieval Augmented Generation")
        else:
            st.warning("Vector store creation failed. Using simple LLM.")
            chain = setup_simple_chain(llm=llm)
    else:
        st.warning("Web content loading failed. Using simple LLM.")
        chain = setup_simple_chain(llm=llm)
else:
    chain = setup_simple_chain(llm=llm)
    st.info("Using simple LLM without retrieval")

# Process user query
user_input = st.text_input("Enter your query:")

#if user_input:
#    try:
#        # isinstance(chain, RetrievalQA) returns True if the chain is an instance of RetrievalQA, 
#        # indicating it's using retrieval-augmented generation.
#        if isinstance(chain, RetrievalQA):
#            # "query" is used for RetrievalQA chains that perform document retrieval
#            result = chain.invoke({"query": user_input})
#            st.write(result['result'])
#        else:
#            # "question" is used for simple LLM chains without retrieval.
#            result = chain.invoke({"question": user_input})
#            st.write(result.content)
#    except Exception as e:
#        st.error(f"An error occurred: {str(e)}")
#        st.error("Please try reloading the page or check your input.")


if user_input:
    try:
        # End any existing runs
        mlflow.end_run()

        with mlflow.start_run():
            # Common parameters for both RAG and non-RAG
            mlflow.log_param("temperature", st.session_state.temperature)
            mlflow.log_param("model_name", "llama-3.2-1b-preview")
            mlflow.log_param("max_retries", 2)
            mlflow.log_param("user_input", user_input)

            if isinstance(chain, RetrievalQA):
                # RAG-specific logging
                mlflow.log_param("chain_type", "RetrievalQA")
                mlflow.log_param("web_path", st.session_state.web_path)

                result = chain.invoke({"query": user_input})
                response = result['result']
                
                # Log RAG-specific metadata
                rag_metadata = {
                    "num_source_documents": len(result.get('source_documents', [])),
                    "source_document_titles": [doc.metadata.get('title', 'Untitled') for doc in result.get('source_documents', [])]
                }
                mlflow.log_dict(rag_metadata, "rag_metadata.json")

            else:
                # Non-RAG specific logging
                mlflow.log_param("chain_type", "Simple LLM")
                
                result = chain.invoke({"question": user_input})
                response = result.content

            # Log the response for both RAG and non-RAG
            mlflow.log_text(response, "response.txt")

            # Log common response metadata
            response_metadata = {
                "response_length": len(response),
            }
            mlflow.log_dict(response_metadata, "response_metadata.json")

            # Display the response
            st.write(response)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try reloading the page or check your input.")
        # Log the error to MLflow
        mlflow.log_param("error", str(e))
        
        
# Display temperature information
st.sidebar.markdown("""
From the Groq Documentation:
- Valid values for temperature are between 0 and 2.
- Higher values like 0.8 will make the output more random.
- Lower values like 0.2 will make it more focused and deterministic.
""")
