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

# Initialize default values
temperature = 0.0
web_path = "https://lilianweng.github.io/posts/2023-06-23-agent/"

def load_and_process_data(web_path):
    loader = WebBaseLoader(
        web_path=(web_path,),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-title", "post-content", "post-header"))),
        requests_kwargs={"headers": {"User-Agent": "MyWebScraperBot/1.0"}}
    )
    text_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(text_documents)
    return docs

def setup_vectorstore(docs):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings()
    # The embeddings are stored in a Chroma vector store, 
    # which allows for efficient retrieval of relevant text chunks.
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    
    # A retriever is created from the vector store, 
    # which can fetch relevant documents based on a query.
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def setup_llm(temperature):
    groq_api_key = os.getenv("GROQ_API_KEY")
    # Set up LLM with specified parameters like temperature.
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.2-1b-preview",
        temperature=temperature,
        max_retries=2
    )

def setup_chain(retriever, llm):
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    # A chain is created that combines the retriever and the LLM.
    # This chain takes a user query, 
    # retrieves relevant documents from the vector store, 
    # and then uses the LLM to generate an answer based on these documents.
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

print("Welcome to the query system!")
print("You can change the temperature, web path, or enter queries.")
print("Type 'temp' to change temperature, 'web' to change web path, or 'bye' to exit.")
print("From the Groq Documentation at https://console.groq.com/docs/api-reference, valid values for temperature are between 0 and 2.")
print("Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic") 

docs = load_and_process_data(web_path)
retriever = setup_vectorstore(docs)
llm = setup_llm(temperature)
chain = setup_chain(retriever, llm)

while True:
    user_input = input(f"\nCurrent temperature: {temperature:.1f}\nCurrent web path: {web_path}\nEnter your query, 'temp', 'web', or 'bye': ")
    
    if user_input.lower() == "bye":
        # print("Goodbye!")
        break
    elif user_input.lower() == "temp":
        try:
            new_temp = float(input("Enter new temperature (0.0 to 2.0): "))
            if 0 <= new_temp <= 2:
                temperature = new_temp
                llm = setup_llm(temperature)
                chain = setup_chain(retriever, llm)
                print(f"Temperature updated to {temperature:.1f}")
            else:
                print("Invalid temperature. Please enter a value between 0.0 and 2.0.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")
    elif user_input.lower() == "web":
        new_web_path = input("Enter new web path: ")
        web_path = new_web_path
        docs = load_and_process_data(web_path)
        retriever = setup_vectorstore(docs)
        chain = setup_chain(retriever, llm)
        print(f"Web path updated to {web_path}")
    else:
        result = chain.invoke({"input": user_input})
        print(result['answer'])

print("Goodbye!")