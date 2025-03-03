# AI Question Answering System

This project implements an AI-powered question answering system using LangChain, OpenAI, and Groq. It retrieves information from web content, processes it, and uses a language model to answer questions based on the retrieved context.

## How Query Processing with RAG works in this program
When a user enters a query, the chain:
1. Uses the retriever to find relevant chunks of text from the web content.
2. Passes these chunks along with the query to the LLM.
3. The LLM generates an answer based on the retrieved context and the query.

## Prerequisites
* MLflow setup on Minikube is required for this application. The setup process is described in the `mlflow` subdirectory of this project

## Setup

1. Clone this repository:
```
git clone <repo url>
cd <repo url>
```


2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```


3. Set up environment variables:
- Create a `.env` file in the project root directory
- Add the following lines to the `.env` file:
  ```
  OPENAI_API_KEY=your_openai_api_key
  GROQ_API_KEY=your_groq_api_key
  ```
Replace `your_openai_api_key` and `your_groq_api_key` with your actual API keys.

## Usage

1. Run the script:
```
python web_content_retriever.py
```
OR
```
streamlit run gui_web_content_retriever.py
```

2. The script will:
- Load content from a specified web page
- Process and split the content into manageable chunks
- Create embeddings and store them in a vector database
- Set up a retrieval system and language model
- Answer a predefined question using the processed information

3. To change the question or add more functionality, modify the `query` variable or extend the script as needed.

## Logging
The application logs metrics, requests and responses to MLflow. This allows for tracking and analysis of the system's performance and behavior over time.

## Requirements

The `requirements.txt` file should include the following packages:
```
langchain
langchain_community
langchain_core
langchain_groq
openai
chromadb
beautifulsoup4
```

Make sure to install these dependencies using the command mentioned in the Setup section.

## Choice to technologies:
1. Chroma was used as a vector store because it is ideal for small sample applications due to its simplicity, lightweight nature, and easy embedding in Python applications. It's good for quick prototypes and small-scale AI projects.
2. OpenAI API was used for generating embeddings
3. Groq's Inferencing API was used for its high-speed performance, low-latency text-generation, and easy integration with language models. Its API is designed to be compatible with popular frameworks like LangChain.
4. MLflow was used for logging parameters, metrics and atrifacts for each run. This helps with reproducibility and analysis of experiments.
   

## Note

Ensure you have valid API keys for OpenAI and Groq services before running the script. The effectiveness of the question answering system depends on the quality and relevance of the web content loaded and processed.

This README provides instructions for setting up the environment, using the script, and lists the required packages. You should create a requirements.txt file with the packages listed in the README for easy installation.

## Mlflow Setup
For detailed instructions on setting up MLflow on Minikube, please refer to the `mlflow` subdirectory in this project. The setup is a prerequiste for running the application enabling the logging functionality.

