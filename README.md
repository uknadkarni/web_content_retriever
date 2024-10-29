# AI Question Answering System

This project implements an AI-powered question answering system using LangChain, OpenAI, and Groq. It retrieves information from web content, processes it, and uses a language model to answer questions based on the retrieved context.

## Setup

1. Clone this repository:
```
git clone git@github.com:uknadkarni/web_content_retriever.git
cd web_content_retriever
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


2. The script will:
- Load content from a specified web page
- Process and split the content into manageable chunks
- Create embeddings and store them in a vector database
- Set up a retrieval system and language model
- Answer a predefined question using the processed information

3. To change the question or add more functionality, modify the `query` variable or extend the script as needed.

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

## Note

Ensure you have valid API keys for OpenAI and Groq services before running the script. The effectiveness of the question answering system depends on the quality and relevance of the web content loaded and processed.

This README provides instructions for setting up the environment, using the script, and lists the required packages. You should create a requirements.txt file with the packages listed in the README for easy installation.
