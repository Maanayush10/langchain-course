import os

from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__=="__main__":
    print("Ingesting...")
    loader = UnstructuredLoader(file_path="C:/Users/maana/OneDrive/Desktop/MyProjects/Langchain/langchain-course/mediumblog1.txt")
    # print("*"*100)
    # print(loader)
    # print("*"*100)
    document= loader.load()
    # print(document)
    # print("*"*100)
    print("Splitting ...")

    text_splitter= CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts= text_splitter.split_documents(document)
    # print("*"*100)
    # print(texts)
    # print("*"*100)

    print(f"created {len(texts)} chunks")

    # embeddings= GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", api_key=os.environ.get("GOOGLE_API_KEY"))

    embeddings= OllamaEmbeddings(model= "bge-m3:latest")

    print("*"*100)
    print(embeddings)
    print("*"*100)
    print("Ingesting")

    PineconeVectorStore.from_documents(
        texts, embeddings, index_name= os.environ["INDEX_NAME"]
    )
    print("finish.")