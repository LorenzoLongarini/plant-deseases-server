# from langchain_community.document_loaders.pdf import PyPDFLoader #, UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
import markdown
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

vectordb, chain = None, None


def split_docs(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


def init(use_ollama = True):
    persist_directory = f"assets/chroma_db_{'ollama' if use_ollama else 'openai'}"

    if use_ollama:
        embeddings = OllamaEmbeddings(model='all-minilm')
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    if not os.path.exists(persist_directory):
        print('Creation of NEW db')
        # pdf_url = "https://drive.google.com/uc?id=1ihoa-Db-PytLVvTHDfIFmBF6FZJpsoXi"
        readme = "./README_bitBattle.md"

        loader = UnstructuredMarkdownLoader(readme)
        # PyPDFLoader(pdf_url, extract_images=True)
        pages = loader.load()
        docs = split_docs(pages)

        vectordb = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory=persist_directory
        )
        vectordb.persist()
    else:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    if use_ollama:
        llm = Ollama(model="llama2")#"stablelm2")
    else:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    chain = load_qa_chain(llm, chain_type="stuff")
    return vectordb, chain


def do_query(vectordb, chain, query):
    matching_docs = vectordb.similarity_search(query, k=8)
    answer = chain.run(input_documents=matching_docs, question=query)
    return matching_docs, answer