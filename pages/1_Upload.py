# Import necessary libraries
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings

from langchain.llms import LlamaCpp
from langchain.vectorstores import Chroma

# used to create the memory
from langchain.memory import ConversationBufferMemory
memory_key = "history"

import os


# base_dir = "C:\\Users\\plampe\\Downloads\\AI\\llms\\llama2\\" 
# llm_model = "llama-2-7b-chat.ggmlv3.q5_K_M.bin"
# local_llm = base_dir+llm_model


# Directories
chromadb = "db"
upload_folder = "upload"
os.makedirs(upload_folder, exist_ok=True)


## Certification hack to bypass SSL verification errors to download new models
# import certifi
# certifi.where()
# "c:\\Users\\plampe\\AppData\\Roaming\\Python\\Python310\\site-packages\\certifi\\cacert.pem"

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')


def upload_documents():
    uploaded_file = st.file_uploader("Upload new documents for embedding", type=["pdf", "txt", "docx"])
    if uploaded_file is not None:
        # Get the file name
        filename = os.path.join(upload_folder, uploaded_file.name)
        with open(filename, 'wb') as f:
            f.write(uploaded_file.read())
        return filename
    return None

uploaded_document = upload_documents()

if uploaded_document:
    # Load documents
    
    def is_pdf(file_path):
        # Get the file extension
        file_extension = os.path.splitext(file_path)[-1].lower()
        # Check if the file extension is '.pdf'
        return file_extension == '.pdf'

    file_path = uploaded_document

    if is_pdf(file_path):
        print(f'{file_path} is a PDF file.')
        documents = PyPDFLoader(file_path=uploaded_document)
        # loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
        text_chunks = documents.load_and_split()
    else:
        print(f'{file_path} is not a PDF file.')
        documents = TextLoader(uploaded_document).load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        text_chunks = text_splitter.split_documents(documents)
    # Split text into chunks


    # text_chunks is a dictionary of:
    #   documents
    #       page_content
    #       metadata
    #           source

    # for document in text_chunks:
    #     page_content = document.page_content
    #     metadata = document.metadata
    # relationships = {}    
    # for i, document in enumerate(text_chunks):
    #     page_content = document.page_content
    #     metadata = document.metadata
        
    #     # Generate embeddings for page_content
    #     page_content_embedding = model.encode(page_content)
        
    #     # Store the relationship using a unique identifier (e.g., document ID)
    #     relationships[i] = {
    #         'page_content': page_content,
    #         'page_content_embedding': page_content_embedding,
    #         'metadata': metadata
    #     }
    #     print(relationships)
        
    #     for rs in relationships: 
    #         print("trying to store some relationships in the vectordb")
    #         print(rs)
    #         # vectorstore = Chroma.from_documents(documents=rs, persist_directory=chromadb)

    # Embed text chunks
    # embed_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    # embeddings = model.encode(text_chunks)
    # print(embeddings)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory=chromadb)
    # vectorstore = Chroma.from_documents(documents=text_chunks, embedding=LlamaCppEmbeddings(model_path=local_llm), persist_directory=chromadb)
    # embeddings = model.encode(sentences=text_chunks)
    # embed = model.
    st.write("Embeddings stored in the vector database.")



## Show contents of the Vector Database 
client = Chroma(persist_directory=chromadb)
dblist = client.get()
embedded_docs = sources = [item['source'] for item in dblist['metadatas']]

st.write("uploaded documents:")
st.write(embedded_docs)