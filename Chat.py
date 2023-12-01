# Import necessary libraries
import os
import streamlit as st
from langchain.llms import LlamaCpp
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA

# used to create the memory
from langchain.memory import ConversationBufferMemory
memory_key = "history"

## Local LLM 
base_dir = "..\\llms\\llama2\\" 
llm_model = "llama-2-7b-chat.ggmlv3.q5_K_M.bin"
local_llm = base_dir+llm_model

## Vector Database 
chromadb = "db"
upload_folder = "upload"
os.makedirs(upload_folder, exist_ok=True)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=chromadb, embedding_function=embeddings)

def load_model():
    llm = LlamaCpp(
        model_path=local_llm,
        n_batch=1024,
        n_ctx=4096,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        # callback_manager=callback_manager,
        verbose=True,
    )
    return llm

# Memory is added to an agent to take it from being stateless to stateful i.e. it remembers previous interactions. 
# This means you can ask follow-up questions, making interaction with the agent much more efficient and feel more natural.
# memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)


# Streamlit UI
st.image('images\idesign.png', width=400)
st.header("AI Assistant")
query = st.text_input("Enter your query")


if st.button("Retrieve"):
    llm = load_model()
    qa = RetrievalQA.from_llm(llm, retriever=vectorstore.as_retriever())
    if query:
        results = qa(query)
        print(len(results))
        answer = results['result']
        print(answer)

        # Display the first part of the answer
        st.write(answer[:500])  # Display the first 500 characters, adjust as needed

        # Check if the answer exceeds the displayed limit
        if len(answer) > 500:
            # Display "Continue" button
            if st.button("Continue"):
                # Display the rest of the answer
                st.write(answer[500:])  # Display the rest of the answer

