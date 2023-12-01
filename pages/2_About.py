import streamlit as st

st.set_page_config(
    page_title="about",
    page_icon="👋",
)

st.write("# Welcome to the AI Assistent for Solution Architects! 👋")

st.markdown(
    """
    iDesign AI is an open-source app framework built specifically for the 
    Solution Architects.
    It provides an AI assistent on 's IT architecture.
    If you want to train the iDesign AI on specific data, you can upload a document or a confluence page (TBD). 
    This document is embedded (e.g. transformed from text to data and stored in a vector database)
    and can be used by the AI assistent with retrieval augmented generation.
    \n
    **👈 Select an option from the sidebar** to see what iDesign AI can do!
    
    ### Want to learn more?
    - Check out [huggingface.co](https://huggingface.co/)
    - Ask a question in our [community
        forums](https://www.reddit.com/r/LocalLLaMA/)
    - or check out the code at [github](https://github.com/PuxAI)
"""
)
