
import torch

# 🩹 Patch to avoid Streamlit crash with torch in some environments
if hasattr(torch, '__path__') and hasattr(torch.__path__, '_path'):
    try:
        del torch.__path__
    except Exception:
        pass

import google.generativeai as genai

import sys
import streamlit as st
from dotenv import load_dotenv
from ai_act_validator import check_ai_act_compliance
from utils import load_documents, create_vectorstore, load_or_create_vectorstore, WeightedRetriever
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os

st.set_page_config(page_title="AI Act Chatbot", layout="wide")

# st.write("Torch version:", torch.__version__)
# st.write("Torch path:", torch.__file__)
# st.write("Dynamo loaded ✅")

# if hasattr(torch, '__path__'):
#     del torch.__path__

load_dotenv()

st.title("🛡️ EU AI Act Chatbot")

# Mode selector
mode = st.radio("Choose a task:", ["Ask a question from the AI Act", "Validate an AI system description"])

# Load docs and chain
@st.cache_resource
def setup_chain():
    vs = load_or_create_vectorstore()
    # retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 8})
    source_weights = {
        "AI_act_EU_full_text.pdf": 1.5,   # prioritize
        # "AI_Act_FAQ.pdf": 1.0,
        # "AI_Act_Annex.pdf": 1.2,
        # "AI_Act_Recitals.pdf": 1.4
    }

    retriever = WeightedRetriever(
        vectorstore=vs,
        source_weights=source_weights,
        k=10
    )

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

qa_chain = setup_chain()

# Track prompts
if "prompt_count" not in st.session_state:
    st.session_state.prompt_count = 0

remaining = 3 - st.session_state.prompt_count
st.info(f"You have {remaining} prompts left.")

user_input = st.text_area("📝 Enter your input:")

if st.button("Submit"):

    if st.session_state.prompt_count >= 3:
        st.warning("❌ Prompt limit reached for this session.")
    elif not user_input.strip():
        st.error("Please enter something.")
    else:
        with st.spinner("Thinking..."):
            if mode == "Ask a question from the AI Act":
                result = qa_chain.invoke(user_input)
                st.success("✅ Answer:")
                st.markdown(result["result"])
                with st.expander("Sources"):
                    for doc in result["source_documents"]:
                        st.write(f"- {doc.metadata.get('source', 'Unknown')}")
            else:
                result = check_ai_act_compliance(user_input)
                st.success("✅ Compliance Assessment:")
                st.markdown(result)

        st.session_state.prompt_count += 1

# Optional reset
if st.button("Reset"):
    st.session_state.prompt_count = 0
    st.success("You can now ask 3 new questions.")
