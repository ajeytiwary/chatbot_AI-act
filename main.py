import torch

# 🩹 Patch to avoid Streamlit crash with torch in some environments
if hasattr(torch, '__path__') and hasattr(torch.__path__, '_path'):
    try:
        del torch.__path__
    except Exception:
        pass

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from ai_act_validator import check_ai_act_compliance
from utils import load_documents, create_vectorstore, load_or_create_vectorstore, WeightedRetriever
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os

st.set_page_config(page_title="AI Act Chatbot", layout="wide")
st.title("🛡️ EU AI Act Chatbot")

load_dotenv()

# ✅ Prompt limit
if "prompt_count" not in st.session_state:
    st.session_state.prompt_count = 0
remaining = 3 - st.session_state.prompt_count
st.info(f"You have {remaining} prompts left.")

# ✅ Load vectorstore + QA chain
@st.cache_resource
def setup_chain():
    vs = load_or_create_vectorstore()
    source_weights = {
        "AI_act_EU_full_text.pdf": 1.5,
    }
    retriever = WeightedRetriever(vectorstore=vs, source_weights=source_weights, k=10)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

qa_chain = setup_chain()

# ✅ Two tabs: Ask and Validate
tab1, tab2 = st.tabs(["📘 Ask a Question", "🔍 Validate AI System"])

with tab1:
    st.subheader("Ask a question from the AI Act")
    user_question = st.text_area("📝 Enter your question about AI act (Not a LEGAL advice):", key="question_input")

    if st.button("Submit Question", key="submit_question"):
        if st.session_state.prompt_count >= 3:
            st.warning("❌ Prompt limit reached for this session.")
        elif not user_question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                result = qa_chain.invoke(user_question)
                st.success("✅ Answer:")
                st.markdown(result["result"])
                with st.expander("📚 Sources"):
                    for doc in result["source_documents"]:
                        st.markdown(f"- **{doc.metadata.get('source', 'Unknown')}** | Page: {doc.metadata.get('page', 'n/a')}")
            st.session_state.prompt_count += 1

with tab2:
    st.subheader("Validate an AI System Description")
    system_description = st.text_area("🧠 Describe your AI system (Not a LEGAL advice):", key="system_input")

    if st.button("Validate Compliance", key="submit_validation"):
        if st.session_state.prompt_count >= 3:
            st.warning("❌ Prompt limit reached for this session.")
        elif not system_description.strip():
            st.error("Please enter your system description.")
        else:
            with st.spinner("Evaluating..."):
                result = check_ai_act_compliance(system_description)
                st.success("✅ Compliance Assessment:")
                st.markdown(result)
            st.session_state.prompt_count += 1

# ✅ Reset prompt count
if st.button("🔄 Reset"):
    st.session_state.prompt_count = 0
    st.success("You can now ask 3 new questions.")
