import torch
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from ai_act_validator import check_ai_act_compliance
from utils import load_or_create_vectorstore, WeightedRetriever
# from extra_streamlit_components import BetterCookieManager


# 🩹 Torch patch for Streamlit compatibility
if hasattr(torch, '__path__') and hasattr(torch.__path__, '_path'):
    try:
        del torch.__path__
    except Exception:
        pass

# Streamlit setup
st.set_page_config(page_title="AI Act Chatbot", layout="wide")
st.title("🛡️ EU AI Act Chatbot")
load_dotenv()

# 🍪 Cookie-based user tracking

# 🔐 UUID-based session token
if "user_token" not in st.session_state:
    st.session_state.user_token = str(uuid.uuid4())

user_token = st.session_state.user_token

# 🔄 Usage counter per session
if "usage_counter" not in st.session_state:
    st.session_state.usage_counter = {}

if user_token not in st.session_state.usage_counter:
    st.session_state.usage_counter[user_token] = 0

remaining = 3 - st.session_state.usage_counter[user_token]
st.info(f"🔁 You have **{remaining}** prompts left in this session.")

# 🔍 Load QA chain
@st.cache_resource
def setup_chain():
    vs = load_or_create_vectorstore()
    retriever = WeightedRetriever(
        vectorstore=vs,
        source_weights={"AI_act_EU_full_text.pdf": 1.5},
        k=10
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

qa_chain = setup_chain()

# 🧭 Interface with two tabs
tab1, tab2 = st.tabs(["📘 Ask a Question", "🔍 Validate AI System"])

with tab1:
    st.subheader("Ask a question from the AI Act")
    user_question = st.text_area("📝 Enter your question about the AI Act (Not legal advice):", key="question_input")

    if st.button("Submit Question", key="submit_question"):
        if st.session_state.usage_counter[user_token] >= 3:
            st.warning("❌ Prompt limit reached.")
        elif not user_question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                result = qa_chain.invoke(user_question)
                st.success("✅ Answer:")
                st.markdown(result["result"])
            st.session_state.usage_counter[user_token] += 1

    # ✅ Right-aligned reset button
    spacer, reset_col = st.columns([5, 1])
    with reset_col:
        if st.button("🔄 Reset Prompt Count", key="reset_prompt_tab1"):
            st.session_state.usage_counter[user_token] = 0
            st.success("Prompt count reset. You can now ask 3 new questions.")


with tab2:
    st.subheader("Validate an AI System Description")
    system_description = st.text_area("🧠 Describe your AI system (Not legal advice):", key="system_input")
cols = st.columns([4, 1])
with cols[0]:
    if st.button("Submit Question", key="submit_question"):
        if st.session_state.usage_counter[user_token] >= 3:
            st.warning("❌ Prompt limit reached.")
        elif not user_question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                result = qa_chain.invoke(user_question)
                st.success("✅ Answer:")
                st.markdown(result["result"])
            st.session_state.usage_counter[user_token] += 1

with cols[1]:
    if st.button("🔄 Reset Prompt Count"):
        st.session_state.usage_counter[user_token] = 0
        st.success("Prompt count reset. You can now ask 3 new questions.")

# ⚖️ Disclaimer + GIF
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown("""
    > ⚠️ **Disclaimer**: This tool is for informational purposes only and does **not** constitute legal advice.  
    > 🔐 **Security Note**: Only self-generated vectorstores are supported. Do not upload untrusted files.  
    > 📊 **Privacy**: No personal data is stored. Inputs are processed temporarily in-session.  
    > 🍪 **Tracking**: We use an anonymous cookie token to enforce prompt limits.
    """)

with col2:
    st.markdown(
        '<img src="https://media.giphy.com/media/l4Jz3a8jO92crUlWM/giphy.gif" width="50%">',
        unsafe_allow_html=True
    )
