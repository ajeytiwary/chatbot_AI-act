import os
import streamlit as st
from pydantic import Field
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema.retriever import BaseRetriever
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import Document
from typing import List, Dict, Any
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

VECTORSTORE_PATH = "vectorstore/ai_act_faiss"
# Load environment variables
load_dotenv()

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}


class WeightedRetriever(BaseRetriever):
    vectorstore: any
    source_weights: Dict[str, float] = Field(default_factory=dict)
    k: int = 8

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.vectorstore.similarity_search_with_score(query, k=self.k)

        reranked = []
        for doc, score in results:
            source = doc.metadata.get("source", "unknown")
            weight = self.source_weights.get(source, 1.0)
            adjusted_score = score / weight  # lower = better
            reranked.append((doc, adjusted_score))

        reranked = sorted(reranked, key=lambda x: x[1])
        return [doc for doc, _ in reranked]
    
# Embedding switch
def get_embeddings():
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
    
    if use_local:
        # from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    

@st.cache_data
def load_documents():
    all_docs = []
    for file in os.listdir("ai_act"):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join("ai_act", file))
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs


def create_vectorstore(documents):
    # splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " "]
    )
    
    # chunks = splitter.split_documents(docs)
    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

def load_or_create_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        embeddings = get_embeddings()
        return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        
    else:
        docs = load_documents()
        return create_vectorstore(docs)