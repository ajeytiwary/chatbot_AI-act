🕵️ AI Act Compliance Chatbot

An interactive chatbot built with Streamlit, LangChain, FAISS, and Gemini (Google Generative AI), designed to help users:

🔹 Ask questions about the EU AI Act from uploaded legal documents (e.g. full text, FAQs, annexes)

🧑‍🎓 Validate if their AI system description is compliant with the AI Act

📄 Cite exact articles, pages, and sources from law text

⚖️ Provide legally structured responses and compliance assessments

⚠️ Disclaimer: This tool is for informational purposes only and does not constitute legal advice. For legal interpretation or decisions, consult a qualified legal professional.

🔐 Security Note: This application uses a local FAISS vectorstore which is deserialized using Python's pickle module. Only trusted and self-generated vectorstores are supported. Never load files from unknown sources.

📊 Privacy Statement: User inputs are processed in-session and are not stored or shared. No personal data is retained.

🚫 Rate Limiting: The chatbot limits users to 3 prompts per session using cookie-based UUID tracking to ensure fair usage and manage backend cost.

💸 API Cost Monitoring: Gemini API calls are usage-based and metered. Keep API keys secure and monitor usage quotas to avoid unexpected charges.

🍪 Cookie Tracking: Each user is assigned a pseudonymous ID stored via a browser cookie (UUID). This enables lightweight, privacy-respecting user tracking for prompt limit enforcement.

🌐 Features

🏰 AI Act QA Chat (RAG)

Uses LangChain's RetrievalQA to query a vectorstore of AI Act documents

Retrieves and ranks document chunks using custom WeightedRetriever

Dynamically ranks more important sources (e.g. prioritizing AI_act_EU_full_text.pdf)

🔒 Compliance Validator

Based on Gemini's gemini-pro or gemini-2.0-flash

Accepts a user-provided AI system description

Returns classification, compliance, article reference, and legal explanation

🏆 Smart Vectorstore

Uses FAISS for local and fast semantic search

Saves vectorstore to disk with metadata (page, filename)

Uses PyMuPDFLoader to preserve structure and improve accuracy

📁 Folder Structure

chatbot_AI_act/
├── main.py                  # Streamlit app UI
├── utils.py                 # Document loading, embedding, custom retriever
├── ai_act_validator.py      # Prompt + Gemini compliance validator
├── ai_act/                  # Folder containing 6 PDF files
├── vectorstore/             # FAISS index stored locally
├── .env                     # API key and config flags
├── requirements.txt         # Python dependencies

📊 How It Works

RetrievalQA Chain (main.py)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

Custom Retriever With Source Weighting (utils.py)
````
class WeightedRetriever(BaseRetriever):
    vectorstore: Any
    source_weights: Dict[str, float] = Field(default_factory=dict)
    k: int = 8

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.vectorstore.similarity_search_with_score(query, k=self.k)
        reranked = [(doc, score / self.source_weights.get(doc.metadata.get("source", "unknown"), 1.0)) for doc, score in results]
        reranked = sorted(reranked, key=lambda x: x[1])
        return [doc for doc, _ in reranked]
        
````       
Compliance Assessment Prompt (ai_act_validator.py)

**Classification:** [Prohibited | High-Risk | Limited Risk | Minimal Risk]  
**Compliance Status:** [Compliant | Non-Compliant | Not Enough Information]  
**Relevant References:** [Mention Article, Title, Annex numbers]  
**Explanation:** [Short legal justification from the AI Act]

🚀 How to Run

1. Clone the project

git clone https://github.com/your-username/ai-act-chatbot.git
cd ai-act-chatbot

2. Install dependencies

pip install -r requirements.txt

3. Set environment variables

Create a .env file:

GOOGLE_API_KEY=your_gemini_api_key
USE_LOCAL_EMBEDDINGS=true

Run
``streamlit run main.py``

📑 Requirements

Python 3.11+

LangChain

FAISS

Google Generative AI SDK

Streamlit

Hugging Face Embeddings

🙏 Credits

Built by @ajeytiwary using LangChain, Google Gemini, and EU Legal PDFs. Inspired by best practices in retrieval-augmented legal document Q&A systems.

✨ Coming Soon

Upload your own compliance doc to check against the AI Act

Export response & sources as Markdown

Conversation memory for follow-ups