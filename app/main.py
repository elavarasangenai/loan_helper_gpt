
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings

# AI Helper for clause explanation and risk highlighting
def explain_clauses(text, llm):
    prompt = f"Explain the key clauses, obligations, and highlight risky terms in the following loan document. Use alert colors for risky terms.\n\n{text}"
    return llm(prompt)

def answer_question(question, qa_chain):
    answer = qa_chain.run(question)
    if "not related" in answer.lower() or "I don't know" in answer:
        return "I don't know"
    return answer

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FIAS_DB_URL = os.getenv("FIAS_DB_URL")

# Initialize LLM
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4")


# Helper function to process PDF and store vectors
def process_pdf(pdf_file):
    # Handle Streamlit UploadedFile by saving to a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# RAG QA Chain
def get_qa_chain(vectorstore):
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# UI
st.title("Loan Helper GPT")
st.write("Upload your loan document (PDF) to analyze key clauses, obligations, and risky terms.")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    if st.button("Save"):
        with st.spinner("Processing document..."):
            try:
                vectorstore = process_pdf(uploaded_file)
                st.success("Document saved and processed successfully!")
                qa_chain = get_qa_chain(vectorstore)
                # Summarize key clauses, obligations, risky terms
                summary_prompt = "Summarize the key clauses, obligations, and highlight risky terms in the document. Use alert colors for risky terms."
                summary = qa_chain.run(summary_prompt)
                st.markdown("### Document Summary")
                st.markdown(summary, unsafe_allow_html=True)
                st.session_state["qa_chain"] = qa_chain
            except Exception as e:
                st.error(f"Error processing document: {e}")

if "qa_chain" in st.session_state:
    st.markdown("---")
    st.markdown("### Ask a Question about the Document")
    question = st.text_input("Enter your question")
    if st.button("Submit") and question:
        with st.spinner("Getting answer..."):
            try:
                answer = answer_question(question, st.session_state["qa_chain"])
                # If not relevant, reply "I don't know"
                if answer == "I don't know":
                    st.warning("I don't know")
                else:
                    st.markdown("**Answer:**")
                    st.markdown(f"- {answer}")
            except Exception as e:
                st.error(f"Error answering question: {e}")
