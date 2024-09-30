

# app.py
import streamlit as st
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import T5Tokenizer, T5ForConditionalGeneration
import bs4

# Streamlit UI Design
st.set_page_config(page_title="RAG-based PDF and Website Summarizor and Reader", page_icon=":books:", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>RAG-based PDF and Website Summarizor & Reader</h1>", unsafe_allow_html=True)

# Side Bar
st.sidebar.image("https://via.placeholder.com/150", use_column_width=True)  # Placeholder for a logo
st.sidebar.markdown("<h3 style='color:#4B8BBE;'>Upload Your Document or Enter a Website</h3>", unsafe_allow_html=True)

# Input section for document uploads and website input
uploaded_file = st.sidebar.file_uploader("Upload a PDF or Text file", type=['pdf', 'txt'])
web_url = st.sidebar.text_input("Or Enter Website URL")

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
generation_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store setup
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding_model,
    persist_directory="./chroma_langchain_db"
)

# Functions to load documents
def load_documents():
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            loader = TextLoader(uploaded_file)
        elif uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(uploaded_file)
        return loader.load()
    elif web_url:
        loader = WebBaseLoader(
            web_paths=(web_url,),
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-title", "post-content", "post-header")))
        )
        return loader.load()
    return []

# Split and vectorize document
def split_and_embed(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "]
    )
    doc_chunks = text_splitter.split_documents(documents)
    return doc_chunks

# Document retrieval
def retrieve_documents(query, n_results=5):
    query_embedding = embedding_model.embed_query(query)
    results = vector_store.similarity_search_by_vector(query_embedding, k=n_results)
    return [result.page_content for result in results]

# Clean documents
def clean_document(doc):
    return doc.replace("<pad>", "").replace("<EOS>", "").strip()

# Generate RAG response
def generate_rag_response(query):
    documents = load_documents()
    if documents:
        doc_chunks = split_and_embed(documents)
        relevant_docs = retrieve_documents(query)
        cleaned_docs = [clean_document(doc) for doc in relevant_docs if len(doc.strip()) > 0]
        context = ' '.join(cleaned_docs[:3])
        prompt = f"Summarize the following context:\n{context}\n\nQuery: {query}"
        inputs = tokenizer(prompt, return_tensors='pt')
        outputs = generation_model.generate(inputs.input_ids, max_length=250, num_return_sequences=1)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    return "No documents or website content to process."

# Display the retrieved response
st.markdown("<h3 style='color:#4B8BBE;'>Query Your Documents:</h3>", unsafe_allow_html=True)
user_query = st.text_input("Enter your query here")

if st.button("Generate Response"):
    if uploaded_file or web_url:
        response = generate_rag_response(user_query)
        st.markdown(f"<p style='color:#333; padding: 15px; background-color: #f0f4f8;'>{response}</p>", unsafe_allow_html=True)
    else:
        st.warning("Please upload a document or enter a website URL.")

# Footer
st.sidebar.markdown("<p style='text-align:center;'>Powered by LangChain, HuggingFace & Streamlit</p>", unsafe_allow_html=True)
