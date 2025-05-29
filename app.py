import streamlit as st
import pandas as pd
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_api_key:
    st.error("Hugging Face API key not found.")
    st.stop()

# Initialize Streamlit app
st.set_page_config(page_title="Excel Chatbot", layout="centered")
# Show an image instead of title
col1, col2 = st.columns([0.1, 0.9], gap="small")
with col1:
    st.image("static/banner.png", width=50)
with col2:
    st.markdown("""
    <h2 style='margin: 0; padding: 0; line-height: 1.2;'>
    NECTBOT Ask anything about your Excel data:
    </h2>
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload an Excel file."}]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Function to process Excel file and create vector store
def process_excel(file):
    try:
        print("Saving uploaded file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(file.read())
            tmp_file_path = tmp.name
        print(f"File saved to: {tmp_file_path}")

        print("Reading Excel file...")
        df = pd.read_excel(tmp_file_path, skiprows=2)
        print(f"Shape: {df.shape}, Columns: {df.columns.tolist()}")

        print("Converting DataFrame to text...")
        text_data = []
        for _, row in df.iterrows():
            row_text = " ".join([str(val) for val in row if pd.notna(val)])
            text_data.append(row_text)
        print(f"Text data: {len(text_data)} rows")

        print("Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.create_documents(text_data)
        print(f"Split into: {len(texts)} chunks")

        print("Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"token": hf_api_key}
        )
        print("Embeddings initialized")

        print("Creating vector store...")
        vector_store = FAISS.from_documents(texts, embeddings)
        print("Vector store created")

        print("Initializing LLM...")
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            huggingfacehub_api_token=hf_api_key,
            temperature=0.7,
            max_new_tokens=512
        )
        print("LLM initialized")

        print("Initializing conversation memory...")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        print("Creating conversational chain...")
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
        print("Conversation chain created")

        os.unlink(tmp_file_path)
        print("Temporary file cleaned up")

        return vector_store, conversation
    except Exception as e:
        st.error(f"Error processing Excel file: {str(e)}")
        print(f"Error: {str(e)}")
        return None, None

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

# Process uploaded file
if uploaded_file and st.session_state.vector_store is None:
    with st.spinner("Processing Excel file..."):
        st.session_state.vector_store, st.session_state.conversation = process_excel(uploaded_file)
        if st.session_state.vector_store:
            st.session_state.messages = [{"role": "assistant", "content": "Excel file processed! Ask me anything about the data."}]
            st.success("File processed successfully!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the Excel data"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.conversation:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    print(f"Processing query: {prompt}")
                    response = st.session_state.conversation({"question": prompt})["answer"]
                    print("Query processed successfully")
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    print(f"Query error: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        with st.chat_message("assistant"):
            st.markdown("Please upload an Excel file first.")
            st.session_state.messages.append({"role": "assistant", "content": "Please upload an Excel file first."})