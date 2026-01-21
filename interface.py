import streamlit as st
import os
import shutil
from rag_features import HybridRetriever

st.set_page_config(page_title="GreenPower RAG", page_icon="⚡")

st.title("⚡ GreenPower Hybrid RAG")
st.markdown("Query your documents using Vector Search + Knowledge Graph.")

# Initialize RAG engine
@st.cache_resource
def get_engine():
    return HybridRetriever()

try:
    rag = get_engine()
    st.success("System initialized successfully!")
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

# Sidebar for file upload
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=['pdf', 'txt', 'json', 'csv'])
    
    if st.button("Ingest Documents") and uploaded_files:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        paths = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            path = os.path.join(temp_dir, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            paths.append(path)
        
        status_text.text("Ingesting documents...")
        result = rag.ingest(paths)
        progress_bar.progress(100)
        
        st.json(result)
        
        # Cleanup
        shutil.rmtree(temp_dir)

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about GreenPower products..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chunks, route = rag.retrieve(prompt)
            response = rag.generate_answer(prompt, chunks, route)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.expander("Debug Details"):
                st.write(f"Route used: **{route}**")
                st.write("Context chunks:", chunks)
