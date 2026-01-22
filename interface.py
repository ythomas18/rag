import streamlit as st
import os
import shutil
import time
import pandas as pd
import csv
from datetime import datetime
from rag_features import HybridRetriever
from auth import require_auth, logout, init_session_state

st.set_page_config(page_title="GreenPower RAG", page_icon="⚡", layout="wide")

# --- Authentication Check ---
init_session_state(st)

if not require_auth(st):
    st.stop()  # Stop here if not authenticated

# --- User is authenticated from this point ---
st.title("⚡ GreenPower Hybrid RAG")
st.markdown("Query your documents using Vector Search + Knowledge Graph.")

# --- Metrics Setup ---
METRICS_FILE = "metrics.csv"
if not os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "latency", "route", "query_length", "response_length", "qdrant_latency", "neo4j_latency"])

def log_metric(latency, route, query_len, response_len, qdrant_lat=0.0, neo4j_lat=0.0):
    with open(METRICS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), latency, route, query_len, response_len, qdrant_lat, neo4j_lat])

# --- RAG Engine ---
@st.cache_resource
def get_engine():
    return HybridRetriever()

try:
    rag = get_engine()
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    # User info and logout
    user = st.session_state.user
    st.markdown(f"👤 **{user['display_name']}** ({user['role']})")
    if st.button("🚪 Logout", use_container_width=True):
        logout(st)
        st.rerun()
    
    st.divider()
    st.header("📄 Document Ingestion")
    
    # Check LlamaParse Status
    try:
        from config import LLAMA_CLOUD_API_KEY
        if LLAMA_CLOUD_API_KEY:
             st.success("✨ Llama Cloud Agent (Parsing) Active")
        else:
             st.info("Using Standard Parser (PyPDF)")
    except ImportError:
        pass

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
    
    st.divider()
    
    # --- Web Scraping Section ---
    st.header("🌐 Web Scraping")
    st.caption("Scrape web pages to enrich your RAG")
    
    # URL input
    urls_input = st.text_area(
        "Enter URLs (one per line)",
        placeholder="https://example.com/page1\nhttps://example.com/page2",
        height=100
    )
    
    # Options
    col1, col2 = st.columns(2)
    with col1:
        follow_links = st.checkbox("Follow internal links", value=False)
    with col2:
        max_pages = st.number_input("Max pages", min_value=1, max_value=50, value=5)
    
    if st.button("🚀 Scrape & Ingest", type="primary"):
        if urls_input.strip():
            # Parse URLs
            urls = [url.strip() for url in urls_input.strip().split('\n') if url.strip()]
            
            if urls:
                with st.spinner(f"Scraping {len(urls)} URL(s)..."):
                    result = rag.ingest_web(
                        urls=urls,
                        follow_links=follow_links,
                        max_pages=int(max_pages)
                    )
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(f"✅ Scraped {result['pages_scraped']} pages")
                    st.json(result)
            else:
                st.warning("Please enter at least one valid URL")
        else:
            st.warning("Please enter at least one URL")

# --- Main Interface ---
tab1, tab2 = st.tabs(["💬 Chat", "📊 Metrics"])

with tab1:
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
                start_time = time.time()
                
                chunks, route, timings = rag.retrieve(prompt)
                response = rag.generate_answer(prompt, chunks, route)
                
                end_time = time.time()
                latency = end_time - start_time
                
                # Log metrics
                log_metric(latency, route, len(prompt), len(response), timings.get("qdrant", 0), timings.get("neo4j", 0))
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.expander("Debug Details"):
                    st.write(f"Route used: **{route}**")
                    st.write(f"Latency: **{latency:.4f}s**")
                    st.write("Context chunks:", chunks)

with tab2:
    st.header("Dashboard Metrics")
    
    if os.path.exists(METRICS_FILE):
        try:
            df = pd.read_csv(METRICS_FILE)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Top Level Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Requests", len(df))
                with col2:
                    avg_lat = df["latency"].mean()
                    st.metric("Avg Latency", f"{avg_lat:.2f} s")
                with col3:
                    req_per_min = len(df) / ((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60) if len(df) > 1 else 0
                    st.metric("Requests/Min", f"{req_per_min:.2f} rpm")
                with col4:
                    success_rate = "100%" # Placeholder if we tracked errors
                    st.metric("Success Rate", success_rate)

                st.divider()

                # Charts
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.subheader("Latency over Time")
                    # Simple line chart
                    chart_data = df.set_index("timestamp")[["latency"]]
                    st.line_chart(chart_data)
                
                with col_chart2:
                    st.subheader("Requests Distribution by Route")
                    route_counts = df["route"].value_counts()
                    st.bar_chart(route_counts)

                st.divider()
                st.subheader("Performance Comparison (Latency by Route)")
                # Calculate average latency per route
                if not df.empty:
                    # Generic Route Comparison
                    latency_by_route = df.groupby("route")["latency"].mean()
                    st.bar_chart(latency_by_route)
                    
                    st.divider()
                    st.subheader("Granular Latency: Qdrant vs Neo4j")
                    st.info("Direct comparison of retrieval times (even within Hybrid requests)")
                    
                    # check if new columns exist (backward compatibility for old csv rows)
                    if "qdrant_latency" in df.columns and "neo4j_latency" in df.columns:
                        # Fill NaN with 0 for older logs
                        df["qdrant_latency"] = df["qdrant_latency"].fillna(0)
                        df["neo4j_latency"] = df["neo4j_latency"].fillna(0)
                        
                        avg_qdrant = df["qdrant_latency"].mean()
                        avg_neo4j = df["neo4j_latency"].mean()
                        
                        # Side by side metrics
                        m1, m2 = st.columns(2)
                        m1.metric("Avg Qdrant Retrieval", f"{avg_qdrant:.4f} s")
                        m2.metric("Avg Neo4j Retrieval", f"{avg_neo4j:.4f} s")
                        
                        # Bar chart comparison
                        comp_data = pd.DataFrame({
                            "Source": ["Qdrant", "Neo4j"],
                            "Avg Latency (s)": [avg_qdrant, avg_neo4j]
                        }).set_index("Source")
                        st.bar_chart(comp_data)
                    
                    # Optional: Textual comparison
                    cols = st.columns(len(latency_by_route))
                    for i, (route_name, avg_val) in enumerate(latency_by_route.items()):
                        cols[i].metric(f"{route_name.title()} Avg", f"{avg_val:.4f} s")

                # Advanced Layout for Analysis
                st.subheader("Detailed Analysis")
                
                # Latency vs Query Length Scatter (Do queries get slower if they are longer?)
                # Streamlit scatter needs a bit more work or using altair/plotly, stick to simple Vega-Lite via st.scatter_chart if available in newer versions, else altair.
                # using st.scatter_chart (available in recent streamlit)
                try:
                    st.scatter_chart(df, x="query_length", y="latency", color="route")
                except:
                    st.info("Scatter chart requires newer streamlit version.")

                with st.expander("View Raw Logs"):
                    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
            else:
                st.info("No data available yet. Make a request in the Chat tab!")
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
            # If CSV is corrupted
            if st.button("Reset Metrics File"):
                if os.path.exists(METRICS_FILE):
                    os.remove(METRICS_FILE)
                    st.experimental_rerun()
    else:
        st.warning("Metrics file not found. It will be created on the first request.")
