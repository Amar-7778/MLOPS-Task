import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load env variables if present
load_dotenv()

# App configuration
st.set_page_config(
    page_title="GitHub Repository Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Inject Custom CSS for premium feel and modern badges
st.markdown("""
<style>
    /* Styling for Gemini and Groq Badges */
    .provider-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        color: white !important;
        margin-bottom: 8px;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    .gemini-badge {
        background: linear-gradient(135deg, #4285F4, #9B51E0);
        box-shadow: 0 0 12px rgba(155, 81, 224, 0.4);
    }
    .groq-badge {
        background: linear-gradient(135deg, #FF9900, #FF5500);
        box-shadow: 0 0 12px rgba(255, 85, 0, 0.4);
    }
    .error-badge {
        background-color: #DC3545;
    }
    
    /* Metrics display */
    .metric-card {
        background-color: #1E222B;
        border: 1px solid #2D3139;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00B4D8;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #8C92A0;
        text-transform: uppercase;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "repo_id" not in st.session_state:
    st.session_state.repo_id = None
if "repo_name" not in st.session_state:
    st.session_state.repo_name = None
if "owner" not in st.session_state:
    st.session_state.owner = None
if "repo_stats" not in st.session_state:
    st.session_state.repo_stats = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "reports" not in st.session_state:
    st.session_state.reports = {"tech_stack": None, "data_flow": None, "system_design": None}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def get_provider_badge(provider: str) -> str:
    """Returns HTML for provider badges."""
    p_clean = provider.strip().lower()
    if "gemini" in p_clean:
        return '<span class="provider-badge gemini-badge">♊ Gemini 2.5 Flash</span>'
    elif "groq" in p_clean:
        return '<span class="provider-badge groq-badge">🔥 Groq Fallback (Llama-3.3)</span>'
    else:
        return f'<span class="provider-badge error-badge">{provider}</span>'


def show_sources(sources):
    """Renders unique source files used for context."""
    if not sources:
        return
    with st.expander(f"🔍 Context Sources ({len(sources)} files)"):
        for idx, src in enumerate(sources):
            file_path = src.get("file_path", "")
            language = src.get("language", "")
            st.markdown(f"**{idx + 1}.** `{file_path}` *(language: {language})*")


def run_query_api(query: str, mode: str, chat_history=None) -> dict:
    """Helper to query the FastAPI backend."""
    payload = {
        "repo_id": st.session_state.repo_id,
        "query": query,
        "mode": mode,
        "chat_history": chat_history or []
    }
    try:
        res = requests.post(f"{BACKEND_URL}/api/query", json=payload)
        if res.status_code == 200:
            return res.json()
        else:
            return {
                "answer": f"Error: Backend API returned status code {res.status_code}. Detail: {res.text}",
                "sources": [],
                "provider": "Error"
            }
    except Exception as e:
        return {
            "answer": f"Failed to connect to backend server: {e}. Make sure FastAPI is running on {BACKEND_URL}.",
            "sources": [],
            "provider": "Error"
        }


# ================= SIDEBAR =================
with st.sidebar:
    st.title("⚙️ Controls")
    
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repository",
        help="Paste the link to any public GitHub repository."
    )
    
    force_reingest = st.checkbox(
        "Force Re-ingest", 
        value=False,
        help="Check this if you want to bypass local ChromaDB cache and force redownloading/reindexing."
    )
    
    analyze_btn = st.button("🚀 Analyze Repository", use_container_width=True)
    
    if analyze_btn:
        if not repo_url:
            st.error("Please enter a valid GitHub URL.")
        else:
            with st.spinner("Indexing repository... This can take 1-2 minutes for larger codebases."):
                try:
                    payload = {"url": repo_url, "force_reingest": force_reingest}
                    res = requests.post(f"{BACKEND_URL}/api/ingest", json=payload)
                    
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.repo_id = data["repo_id"]
                        st.session_state.repo_name = data["repo_name"]
                        st.session_state.owner = data["owner"]
                        st.session_state.summary = data["summary"]
                        st.session_state.repo_stats = {
                            "total_files": data["total_files"],
                            "total_chunks": data["total_chunks"],
                            "languages": data["languages"]
                        }
                        # Clear previous reports and chats for new repo
                        st.session_state.reports = {"tech_stack": None, "data_flow": None, "system_design": None}
                        st.session_state.chat_history = []
                        st.success("Indexing complete!")
                    else:
                        st.error(f"Ingestion failed: {res.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Could not connect to FastAPI backend: {e}. Check if uvicorn is running.")
                    
    # Display Stats if loaded
    if st.session_state.repo_id:
        st.markdown("---")
        st.subheader("📊 Repository Stats")
        st.markdown(f"**Indexed Repo:** `{st.session_state.repo_id}`")
        
        # Grid of stats
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{st.session_state.repo_stats['total_files']}</div>
                <div class="metric-lbl">Files Scanned</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{st.session_state.repo_stats['total_chunks']}</div>
                <div class="metric-lbl">Vector Chunks</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("**Primary Languages:**")
        langs = st.session_state.repo_stats['languages']
        if langs:
            st.write(", ".join([f"`{l}`" for l in langs]))
        else:
            st.write("*None detected*")


# ================= MAIN AREA =================
st.title("🤖 GitHub Repository Explainer")
st.markdown("Understand tech stack, execution flow, architecture, and chat with repository source code using local RAG.")

if not st.session_state.repo_id:
    # Landing page layout if no repo loaded
    st.info("👈 Please enter a public GitHub URL and click **Analyze Repository** in the sidebar to get started.")
    
    st.markdown("""
    ### Features:
    - **Language-Aware Splitting**: Extracts code components from 15+ popular file formats (`.py`, `.js`, `.ts`, `.go`, `.java`, etc.).
    - **Offline Embeddings**: Uses SentenceTransformers locally to create vector databases without sharing code with OpenAI.
    - **RAG Architecture Analysis**: Automatically prepares detailed summaries of Tech Stack, Data Flow, and high-level System Design.
    - **Dual LLM Cooldown Switch**: Works primarily with Gemini 2.5 Flash. If hitting quota/token limits, it falls back to Groq Llama 3.3 automatically for 60 seconds.
    """)
else:
    # Summary Card Banner
    with st.container(border=True):
        st.markdown(st.session_state.summary)
        
    # Tab Layout
    tab_tech, tab_flow, tab_design, tab_chat = st.tabs([
        "💻 Tech Stack", 
        "🔄 Data Flow", 
        "🏛️ System Design", 
        "💬 Chat Q&A"
    ])
    
    # 1. Tech Stack Tab
    with tab_tech:
        if not st.session_state.reports["tech_stack"]:
            with st.spinner("Analyzing Tech Stack... compiling report..."):
                res = run_query_api("Analyze Tech Stack", "tech_stack")
                st.session_state.reports["tech_stack"] = res
                
        report = st.session_state.reports["tech_stack"]
        if report:
            badge_html = get_provider_badge(report.get("provider", "Gemini"))
            st.markdown(f"<div style='margin-bottom: 10px;'><b>Analysis Provider:</b> {badge_html}</div>", unsafe_allow_html=True)
            st.markdown(report["answer"])
            show_sources(report["sources"])
            
    # 2. Data Flow Tab
    with tab_flow:
        if not st.session_state.reports["data_flow"]:
            with st.spinner("Tracing Data Flow... compiling report..."):
                res = run_query_api("Analyze Data Flow", "data_flow")
                st.session_state.reports["data_flow"] = res
                
        report = st.session_state.reports["data_flow"]
        if report:
            badge_html = get_provider_badge(report.get("provider", "Gemini"))
            st.markdown(f"<div style='margin-bottom: 10px;'><b>Analysis Provider:</b> {badge_html}</div>", unsafe_allow_html=True)
            st.markdown(report["answer"])
            show_sources(report["sources"])
            
    # 3. System Design Tab
    with tab_design:
        if not st.session_state.reports["system_design"]:
            with st.spinner("Evaluating System Design... compiling report..."):
                res = run_query_api("Analyze System Design", "system_design")
                st.session_state.reports["system_design"] = res
                
        report = st.session_state.reports["system_design"]
        if report:
            badge_html = get_provider_badge(report.get("provider", "Gemini"))
            st.markdown(f"<div style='margin-bottom: 10px;'><b>Analysis Provider:</b> {badge_html}</div>", unsafe_allow_html=True)
            st.markdown(report["answer"])
            show_sources(report["sources"])

    # 4. Chat Q&A Tab
    with tab_chat:
        st.subheader("💬 Chat with Repository Code")
        
        # Display message history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    badge_html = get_provider_badge(msg.get("provider", "Gemini"))
                    st.markdown(f"<div style='margin-bottom: 5px;'>{badge_html}</div>", unsafe_allow_html=True)
                st.markdown(msg["content"])
                if msg.get("sources"):
                    show_sources(msg["sources"])
                    
        # User input box
        if prompt := st.chat_input("Ask about functions, API endpoints, configurations..."):
            # Render user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Render assistant spinner & call backend
            with st.chat_message("assistant"):
                with st.spinner("Searching repository code & thinking..."):
                    res = run_query_api(
                        query=prompt, 
                        mode="chat", 
                        chat_history=st.session_state.chat_history[:-1]
                    )
                
                badge_html = get_provider_badge(res["provider"])
                st.markdown(f"<div style='margin-bottom: 5px;'>{badge_html}</div>", unsafe_allow_html=True)
                st.markdown(res["answer"])
                show_sources(res["sources"])
                
                # Append assistant response to state
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": res["answer"],
                    "provider": res["provider"],
                    "sources": res["sources"]
                })
