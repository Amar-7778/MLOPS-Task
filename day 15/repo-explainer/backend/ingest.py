import os
import re
import json
import base64
import logging
from typing import List, Dict, Any, Tuple, Optional
from github import Github
from dotenv import load_dotenv
import tiktoken

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

# Load env
load_dotenv()

logger = logging.getLogger(__name__)

# Constants
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
SUMMARIES_FILE = os.path.join(CHROMA_DIR, "repo_summaries.json")

# Map of file extensions to LangChain Language enums
EXTENSION_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".cpp": Language.CPP,
    ".c": Language.CPP,
    ".cs": Language.CSHARP,
    ".rb": Language.RUBY,
    ".md": Language.MARKDOWN,
    # Config files (.json, .toml, .yaml, .yml) will use standard text splitters
}

# Embedding model singleton
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        logger.info("Loading SentenceTransformers embedding model (all-MiniLM-L6-v2)...")
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings

def get_vector_store():
    embeddings = get_embeddings()
    # Ensure directory exists
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )


# Token length function using tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")

def get_token_length(text: str) -> int:
    return len(tokenizer.encode(text))


def parse_github_url(url: str) -> Tuple[str, str]:
    """
    Parses GitHub URL to extract owner and repository name.
    Handles formats like:
      - https://github.com/owner/repo
      - https://github.com/owner/repo.git
      - github.com/owner/repo
    """
    clean_url = url.strip()
    if not clean_url.startswith("http"):
        clean_url = "https://" + clean_url
        
    # Match owner and repo
    match = re.search(r"github\.com/([^/]+)/([^/.]+)", clean_url)
    if not match:
        raise ValueError("Invalid GitHub URL. Must be of the form github.com/owner/repository")
        
    owner = match.group(1)
    repo = match.group(2)
    
    # Strip any trailing .git or trailing slashes
    if repo.endswith(".git"):
        repo = repo[:-4]
    
    return owner, repo


def should_skip_path(path: str) -> bool:
    """Check if the path belongs to standard folders that should be skipped."""
    parts = path.lower().replace("\\", "/").split("/")
    skip_dirs = {"node_modules", ".git", "__pycache__", "dist", "build", "venv", ".idea", ".vscode", "env"}
    return any(part in skip_dirs for part in parts)


def get_file_extension(path: str) -> str:
    _, ext = os.path.splitext(path)
    return ext.lower()


def is_supported_file(path: str) -> bool:
    """Check if file extension is supported."""
    supported_extensions = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", 
        ".cpp", ".c", ".cs", ".rb", ".md", ".yaml", ".yml", ".json", ".toml"
    }
    return get_file_extension(path) in supported_extensions


# Cache Helpers for Repo Summaries
def load_repo_summaries() -> Dict[str, Any]:
    if not os.path.exists(SUMMARIES_FILE):
        return {}
    try:
        with open(SUMMARIES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading summaries file: {e}")
        return {}

def save_repo_summary(repo_id: str, data: Dict[str, Any]):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    summaries = load_repo_summaries()
    summaries[repo_id] = data
    try:
        with open(SUMMARIES_FILE, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving summary file: {e}")


def ingest_repository(url: str, force_reingest: bool = False) -> Dict[str, Any]:
    """
    Ingests a public GitHub repository, chunks code files, stores in ChromaDB,
    and returns metadata + summary.
    """
    owner, repo_name = parse_github_url(url)
    repo_id = f"{owner}/{repo_name}".lower()
    
    # Check cache first
    summaries = load_repo_summaries()
    if repo_id in summaries and not force_reingest:
        logger.info(f"Repository {repo_id} already exists in cache. Returning cached summary.")
        return summaries[repo_id]
        
    logger.info(f"Starting ingestion for repository: {repo_id}...")
    
    # Connect to GitHub
    github_token = os.getenv("GITHUB_TOKEN")
    g = Github(github_token) if github_token else Github()
    
    try:
        repo = g.get_repo(f"{owner}/{repo_name}")
    except Exception as e:
        raise ValueError(f"Failed to access GitHub repository {owner}/{repo_name}. Check if it is public and url is correct. Error: {e}")
        
    # Vector store setup
    db = get_vector_store()
    
    # Clean up existing docs if force_reingest is True
    if force_reingest:
        logger.info(f"Force re-ingest active. Cleaning old vectors for {repo_id}...")
        try:
            db.delete(where={"repo_id": repo_id})
        except Exception as e:
            logger.warning(f"Error cleaning up old vectors: {e}")
            
    # Fetch file tree using Git Tree API (recursive)
    # This is much faster and avoids multiple API calls for folder traversals
    try:
        default_branch = repo.default_branch
        tree = repo.get_git_tree(sha=default_branch, recursive=True)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch repository file tree: {e}")
        
    files_to_fetch = []
    skipped_count = 0
    size_skipped_count = 0
    readme_sha = None
    readme_path = None
    
    # Analyze the tree
    for element in tree.tree:
        if element.type == "blob":  # It is a file
            path = element.path
            
            # Check skip rules
            if should_skip_path(path):
                skipped_count += 1
                continue
                
            if not is_supported_file(path):
                continue
                
            # Skip files larger than 200KB (204,800 bytes)
            # element.size is provided directly in tree API response
            if element.size and element.size > 204800:
                logger.info(f"Skipping large file: {path} ({element.size} bytes)")
                size_skipped_count += 1
                continue
                
            # Track README for the summary card
            filename = os.path.basename(path).lower()
            if filename == "readme.md":
                readme_sha = element.sha
                readme_path = path
                
            files_to_fetch.append(element)
            
    total_files = len(files_to_fetch)
    logger.info(f"Found {total_files} supported files to download. (Skipped: {skipped_count} dir-skipped, {size_skipped_count} size-skipped)")
    
    documents = []
    chunk_count = 0
    languages_found = set()
    
    # Download and chunk files
    for idx, element in enumerate(files_to_fetch):
        path = element.path
        ext = get_file_extension(path)
        
        # Display progress in console
        if (idx + 1) % 10 == 0 or (idx + 1) == total_files:
            logger.info(f"Downloading file {idx + 1}/{total_files}: {path}")
            
        try:
            blob = repo.get_git_blob(element.sha)
            content_bytes = base64.b64decode(blob.content)
            content = content_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Failed to download/decode {path}: {e}")
            continue
            
        # Determine language
        lang_enum = EXTENSION_TO_LANGUAGE.get(ext)
        language_name = lang_enum.value if lang_enum else ext.lstrip(".")
        languages_found.add(language_name)
        
        # Chunking configuration
        if lang_enum:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang_enum,
                chunk_size=1000,
                chunk_overlap=150,
                length_function=get_token_length
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                length_function=get_token_length
            )
            
        chunks = splitter.split_text(content)
        
        for c_idx, chunk_text in enumerate(chunks):
            # Create LangChain Document style dict/object for ingestion
            # Store metadata
            doc_metadata = {
                "file_path": path,
                "language": language_name,
                "extension": ext,
                "chunk_index": c_idx,
                "repo_id": repo_id
            }
            
            # Format chunk with file header so the model knows where it comes from
            chunk_content = f"--- FILE: {path} ---\n{chunk_text}"
            
            documents.append((chunk_content, doc_metadata))
            chunk_count += 1
            
    # Add to ChromaDB in batches of 100 to avoid limits or lockups
    if documents:
        logger.info(f"Inserting {chunk_count} chunks into ChromaDB...")
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc[0] for doc in batch]
            metadatas = [doc[1] for doc in batch]
            db.add_texts(texts=texts, metadatas=metadatas)
    
    # Fetch README content for the summary card if available
    readme_content = ""
    if readme_sha:
        try:
            blob = repo.get_git_blob(readme_sha)
            content_bytes = base64.b64decode(blob.content)
            readme_content = content_bytes.decode("utf-8", errors="replace")[:4000] # Limit size passed to summary LLM
        except Exception as e:
            logger.warning(f"Failed to fetch README content: {e}")
            
    # Generate the Repository Summary Card
    logger.info("Generating repository summary...")
    summary = generate_summary_via_llm(owner, repo_name, list(languages_found), total_files, readme_content)
    
    # Cache result
    result_data = {
        "repo_id": repo_id,
        "repo_name": repo_name,
        "owner": owner,
        "total_files": total_files,
        "total_chunks": chunk_count,
        "languages": sorted(list(languages_found)),
        "summary": summary
    }
    
    save_repo_summary(repo_id, result_data)
    logger.info(f"Ingestion completed for {repo_id} successfully.")
    
    return result_data


def generate_summary_via_llm(
    owner: str, 
    repo_name: str, 
    languages: List[str], 
    total_files: int, 
    readme_content: str
) -> str:
    """Generates a concise markdown summary card of the repo using the fallback-enabled LLM manager."""
    from backend.llm_manager import CooldownFallbackChatModel
    
    llm = CooldownFallbackChatModel()
    
    languages_str = ", ".join(languages) if languages else "Unknown"
    
    prompt = f"""You are a professional technical writer and system architect.
Generate a concise, high-impact repository summary card for the repository: '{owner}/{repo_name}'.

Repository Details:
- Owner: {owner}
- Repository Name: {repo_name}
- Total files scanned: {total_files}
- Languages/file formats detected: {languages_str}

{"Here is an excerpt from the repository's README.md to help understand its purpose:" if readme_content else "No README file was found. Please estimate the repository purpose based on the name and detected files."}
{readme_content}

Create a summary card formatted in markdown. It must contain:
1. **Core Purpose**: A 2-3 sentence overview of what the project does.
2. **Key Capabilities**: 3-4 bullet points highlighting main features/components.
3. **Tech Highlights**: A brief note on the main frameworks/technologies visible.

Keep it readable, engaging, and professional. Avoid lengthy intros/outros. Generate ONLY the markdown content.
"""
    
    messages = [
        SystemMessage(content="You are a helpful software engineering assistant who creates repo summaries."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Failed to generate summary card: {e}")
        return f"### {repo_name}\nThis repository has been successfully indexed. (Unable to generate automated summary card: {e})"
