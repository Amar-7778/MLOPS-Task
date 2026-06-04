import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from backend.ingest import get_vector_store
from backend.llm_manager import CooldownFallbackChatModel

logger = logging.getLogger(__name__)

# Specialized search query maps for each RAG mode to optimize retrieval
MODE_SEARCH_QUERIES = {
    "tech_stack": "project configuration dependencies package manifest framework libraries database architecture infra testing deploy docker compose",
    "data_flow": "data flow entry point route handler controllers database models query pipeline processing stream workflow input output",
    "system_design": "high level architecture component breakdown services interfaces modules communication scalability patterns design folder layout",
}

# Specialized System Prompt Templates
TECH_STACK_SYSTEM_PROMPT = """You are an expert software architect analyzing the tech stack of a repository.
Your task is to identify and describe the technical architecture based on the provided code snippets and files.

Use the retrieved files to analyze:
1. **Programming Languages & Dialects**: Highlight the primary and secondary languages.
2. **Frameworks & Libraries**: Main application frameworks, routing libraries, utility frameworks.
3. **Databases, Caching & Storage**: Database engines, ORMs/ODMs, KV-stores, file systems.
4. **DevOps & Infrastructure**: Containers, CI/CD configs, cloud services, packaging.
5. **Testing & Quality Assurance**: Test runners, assertion libraries, linters.
6. **Architecture Patterns**: Identify if MVC, hexagonal, event-driven, serverless, monolith, etc.

Structure your response clearly with headers, lists, and tables where helpful. Do not make things up. If you cannot find details about a section, state that it was not detected in the provided codebase.
"""

DATA_FLOW_SYSTEM_PROMPT = """You are an expert systems engineer analyzing the data flow of a repository.
Your task is to trace how data enters, moves through, is processed by, and exits the application using the provided code snippets.

Explain clearly:
1. **Entry Points**: Where do requests or executions start? (e.g. main files, server setup, REST/GraphQL routes, cron jobs, event listeners)
2. **Processing & Business Logic**: How are inputs validated, parsed, and passed through business layers/services?
3. **State & Storage Access**: How does the application interact with databases, files, or external services to read/write state?
4. **Outputs & Responses**: What does the system return, publish, or log when processing completes?
5. **Key Data Models**: List/describe the main data entities, models, schemas, or interfaces found in the code.

Provide a step-by-step walk-through of the data flow. Use markdown formatting to make the flow easy to read.
"""

SYSTEM_DESIGN_SYSTEM_PROMPT = """You are an expert systems architect analyzing the high-level design of a repository.
Your task is to dissect the modular design, components, and design decisions evident in the provided files.

Provide an architectural document covering:
1. **High-Level Design**: Monolithic vs. Microservices vs. Serverless, layering layout (e.g. presentation/business/data layers).
2. **Component Breakdown**: Describe major directories, modules, or services and their responsibilities.
3. **Inter-Component Communication**: How do different services or modules interact? (e.g. direct imports, function calls, internal events, REST, gRPC, pub-sub)
4. **Design Quality & Decisions**: Comment on clean code practices, design pattern usage (e.g. singleton, factory, dependency injection), potential single points of failure, or bottlenecks.
5. **Scalability & Security**: Analyze how the project handles load, configuration, security headers, authentication, or secrets.

Be constructive, architectural, and analytical. Use markdown headings and lists.
"""

CHAT_SYSTEM_PROMPT = """You are a helpful, expert AI coding assistant. You have access to code chunks retrieved from a repository.
Your task is to answer the user's question about the repository in a precise, helpful, and technically accurate manner.

Feel free to write code examples based on the retrieved code, explain logic, or walk through files.
Always base your response on the retrieved context. If the answer cannot be found in the context, be honest and state that, but feel free to provide general educational guidance based on standard practices if helpful.
"""


def query_repository(
    repo_id: str, 
    query: str, 
    mode: str, 
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Retrieves relevant snippets from ChromaDB and invokes the fallback-enabled LLM model.
    Returns:
      {
        "answer": str,
        "sources": List[Dict[str, str]], # unique source files used
        "provider": str                 # LLM provider ("Gemini" or "Groq")
      }
    """
    repo_id_clean = repo_id.lower()
    
    # 1. Determine retrieval search query and system prompt based on mode
    system_prompt = CHAT_SYSTEM_PROMPT
    search_query = query
    
    if mode == "tech_stack":
        system_prompt = TECH_STACK_SYSTEM_PROMPT
        search_query = MODE_SEARCH_QUERIES["tech_stack"]
    elif mode == "data_flow":
        system_prompt = DATA_FLOW_SYSTEM_PROMPT
        search_query = MODE_SEARCH_QUERIES["data_flow"]
    elif mode == "system_design":
        system_prompt = SYSTEM_DESIGN_SYSTEM_PROMPT
        search_query = MODE_SEARCH_QUERIES["system_design"]
        
    # 2. Retrieve documents from ChromaDB
    db = get_vector_store()
    
    # Configure retriever filtering by repo_id
    # We retrieve up to 15 chunks for analysis modes to ensure rich context, 10 for chat
    k = 15 if mode in ["tech_stack", "data_flow", "system_design"] else 10
    
    logger.info(f"Retrieving top {k} chunks from ChromaDB for {repo_id_clean} using query: '{search_query[:50]}...'")
    
    try:
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                "filter": {"repo_id": repo_id_clean}
            }
        )
        docs = retriever.invoke(search_query)
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        docs = []
        
    # 3. Format retrieved documents as context
    context_chunks = []
    sources = []
    seen_sources = set()
    
    for doc in docs:
        context_chunks.append(doc.page_content)
        path = doc.metadata.get("file_path", "unknown")
        lang = doc.metadata.get("language", "unknown")
        if path not in seen_sources:
            seen_sources.add(path)
            sources.append({"file_path": path, "language": lang})
            
    context_text = "\n\n".join(context_chunks)
    
    # 4. Construct messages for LangChain
    messages = [
        SystemMessage(content=system_prompt)
    ]
    
    # Append chat history (only for chat mode, keep last 10 messages max)
    if mode == "chat" and chat_history:
        for msg in chat_history[-10:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
                
    # Final User Message combining Context and Query/Instructions
    if mode in ["tech_stack", "data_flow", "system_design"]:
        user_content = f"""Please perform the requested repository analysis.

Retrieved repository files/code context:
========================================
{context_text}
========================================

Generate the report now.
"""
    else:
        # Chat mode
        user_content = f"""Retrieved repository files/code context:
========================================
{context_text}
========================================

User Question: {query}
"""
    
    messages.append(HumanMessage(content=user_content))
    
    # 5. Invoke CooldownFallbackChatModel
    llm = CooldownFallbackChatModel()
    
    try:
        logger.info(f"Invoking Fallback LLM manager. Mode={mode}")
        response = llm.invoke(messages)
        
        # Get provider from response metadata
        provider = response.response_metadata.get("provider", "Gemini")
        
        return {
            "answer": response.content,
            "sources": sources,
            "provider": provider
        }
    except Exception as e:
        logger.error(f"Failed to generate response in chain: {e}")
        return {
            "answer": f"An error occurred while generating the response: {e}\n\nPlease check your API keys and console logs.",
            "sources": [],
            "provider": "Error"
        }
