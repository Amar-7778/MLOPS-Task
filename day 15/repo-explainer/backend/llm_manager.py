import os
import time
import logging
from typing import List, Any, Dict, Optional
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import RunnableSerializable, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def is_rate_limit_exception(e: Exception) -> bool:
    """Helper to check if an exception represents a rate limit or resource exhaustion error."""
    err_str = str(e).lower()
    if any(p in err_str for p in ["429", "rate limit", "quota", "exhausted", "resource_exhausted", "resource exhausted"]):
        return True
        
    try:
        from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError
        if isinstance(e, ResourceExhausted):
            return True
        if isinstance(e, GoogleAPICallError) and e.code == 429:
            return True
    except ImportError:
        pass
        
    return False


class LLMCooldownManager:
    """Manages the fallback logic and 60-second cooldown window for Gemini to Groq."""
    
    def __init__(self):
        self.cooldown_duration = 60.0  # seconds
        self.gemini_cooldown_until = 0.0
        
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        self.gemini_model = None
        self.groq_model = None
        
        # Initialize Gemini if key exists
        if self.gemini_api_key:
            logger.info("Initializing Gemini 2.5 Flash...")
            self.gemini_model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=self.gemini_api_key,
                temperature=0.2
            )
        else:
            logger.warning("GEMINI_API_KEY not found in environment.")

        # Initialize Groq fallback if key exists
        if self.groq_api_key:
            logger.info("Initializing Groq (llama-3.3-70b-versatile)...")
            self.groq_model = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=self.groq_api_key,
                temperature=0.2
            )
        else:
            logger.warning("GROQ_API_KEY not found in environment.")

    def is_gemini_cooling_down(self) -> bool:
        return time.time() < self.gemini_cooldown_until

    def trigger_cooldown(self):
        self.gemini_cooldown_until = time.time() + self.cooldown_duration
        logger.warning(f"Gemini API rate limit/quota hit. Switched to Groq fallback for {self.cooldown_duration}s.")

    def get_effective_model(self) -> tuple[Any, str]:
        """Returns the active model instance and its provider name ('Gemini' or 'Groq')."""
        if not self.gemini_model and not self.groq_model:
            raise ValueError("Neither GEMINI_API_KEY nor GROQ_API_KEY is configured in the environment.")
            
        if not self.gemini_model:
            return self.groq_model, "Groq"
            
        if not self.groq_model:
            return self.gemini_model, "Gemini"
            
        if self.is_gemini_cooling_down():
            remaining = int(self.gemini_cooldown_until - time.time())
            logger.info(f"Gemini is in cooldown. Remaining: {remaining}s. Using Groq.")
            return self.groq_model, "Groq"
            
        return self.gemini_model, "Gemini"


# Singleton instance of the cooldown manager
cooldown_manager = LLMCooldownManager()


class CooldownFallbackChatModel(RunnableSerializable[List[BaseMessage], AIMessage]):
    """
    A custom LangChain Runnable that delegates execution to either Gemini or Groq,
    implementing 60-second cooldown switching logic.
    """
    
    class Config:
        arbitrary_types_allowed = True

    def invoke(
        self,
        input: List[BaseMessage],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AIMessage:
        model, provider = cooldown_manager.get_effective_model()
        
        if provider == "Groq":
            try:
                response = model.invoke(input, config, **kwargs)
                if not response.response_metadata:
                    response.response_metadata = {}
                response.response_metadata["provider"] = "Groq"
                return response
            except Exception as e:
                logger.error(f"Groq API call failed: {e}")
                raise e
                
        # Try Gemini
        try:
            response = model.invoke(input, config, **kwargs)
            if not response.response_metadata:
                response.response_metadata = {}
            response.response_metadata["provider"] = "Gemini"
            return response
        except Exception as e:
            if is_rate_limit_exception(e):
                cooldown_manager.trigger_cooldown()
                if cooldown_manager.groq_model:
                    logger.info("Retrying query with Groq fallback immediately.")
                    fallback_model = cooldown_manager.groq_model
                    response = fallback_model.invoke(input, config, **kwargs)
                    if not response.response_metadata:
                        response.response_metadata = {}
                    response.response_metadata["provider"] = "Groq"
                    return response
                else:
                    logger.error("Gemini failed and no Groq fallback is configured.")
                    raise e
            else:
                logger.error(f"Gemini failed with a non-rate-limit error: {e}")
                raise e
