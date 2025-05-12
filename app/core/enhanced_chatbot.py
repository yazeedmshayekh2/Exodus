"""
Enhanced Chatbot implementation with memory and model switching capabilities
"""

import os
import logging
from typing import Dict, List, Optional, Any
import json
import shutil
import tempfile
import subprocess
import time
import torch

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.language_models import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.pydantic_v1 import Field

from app.core.chatbot import FAQChatbot
from app.core.memory import ConversationMemory
from app.core.response import format_response
from app.utils.moderation import moderate_content

# Configure logging
logger = logging.getLogger(__name__)

class HuggingFaceTransformersLLM(LLM):
    """
    Wrapper around Hugging Face Transformers models.
    """
    model_name: str = Field(...)
    tokenizer: Any = Field(default=None)
    model: Any = Field(default=None)
    pipeline: Any = Field(default=None)
    max_length: int = Field(default=2048)
    temperature: float = Field(default=0.7)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model and tokenizer from HuggingFace: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with appropriate precision based on available hardware
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            # Create a pipeline for text generation
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                truncation=True,
                max_new_tokens=self.max_length,
                temperature=self.temperature
            )
            
            logger.info(f"Successfully loaded HuggingFace model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading HuggingFace model {self.model_name}: {e}")
            raise
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        """Generate text using the model."""
        try:
            logger.info(f"Generating text with HuggingFace model, input length: {len(prompt)}")
            
            # Generate text using the pipeline
            outputs = self.pipeline(
                prompt,
                max_new_tokens=kwargs.get("max_new_tokens", self.max_length),
                do_sample=True,
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", 0.95),
                top_k=kwargs.get("top_k", 50),
                truncation=True,
                return_full_text=False
            )
            
            # Extract the generated text
            generated_text = outputs[0]["generated_text"]
            
            # Apply stop sequences if provided
            if stop:
                for sequence in stop:
                    if sequence in generated_text:
                        generated_text = generated_text[:generated_text.find(sequence)]
            
            return generated_text
        except Exception as e:
            logger.error(f"Error generating text with HuggingFace model: {e}")
            raise
    
    @property
    def _llm_type(self) -> str:
        return "huggingface_transformers"

class EnhancedChatbot(FAQChatbot):
    """
    Enhanced chatbot with memory and model switching capabilities
    
    This extends the base FAQChatbot to add:
    - Conversation memory to maintain context
    - Model switching functionality
    - Enhanced prompt templates using conversation history
    """
    
    # Available models that can be used by the chatbot
    AVAILABLE_MODELS = {
        "llama3.1:8b": {
            "name": "Llama 3.1 8B",
            "description": "Meta's Llama 3.1 8B model for general purpose tasks",
            "context_length": 8192,
            "temperature": 0.7,
            "model_type": "ollama"
        },
        "Qwen/Qwen3-8B-FP8": {
            "name": "Qwen3 8B FP8",
            "description": "Alibaba's Qwen3 8B model with FP8 quantization supporting thinking mode",
            "context_length": 32768,
            "temperature": 0.6,
            "truncation": True,
            "model_type": "huggingface"
        }
    }
    
    def __init__(self):
        """Initialize the enhanced chatbot with memory and configurable model"""
        super().__init__()
        
        # Initialize conversation memory
        self.memory = ConversationMemory(
            max_users=int(os.getenv('MAX_USERS', '1000')),
            max_messages_per_user=int(os.getenv('MAX_MESSAGES_PER_USER', '20'))
        )
        
        # Set up memory configuration
        self.memory_enabled = os.getenv('ENABLE_MEMORY', 'true').lower() == 'true'
        
        # Create enhanced prompts that use conversation history
        # Enhanced chat prompt that includes conversation history
        self.memory_chat_prompt = PromptTemplate(
            input_variables=["language", "history", "query"],
            template="""You are a helpful customer service assistant for an Arabic and English bilingual organization.
            
            Language: {language}
            
            Previous conversation history:
            {history}
            
            Current query: {query}
            
            Guidelines:
            - Be concise but helpful and friendly
            - For Arabic, use proper formal Arabic but be conversational
            - For English, be friendly and conversational
            - Maintain context from the previous conversation
            - If you're being asked about something from earlier in the conversation, refer back to it
            - Keep responses brief, focused, and helpful

            Response:"""
        )
        
        # Intent classification prompt for distinguishing between FAQ and conversational queries
        self.query_type_prompt = PromptTemplate(
            input_variables=["query", "language"],
            template="""Determine if the user's query is looking for factual information (FAQ) or is a conversational message.

            User query: {query}
            Language: {language}
            
            Guidelines:
            - FAQ: Questions seeking specific information, facts, procedures, or technical details
            - CONVERSATION: Greetings, casual chat, opinions, personal assistance, or open-ended questions
            
            Examples:
            "What are your office hours?" -> FAQ
            "How do I reset my password?" -> FAQ
            "What documents do I need for a visa application?" -> FAQ
            "Hello, how are you today?" -> CONVERSATION
            "Can you help me with something?" -> CONVERSATION
            "What do you think about this situation?" -> CONVERSATION
            "I'm having trouble understanding this." -> CONVERSATION
            "Could you help me?" -> CONVERSATION
            
            Respond with only one of these categories:
            FAQ - seeking specific information or facts
            CONVERSATION - greeting, casual chat, or personal assistance
            
            Category:"""
        )
        
        # Set up model configuration AFTER defining prompts
        self.current_model = os.getenv('MODEL_NAME', 'llama3.1:8b')
        self.previously_loaded_model = None
        self.configure_model(self.current_model)
        
        # Initialize chains after configure_model has set up the LLM
        self.query_type_chain = (
            RunnablePassthrough() |
            self.query_type_prompt |
            self.llm |
            StrOutputParser()
        )
        
        # Enhanced chain with memory
        self.memory_chat_chain = (
            RunnablePassthrough() |
            self.memory_chat_prompt |
            self.llm |
            StrOutputParser()
        )
    
    def configure_model(self, model_name: str) -> bool:
        """
        Configure the chatbot to use a different model
        
        Args:
            model_name (str): Name of the model to use
            
        Returns:
            bool: Whether the model was successfully configured
        """
        # Check if model name is valid
        if model_name not in self.AVAILABLE_MODELS:
            logger.warning(f"Invalid model name: {model_name}. Using default model.")
            model_name = "llama3.1:8b"
        
        logger.info(f"Configuring chatbot to use model: {model_name}")
        
        # If it's the same model as current, no need to reload
        if model_name == self.current_model:
            logger.info(f"Model {model_name} is already loaded and active")
            return True
        
        # Store current model to unload it after successful switch
        old_model = self.previously_loaded_model
        
        # Update model configuration
        model_config = self.AVAILABLE_MODELS[model_name]
        previous_model = self.current_model
        
        # Create new LLM with updated configuration based on model type
        try:
            model_type = model_config.get("model_type", "ollama")
            
            if model_type == "huggingface":
                logger.info(f"Creating Hugging Face Transformers LLM with model {model_name}")
                
                # Create a HuggingFaceTransformersLLM instance
                self.llm = HuggingFaceTransformersLLM(
                    model_name=model_name,
                    max_length=model_config.get("context_length", 4096),
                    temperature=model_config.get("temperature", 0.7)
                )
            else:  # Default to Ollama
                logger.info(f"Creating Ollama LLM with model {model_name}")
                
                # Try to pull the model if needed
                try:
                    command = f"ollama pull {model_name}"
                    logger.info(f"Pulling Ollama model {model_name} if needed")
                    pull_result = subprocess.run(command, shell=True, capture_output=True, text=True)
                    if pull_result.returncode != 0:
                        logger.warning(f"Failed to pull Ollama model {model_name}: {pull_result.stderr}")
                        # Continue anyway, as the model might already be available
                except Exception as pull_err:
                    logger.warning(f"Error pulling Ollama model {model_name}: {pull_err}")
                    # Continue anyway
                
                # Create Ollama LLM
                base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
                self.llm = OllamaLLM(
                    base_url=base_url,
                    model=model_name,
                    temperature=model_config.get("temperature", 0.7),
                    top_p=0.95,
                    num_ctx=model_config.get("context_length", 4096),
                    num_predict=-1
                )
            
            # Update current model only after successful LLM creation
            self.current_model = model_name
            
            # Update all chains that use the LLM
            logger.info("Updating all LLM chains")
            self.intent_chain = (
                RunnablePassthrough() |
                self.intent_prompt |
                self.llm |
                StrOutputParser()
            )
            
            self.faq_chain = (
                RunnablePassthrough() |
                self.faq_prompt |
                self.llm |
                StrOutputParser()
            )
            
            self.greeting_chain = (
                RunnablePassthrough() |
                self.greeting_prompt |
                self.llm |
                StrOutputParser()
            )
            
            self.chitchat_chain = (
                RunnablePassthrough() |
                self.chitchat_prompt |
                self.llm |
                StrOutputParser()
            )
            
            self.query_type_chain = (
                RunnablePassthrough() |
                self.query_type_prompt |
                self.llm |
                StrOutputParser()
            )
            
            self.memory_chat_chain = (
                RunnablePassthrough() |
                self.memory_chat_prompt |
                self.llm |
                StrOutputParser()
            )
            
            # Handle previous model cleanup based on model type
            if old_model and old_model != model_name:
                old_model_config = self.AVAILABLE_MODELS.get(old_model, {})
                old_model_type = old_model_config.get("model_type", "ollama")
                
                if old_model_type == "huggingface" and hasattr(torch, 'cuda'):
                    logger.info(f"Clearing previous HuggingFace model from GPU memory: {old_model}")
                    try:
                        # Force garbage collection to clear memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e:
                        logger.warning(f"Error clearing GPU memory: {e}")
                elif old_model_type == "ollama":
                    logger.info(f"Unloading previous Ollama model: {old_model}")
                    # Give a moment for the new model to initialize fully
                    time.sleep(1)
                    try:
                        # Try unloading via Ollama API
                        base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
                        api_command = f"curl -X DELETE {base_url}/api/remove -d '{{\"name\":\"{old_model}\"}}'"
                        subprocess.run(api_command, shell=True, capture_output=True, text=True)
                        logger.info(f"Sent unload request for Ollama model: {old_model}")
                    except Exception as unload_err:
                        logger.warning(f"Error unloading Ollama model {old_model}: {unload_err}")
            
            # Save currently loaded model name
            self.previously_loaded_model = model_name
            
            logger.info(f"Successfully configured model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error configuring model: {e}")
            # Revert to previous model in case of failure
            self.current_model = previous_model
            return False
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models
        
        Returns:
            List[Dict[str, Any]]: List of available models with details
        """
        return [
            {
                "id": model_id,
                "name": config["name"],
                "description": config["description"],
                "current": model_id == self.current_model,
                "model_type": config.get("model_type", "ollama")
            }
            for model_id, config in self.AVAILABLE_MODELS.items()
        ]
    
    def remove_model(self, model_name: str) -> bool:
        """
        Remove a model from the available models
        
        Args:
            model_name (str): Name of the model to remove
            
        Returns:
            bool: Whether the model was successfully removed
        """
        # For safety, don't allow removing any models
        logger.warning(f"Cannot remove built-in model {model_name}")
        return False
    
    def format_history(self, history: List[Dict[str, str]]) -> str:
        """
        Format conversation history for inclusion in prompts
        
        Args:
            history (List[Dict[str, str]]): List of message dictionaries
            
        Returns:
            str: Formatted conversation history
        """
        if not history:
            return "No previous conversation."
            
        formatted = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
            
        return "\n".join(formatted)
    
    def get_enhanced_response(self, query: str, user_id: str = "default") -> str:
        """
        Generate a response using conversation history
        
        Args:
            query (str): User's input query
            user_id (str): Unique identifier for the user
            
        Returns:
            str: Formatted response in the appropriate language
        """
        try:
            language = self.detect_language(query)
            logger.info(f"Processing query with memory: '{query}' (Language: {language})")
            
            # Check for special commands
            if query.startswith("/model "):
                model_name = query[7:].strip()
                success = self.configure_model(model_name)
                if success:
                    return format_response(f"Model changed to {model_name}", language)
                else:
                    return format_response(f"Error changing model to {model_name}. Available models: {', '.join(self.AVAILABLE_MODELS.keys())}", language)
                    
            elif query == "/models":
                models_list = self.list_available_models()
                models_text = "\n".join([f"- {m['name']} ({m['id']}): {m['description']}{' (current)' if m['current'] else ''}" for m in models_list])
                return format_response(f"Available models:\n{models_text}", language)
                
            elif query == "/clear":
                self.memory.clear_user_history(user_id)
                return format_response("Conversation history cleared", language)
            
            # Add user message to history
            if self.memory_enabled:
                self.memory.add_message(user_id, "user", query)
                
            # Content moderation
            is_inappropriate, reason = moderate_content(query)
            if is_inappropriate:
                logger.warning(f"Inappropriate content detected: {reason}")
                response = self.inappropriate_chain.invoke({"language": language})
                
                # Add assistant response to history
                if self.memory_enabled:
                    self.memory.add_message(user_id, "assistant", response)
                    
                return format_response(response, language)
            
            # Get conversation history
            if self.memory_enabled:
                history = self.memory.get_formatted_history(user_id)
                formatted_history = self.format_history(history[:-1])  # Exclude the just-added message
            else:
                formatted_history = "Memory disabled."
                history = []
            
            # Detect if this is a FAQ query or conversational query
            query_type = self.query_type_chain.invoke({
                "query": query,
                "language": language
            }).strip().upper()
            
            logger.info(f"Query type classification: {query_type}")
            
            # Handle based on query type
            if query_type == "FAQ" and not any(special in query.lower() for special in ["hello", "hi", "hey", "السلام", "مرحبا"]):
                # Use the FAQ matching system for factual queries
                logger.info("Using FAQ response system for factual query")
                response = super().get_response(query, user_id)
            elif self.memory_enabled and len(history) > 1:
                # Use memory-based conversation for continuity
                logger.info("Using conversation memory for response")
                response = self.memory_chat_chain.invoke({
                    "language": language,
                    "history": formatted_history,
                    "query": query
                })
            else:
                # Handle simple conversational queries without enough history
                logger.info("Using standard conversation response (no sufficient memory)")
                response = super().get_response(query, user_id)
            
            # Add assistant response to history
            if self.memory_enabled:
                self.memory.add_message(user_id, "assistant", response)
                
            return format_response(response, language)
            
        except Exception as e:
            logger.error(f"Error generating enhanced response: {e}", exc_info=True)
            return super().get_response(query, user_id)  # Fall back to standard response 