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
import triton.compiler.errors

from app.core.chatbot import FAQChatbot
from app.core.memory import ConversationMemory
from app.core.response import format_response
from app.utils.moderation import moderate_content
from app.core.model_supervisor import ModelSupervisor

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
    awq_config_path: Optional[str] = Field(default=None)
    model_path: Optional[str] = Field(default=None)  # Added for local model support
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            # Determine if we're loading from a local path or HuggingFace
            model_source = self.model_path if self.model_path else self.model_name
            logger.info(f"Loading model and tokenizer from: {model_source}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            
            # Check if we're using the AWQ model
            is_awq_model = "AWQ" in self.model_name
            logger.info(f"Is AWQ model: {is_awq_model}")
            
            # Try to determine if CUDA is available
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {cuda_available}")
            
            # For AWQ models, always use float16 (AWQ is already quantized)
            if is_awq_model and cuda_available:
                logger.info(f"Loading AWQ model with float16 precision")
                
                # Set up kwargs for AWQ model
                kwargs = {
                    "torch_dtype": torch.float16,  # Force float16 for AWQ
                    "device_map": "auto",
                    "trust_remote_code": True
                }
                
                # Load AWQ config if provided
                if self.awq_config_path:
                    logger.info(f"Using AWQ config from {self.awq_config_path}")
                    # For AWQ models, we need to use specific loading parameters
                    # rather than passing a quantization config directly
                    kwargs["load_in_4bit"] = True
                    kwargs["bnb_4bit_compute_dtype"] = torch.float16
                    kwargs["bnb_4bit_use_double_quant"] = True
                    kwargs["bnb_4bit_quant_type"] = "nf4"
                
                # Load the model with AWQ optimizations
                try:
                    # Try two approaches for loading AWQ models
                    try:
                        # First try loading with BitsAndBytes quantization
                        import bitsandbytes as bnb
                        from transformers import BitsAndBytesConfig
                        
                        logger.info("Configuring AWQ model with BitsAndBytes")
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        
                        # Remove any config params that might conflict
                        if "load_in_4bit" in kwargs:
                            del kwargs["load_in_4bit"]
                        if "bnb_4bit_compute_dtype" in kwargs:
                            del kwargs["bnb_4bit_compute_dtype"]
                        if "bnb_4bit_use_double_quant" in kwargs:
                            del kwargs["bnb_4bit_use_double_quant"]
                        if "bnb_4bit_quant_type" in kwargs:
                            del kwargs["bnb_4bit_quant_type"]
                        
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_source,
                            quantization_config=quantization_config,
                            **kwargs
                        )
                        
                    except Exception as e1:
                        # If BitsAndBytes approach fails, try with AWQ module
                        logger.warning(f"BitsAndBytes loading failed: {e1}. Trying AWQ module...")
                        
                        try:
                            from awq import AutoAWQForCausalLM
                            logger.info("Using AutoAWQForCausalLM for loading")
                            
                            # Use AWQ-specific loading
                            self.model = AutoAWQForCausalLM.from_quantized(
                                model_source,
                                **kwargs
                            )
                        except ImportError:
                            # Fall back to standard loading if specific modules not available
                            logger.warning("AWQ specific modules not found, using standard loading")
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_source,
                                **kwargs
                            )
                        
                except Exception as e:
                                            # Try yet another approach for AWQ models
                        try:
                            # For Qwen AWQ models specifically
                            if "Qwen" in self.model_name and "AWQ" in self.model_name:
                                logger.info("Trying specialized loading for Qwen AWQ model")
                                # Load the model skipping AWQ-specific quantization
                                non_awq_name = self.model_name.replace("-AWQ", "")
                                
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    non_awq_name,  # Try with non-AWQ model name
                                    torch_dtype=torch.float16,
                                    device_map="auto",
                                    trust_remote_code=True
                                )
                                logger.info(f"Successfully loaded using non-AWQ model name: {non_awq_name}")
                            else:
                                # Last resort - try with minimal parameters
                                logger.warning(f"Error loading AWQ model: {e}. Using basic parameters.")
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    model_source,
                                    torch_dtype=torch.float16,
                                    device_map="auto",
                                    trust_remote_code=True
                                )
                        except Exception as e_final:
                            # Absolute last resort - raise a clear error
                            logger.error(f"All AWQ loading approaches failed: {e_final}")
                            raise ValueError(f"Could not load model {model_source} with any available approach")
            # For non-AWQ models, try different precisions
            elif cuda_available:
                try:
                    logger.info(f"Attempting to load model with bfloat16 precision")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_source,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                except Exception as e1:
                    logger.warning(f"Failed to load with bfloat16 precision: {e1}. Trying with float16...")
                    
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_source,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            trust_remote_code=True
                        )
                    except Exception as e2:
                        logger.warning(f"Failed to load with float16 precision: {e2}. Falling back to CPU...")
                        
                        # Fall back to CPU with lower precision
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_source,
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        )
            else:
                # No CUDA, load on CPU
                logger.info("CUDA not available, loading model on CPU")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            
            # Create a pipeline for text generation with specific dtype matching
            try:
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    truncation=True,
                    max_new_tokens=self.max_length,
                    temperature=self.temperature,
                    torch_dtype=self.model.dtype  # Match pipeline dtype to model dtype
                )
            except Exception as pipeline_error:
                logger.warning(f"Error creating pipeline with dtype matching: {pipeline_error}")
                # Fallback to a simpler pipeline configuration
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    truncation=True,
                    max_new_tokens=self.max_length,
                    temperature=self.temperature
                )
            
            logger.info(f"Successfully loaded HuggingFace model: {model_source}")
        except Exception as e:
            logger.error(f"Error loading HuggingFace model {model_source}: {e}")
            raise
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        """Generate text using the model."""
        try:
            logger.info(f"Generating text with HuggingFace model, input length: {len(prompt)}")
            
            # Check if model is AWQ model
            is_awq_model = "AWQ" in self.model_name
            
            try:
                # Generate text using the pipeline
                if is_awq_model:
                    # For AWQ models, use more conservative settings to avoid dtype errors
                    outputs = self.pipeline(
                        prompt,
                        max_new_tokens=kwargs.get("max_new_tokens", self.max_length),
                        do_sample=True,
                        temperature=kwargs.get("temperature", self.temperature),
                        top_p=kwargs.get("top_p", 0.9),
                        top_k=kwargs.get("top_k", 40),
                        truncation=True,
                        return_full_text=False,
                        use_cache=True
                    )
                else:
                    # For non-AWQ models, use standard settings
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
                
            except triton.compiler.errors.CompilationError as e:
                logger.error(f"Triton compilation error with AWQ model: {e}")
                # Return a fallback response for AWQ models with compilation errors
                return "I'm having some technical difficulties with my advanced language capabilities. Let me switch to a more reliable model to assist you. Please try again or ask another question."
                
            except Exception as e2:
                logger.error(f"Pipeline execution error: {e2}")
                raise e2
                
        except Exception as e:
            logger.error(f"Error generating text with HuggingFace model: {e}")
            # Fall back to a simple fixed response
            return "I'm currently experiencing technical difficulties. Please try again in a moment."
    
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
    
    # Models are managed by ModelSupervisor
    
    def __init__(self):
        """Initialize the enhanced chatbot with memory and configurable model"""
        super().__init__()
        
        # Initialize model supervisor
        self.model_supervisor = ModelSupervisor()
        
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
            template="""You are a highly knowledgeable customer service assistant for an Arabic and English bilingual organization.

Language: {language}

Previous conversation history:
{history}

Current query: {query}

Response Guidelines:
1. Answer Format:
   - If the question asks for steps or procedures, use numbered steps (1., 2., 3., etc.)
   - If it's a factual question, provide specific details and examples
   - For complex topics, break down the information into clear sections

2. Content Requirements:
   - Always provide specific, relevant information instead of generic responses
   - Include exact details, numbers, or references when available
   - If technical terms are used, briefly explain them
   - If there are prerequisites or important notes, list them first

3. Language Style:
   - For Arabic: Use formal Modern Standard Arabic (فصحى) but maintain clarity
   - For English: Use clear, professional language
   - Avoid generic phrases like "I can help you with that" without actual help
   - Be direct and specific in addressing the query

4. Structure:
   - For multi-part questions, address each part separately
   - Use bullet points or numbering for lists
   - If relevant, include:
     * Prerequisites or requirements
     * Important warnings or notes
     * Next steps or follow-up actions

5. Context Awareness:
   - Reference relevant information from previous conversation
   - If clarification is needed, ask specific questions
   - If information is incomplete, state what additional details are needed

Remember: Your response should be tailored to the exact query and context, not a generic template answer.

Response:"""
        )

        # Intent classification prompt for distinguishing between FAQ and conversational queries
        self.query_type_prompt = PromptTemplate(
            input_variables=["query", "language"],
            template="""Analyze the user's query to determine if it requires factual information (FAQ) or is conversational in nature.

Query for analysis: {query}
Language: {language}

Classification Guidelines:

1. FAQ Category (Factual/Informational):
   - Questions seeking specific information or procedures
   - Technical or process-related inquiries
   - Questions about policies, requirements, or specifications
   - Queries that need precise, factual answers
   Examples:
   - "What documents do I need for X?"
   - "How do I perform X procedure?"
   - "What are the requirements for X?"

2. CONVERSATION Category:
   - Greetings and social interactions
   - Open-ended discussions
   - Personal opinions or advice
   - General assistance requests
   Examples:
   - "Hello, how are you?"
   - "Can you help me understand..."
   - "What do you think about..."

Respond with exactly one category:
FAQ
or
CONVERSATION

Category:"""
        )
        
        # Set up model configuration AFTER defining prompts
        self.current_model = self.model_supervisor.get_current_model()
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
        # Let the model supervisor handle the model switch
        success = self.model_supervisor.switch_model(model_name)
        
        if not success:
            logger.error(f"Model supervisor failed to switch to model: {model_name}")
            return False
        
        # Update current model name from supervisor
        self.current_model = self.model_supervisor.get_current_model()
        logger.info(f"Using model: {self.current_model}")
        
        # Choose the appropriate LLM implementation
        model_config = self.model_supervisor.AVAILABLE_MODELS.get(self.current_model, {})
        model_type = model_config.get("model_type", "ollama")
        
        if model_type == "huggingface":
            # Initialize HuggingFace model with appropriate configuration
            self.llm = HuggingFaceTransformersLLM(
                model_name=self.current_model,
                model_path=model_config.get("model_path"),  # Pass the model path if available
                temperature=model_config.get("temperature", 0.7),
                max_length=model_config.get("context_length", 2048)
            )
        else:
            # Initialize Ollama model
            self.llm = OllamaLLM(
                base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                model=self.current_model,
                temperature=model_config.get("temperature", 0.7),
                num_ctx=model_config.get("context_length", 2048)
            )
        
        return True
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models
        
        Returns:
            List[Dict[str, Any]]: List of available models with details
        """
        return self.model_supervisor.list_available_models()
    
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
            
            # Log the detected query type
            logger.info(f"Proceeding with detected query type: {query_type}")
            
            # Handle different query types
            try:
                if query_type == "FAQ" and not any(special in query.lower() for special in ["hello", "hi", "hey", "السلام", "مرحبا"]):
                    # Use the FAQ matching system for factual queries
                    logger.info("Using FAQ response system for factual query")
                    response = super().get_response(query, user_id)
                elif self.memory_enabled and len(history) > 1:
                    # Use memory-based conversation for continuity
                    logger.info("Using conversation memory for response")
                    
                    # First try with current model
                    try:
                        response = self.memory_chat_chain.invoke({
                            "language": language,
                            "history": formatted_history,
                            "query": query
                        })
                    except Exception as chain_error:
                        logger.error(f"Memory chain error: {chain_error}. Falling back to Ollama model.")
                        
                        # Try to switch to Ollama model if current model failed
                        if self.current_model != "llama3.1:8b":
                            old_model = self.current_model
                            logger.info(f"Attempting to switch to llama3.1:8b due to error with {old_model}")
                            
                            if self.configure_model("llama3.1:8b"):
                                response = self.memory_chat_chain.invoke({
                                    "language": language,
                                    "history": formatted_history,
                                    "query": query
                                })
                            else:
                                # If switch fails, use standard response
                                response = super().get_response(query, user_id)
                        else:
                            # Already using Ollama, fall back to standard response
                            response = super().get_response(query, user_id)
                else:
                    # Handle simple conversational queries without enough history
                    logger.info("Using standard conversation response (no sufficient memory)")
                    response = super().get_response(query, user_id)
            except Exception as e_response:
                logger.error(f"Error generating response: {e_response}")
                # Provide a generic fallback response
                if language == 'ar':
                    response = "عذراً، واجهت مشكلة فنية. يرجى المحاولة مرة أخرى."
                else:
                    response = "Sorry, I encountered a technical issue. Please try again."
            
            # Add assistant response to history if available
            if self.memory_enabled:
                self.memory.add_message(user_id, "assistant", response)
                
            return format_response(response, language)
            
        except Exception as e:
            logger.error(f"Error generating enhanced response: {e}", exc_info=True)
            
            # Provide a reliable fallback response
            if language == 'ar':
                fallback = "عذراً، هناك خطأ في النظام. يرجى المحاولة لاحقاً."
            else:
                fallback = "Sorry, there's a system error. Please try again later."
                
            return format_response(fallback, language)