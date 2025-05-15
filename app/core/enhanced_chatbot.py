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
from app.core.question_matcher import QuestionMatcher

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
        """Initialize enhanced chatbot with memory and model supervision"""
        # Call parent initializer first to ensure attributes are available
        super().__init__()
        
        # Create question matcher instance after super().__init__() is called
        from app.core.question_matcher import QuestionMatcher
        self.question_matcher = QuestionMatcher(similarity_threshold=self.similarity_threshold)
        
        # Initialize memory system
        self.memory_enabled = os.getenv('ENABLE_MEMORY', 'true').lower() == 'true'
        self.memory = ConversationMemory()
        
        # Initialize model supervisor
        self.model_supervisor = ModelSupervisor()
        self.current_model = os.getenv('MODEL_NAME', 'llama3.1:8b')
        
        # Question type classifier with bias toward FAQ lookup over conversation
        self.query_type_prompt = PromptTemplate(
            input_variables=["query", "language"],
            template="""Analyze the following user message and classify it as either FAQ (factual query) or CHAT (conversational).

User message: {query}
Language: {language}

Guidelines:
- Classify as FAQ if the query is asking for factual information, instructions, or specific details
- Classify as FAQ if it contains specific keywords or professional terminology
- Classify as FAQ if it's a question that would likely have a definitive answer
- Only classify as CHAT if it's purely conversational, casual, or social
- When in doubt, classify as FAQ to prioritize accurate information over conversation

Your response should be ONLY 'FAQ' or 'CHAT' with no additional text.

Classification:"""
        )
        
        # Set up the model
        self.configure_model(self.current_model)
        
        # Chain for determining query type
        self.query_type_chain = (
            RunnablePassthrough() | 
            self.query_type_prompt | 
            self.llm | 
            StrOutputParser()
        )
        
        # Memory-based conversational prompt with reduced history reliance
        self.memory_prompt = PromptTemplate(
            input_variables=["language", "history", "query", "wants_detail"],
            template="""You are a helpful assistant who {detail_level}.

Language: {language}

Previous conversation summary (use this for understanding context and helpful references):
{history}

Current user message: {query}

Important instructions:
{instructions}

Response:"""
        )
        
        # Chain for memory-based response
        self.memory_chat_chain = (
            RunnablePassthrough() | 
            self.memory_prompt | 
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
    
    def format_memory_prompt(self, language: str, history: str, query: str, wants_detail: bool) -> dict:
        """
        Format the memory prompt based on whether user wants details or not.
        
        Args:
            language (str): Language code
            history (str): Formatted conversation history
            query (str): User query
            wants_detail (bool): Whether user wants detailed response
            
        Returns:
            dict: Formatted prompt variables
        """
        if wants_detail:
            detail_level = "provides comprehensive and detailed information"
            instructions = """1. Use conversation history to provide context-aware responses
2. Provide an in-depth, thorough explanation
3. Explain key concepts comprehensively with examples
4. Include details from previous conversation when relevant
5. Organize your answer into sections if needed
6. Conclude with a summary of the main points"""
        else:
            detail_level = "prioritizes brevity and directness"
            instructions = """1. Be extremely concise - respond in 2-3 sentences maximum
2. Focus solely on answering the current query without any fluff
3. Only reference conversation history if absolutely necessary
4. Avoid all unnecessary words, explanations, or pleasantries
5. Provide only essential, factual information
6. Use clear, direct language"""
            
        return {
            "language": language,
            "history": history,
            "query": query,
            "wants_detail": wants_detail,
            "detail_level": detail_level,
            "instructions": instructions
        }
    
    def get_enhanced_response(self, query: str, user_id: str = "default") -> str:
        """
        Generate a response prioritizing exact question matches over conversation history.
        
        Process:
        1. Check for exact question match in database first
        2. Use conversation context only when necessary
        3. Fall back to standard response when appropriate
        
        Args:
            query (str): User's input query
            user_id (str): Unique identifier for the user
            
        Returns:
            str: Formatted response in the appropriate language
        """
        try:
            language = self.detect_language(query)
            logger.info(f"Processing query: '{query}' (Language: {language})")
            
            # Check if user is asking for more details
            wants_details = self.is_asking_for_detail(query, language)
            logger.info(f"User wants detailed response: {wants_details}")
            
            # Check for special commands
            if query.startswith("/model "):
                model_name = query[7:].strip()
                success = self.configure_model(model_name)
                if success:
                    return format_response(f"Model changed to {model_name}", language)
                else:
                    return format_response(f"Error changing model to {model_name}. Available models: {', '.join(self.model_supervisor.AVAILABLE_MODELS.keys())}", language)
                    
            elif query == "/models":
                models_list = self.list_available_models()
                models_text = "\n".join([f"- {m['name']} ({m['id']}): {m['description']}{' (current)' if m['current'] else ''}" for m in models_list])
                return format_response(f"Available models:\n{models_text}", language)
                
            elif query == "/clear":
                self.memory.clear_user_history(user_id)
                return format_response("Conversation history cleared", language)
            
            # Content moderation
            is_inappropriate, reason = moderate_content(query)
            if is_inappropriate:
                logger.warning(f"Inappropriate content detected: {reason}")
                response = self.inappropriate_chain.invoke({"language": language})
                return format_response(response, language)
            
            # STEP 1: First check for exact question matches from FAQ database
            faqs, scores = self.find_most_similar_faq(query, language)
            
            if faqs and scores:
                # Check for exact match using question_matcher
                for i, faq in enumerate(faqs[:5]):
                    candidate_q = faq.question_en if language == 'en' else faq.question_ar
                    if self.question_matcher.is_exact_match(query, candidate_q, language):
                        logger.info(f"Found exact question match: {candidate_q}")
                        # Return exact answer directly without using history
                        exact_answer = faq.answer_en if language == 'en' else faq.answer_ar
                        
                        # If user wants more details and it's not asking for details itself
                        if wants_details and not self.is_asking_for_detail(candidate_q, language):
                            logger.info("User wants more details on an exact match")
                            # Get the detailed response
                            if language == 'ar':
                                prompt = f"""أنت مساعد ذكي ومفيد وخبير. المستخدم سأل:
{query}

وجدت إجابة دقيقة في قاعدة البيانات:
{exact_answer}

المستخدم يطلب تفاصيل أكثر. الرجاء توسيع الإجابة الموجودة بتفاصيل وشرح أكثر.

تعليمات إضافية:
1. استخدم الإجابة الأصلية كأساس
2. أضف تفاصيل وأمثلة إضافية
3. اشرح أي مصطلحات فنية
4. قدم معلومات أكثر عمقاً حول الموضوع"""
                            else:
                                prompt = f"""You are an intelligent, helpful expert assistant. The user asked:
{query}

I found an exact answer in our database:
{exact_answer}

The user is requesting more detailed information. Please expand on the existing answer with more details and explanation.

Additional instructions:
1. Use the original answer as a foundation
2. Add additional details and examples
3. Explain any technical terms
4. Provide more in-depth information on the topic"""
                            
                            try:
                                # Generate detailed response using the model
                                detailed_answer = self.llm.invoke(prompt)
                                
                                # Add interaction to memory if enabled, but don't use it for response
                                if self.memory_enabled:
                                    self.memory.add_message(user_id, "user", query)
                                    self.memory.add_message(user_id, "assistant", detailed_answer)
                                    
                                return format_response(detailed_answer, language)
                            except Exception as e:
                                logger.error(f"Error generating detailed response: {e}")
                                # Fall back to the exact answer if there's an error
                                return format_response(exact_answer, language)
                        
                        # Add interaction to memory if enabled, but don't use it for response
                        if self.memory_enabled:
                            self.memory.add_message(user_id, "user", query)
                            self.memory.add_message(user_id, "assistant", exact_answer)
                            
                        return format_response(exact_answer, language)
            
            # Add user message to history for subsequent processing
            if self.memory_enabled:
                self.memory.add_message(user_id, "user", query)
                
            # STEP 2: Detect if this is a FAQ query or conversational query
            query_type = self.query_type_chain.invoke({
                "query": query,
                "language": language
            }).strip().upper()
            
            logger.info(f"Query type classification: {query_type}")
            
            # STEP 3: Handle different query types
            try:
                # FAQ queries should prioritize knowledge over conversation history
                if query_type == "FAQ" or "FAQ" in query_type:
                    logger.info("Using FAQ response system for factual query")
                    # Use the modified parent FAQ method
                    response = super().get_response(query, user_id)
                    
                # For chat queries with sufficient history, use memory
                elif self.memory_enabled:
                    history = self.memory.get_formatted_history(user_id)
                    
                    # Always use history if user wants details
                    if wants_details or len(history) > 2:
                        if wants_details:
                            logger.info("User wants detailed response with conversation context")
                        else:
                            logger.info("Using minimal conversation memory for response")
                        
                        # Format the history
                        formatted_history = self.format_history(history[-5:] if wants_details else history[-3:-1])
                        
                        # Generate response with appropriate detail level
                        prompt_vars = self.format_memory_prompt(
                            language=language,
                            history=formatted_history,
                            query=query,
                            wants_detail=wants_details
                        )
                        
                        # Generate response with memory context
                        response = self.memory_chat_chain.invoke(prompt_vars)
                    else:
                        # Not enough history, use standard response
                        logger.info("Not enough conversation history, using standard response")
                        response = super().get_response(query, user_id)
                else:
                    # Memory disabled, use standard response
                    logger.info("Memory disabled, using standard response")
                    response = super().get_response(query, user_id)
                    
            except Exception as e_response:
                logger.error(f"Error generating response: {e_response}")
                response = super().get_response(query, user_id)
            
            # Add assistant response to history if enabled
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