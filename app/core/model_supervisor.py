"""
Model Supervisor for managing and switching between different LLM models
"""

import os
import logging
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class ModelSupervisor:
    """
    A supervisor class that manages different models and controls model switching.
    
    This class provides:
    - Centralized model management
    - Unified interface for model switching
    - Proper initialization with model-specific settings
    - Error handling and fallback mechanisms
    """
    
    # Model configurations
    AVAILABLE_MODELS = {
        "llama3.1:8b": {
            "name": "Llama 3.1 8B",
            "description": "Meta's Llama 3.1 8B model for general purpose tasks",
            "context_length": 8192,
            "temperature": 0.7,
            "model_type": "ollama"
        },
        "Qwen/Qwen2.5-7B-Instruct-AWQ": {
            "name": "Qwen2.5 7B Instruct AWQ",
            "description": "Alibaba's Qwen2.5 7B Instruct model with AWQ 4-bit quantization",
            "context_length": 32768,
            "temperature": 0.6,
            "model_type": "huggingface"
        },
        "yazeed-mshayekh/Exodus-Arabic-Model": {
            "name": "Exodus Arabic Model",
            "description": "Bilingual FAQ chatbot model optimized for English and Arabic queries",
            "context_length": 32768,
            "temperature": 0.6,
            "model_type": "huggingface",
            "awq_config_path": "awq_config.json"  # Using the existing AWQ config
        },
        "fine-tuned-model": {
            "name": "Fine-tuned FAQ Model",
            "description": "Custom fine-tuned model for FAQ handling",
            "context_length": 32768,
            "temperature": 0.6,
            "model_type": "huggingface",
            "model_path": "/home/user/Desktop/Test/Exodus/Fine-tune/merged_model"
        }
    }
    
    def __init__(self):
        """Initialize the ModelSupervisor"""
        self.current_model_name = os.getenv('MODEL_NAME', 'llama3.1:8b')
        self.current_model = None
        self.current_tokenizer = None
        self.model_cache = {}  # Cache for loaded models
        self.loaded_models = set()  # Track which models have been loaded
        
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
                "current": model_id == self.current_model_name,
                "model_type": config.get("model_type", "ollama")
            }
            for model_id, config in self.AVAILABLE_MODELS.items()
        ]
    
    def get_current_model(self) -> str:
        """
        Get the name of the currently active model
        
        Returns:
            str: Name of the current model
        """
        return self.current_model_name
    
    def load_qwen_awq(self) -> tuple:
        """
        Load the Qwen2.5-7B-Instruct-AWQ model using the recommended approach
        
        Returns:
            tuple: (model, tokenizer)
        """
        model_name = "Qwen/Qwen2.5-7B-Instruct-AWQ"
        logger.info(f"Loading {model_name} using recommended approach")
        
        try:
            # Load using the approach from the model card
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            logger.info(f"Successfully loaded {model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            # Try non-AWQ version as fallback
            fallback_model = "Qwen/Qwen2.5-7B-Instruct"
            logger.warning(f"Attempting fallback to {fallback_model}")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                model = AutoModelForCausalLM.from_pretrained(
                    fallback_model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info(f"Successfully loaded fallback model {fallback_model}")
                return model, tokenizer
            except Exception as e2:
                logger.error(f"Error loading fallback model: {e2}")
                raise ValueError(f"Failed to load both primary and fallback models: {e}, {e2}")
    
    def generate_response(self, prompt: str, messages: List[Dict[str, str]] = None, 
                          max_new_tokens: int = 512) -> str:
        """
        Generate a response using the currently loaded model
        
        Args:
            prompt (str): Raw prompt text (used if messages not provided)
            messages (List[Dict[str, str]], optional): List of message dicts with role and content
            max_new_tokens (int, optional): Maximum number of tokens to generate
            
        Returns:
            str: Generated response
        """
        if self.current_model is None or self.current_tokenizer is None:
            logger.warning("No model loaded. Loading default model.")
            self.switch_model(self.current_model_name)
            
        try:
            model_type = self.AVAILABLE_MODELS[self.current_model_name].get("model_type")
            
            if model_type == "huggingface":
                # For HuggingFace models
                if messages:
                    # Process as a chat with template
                    text = self.current_tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    # Use raw prompt
                    text = prompt
                    
                model_inputs = self.current_tokenizer([text], return_tensors="pt").to(self.current_model.device)
                
                generated_ids = self.current_model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens
                )
                
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                
                response = self.current_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return response
                
            else:
                # For Ollama models
                # We'll need to implement Ollama-specific handling here if needed
                # Or we could call back to the EnhancedChatbot to handle this
                logger.error("Ollama models not directly supported by the supervisor yet")
                raise NotImplementedError("Ollama model generation not implemented in supervisor")
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error when trying to generate a response. Please try again or try a different model."
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different model
        
        Args:
            model_name (str): Name of the model to switch to
            
        Returns:
            bool: Whether the switch was successful
        """
        if model_name not in self.AVAILABLE_MODELS:
            logger.warning(f"Unknown model: {model_name}")
            return False
            
        logger.info(f"Switching to model: {model_name}")
        
        # If model is already loaded and cached, use it
        if model_name in self.model_cache:
            logger.info(f"Using cached model: {model_name}")
            self.current_model, self.current_tokenizer = self.model_cache[model_name]
            self.current_model_name = model_name
            return True
        
        # Otherwise load the model
        try:
            # Handle different model types
            model_type = self.AVAILABLE_MODELS[model_name].get("model_type")
            
            if model_type == "huggingface":
                # Check if it's the Qwen2.5 AWQ model
                if model_name == "Qwen/Qwen2.5-7B-Instruct-AWQ":
                    model, tokenizer = self.load_qwen_awq()
                else:
                    # Generic HuggingFace model loading
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                
                # Cache the model
                self.model_cache[model_name] = (model, tokenizer)
                self.current_model = model
                self.current_tokenizer = tokenizer
                self.current_model_name = model_name
                self.loaded_models.add(model_name)
                
                return True
                
            else:
                # Ollama models - we don't directly load these
                # Just update the current model name, actual loading will be 
                # handled by EnhancedChatbot
                self.current_model_name = model_name
                self.current_model = None
                self.current_tokenizer = None
                return True
                
        except Exception as e:
            logger.error(f"Error switching to model {model_name}: {e}")
            # Try to fall back to another model if possible
            if self.current_model is not None:
                logger.info(f"Keeping current model: {self.current_model_name}")
                return False
            elif "llama3.1:8b" != model_name:
                # Try to fall back to Llama
                logger.info(f"Falling back to llama3.1:8b model")
                return self.switch_model("llama3.1:8b")
            else:
                logger.error("Failed to load any model")
                return False
    
    def cleanup(self):
        """
        Clean up resources when shutting down
        """
        for model_name, (model, _) in self.model_cache.items():
            logger.info(f"Cleaning up model: {model_name}")
            del model
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model_cache = {}
        self.loaded_models = set()
        self.current_model = None
        self.current_tokenizer = None 