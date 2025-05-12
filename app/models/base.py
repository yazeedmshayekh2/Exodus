"""
Base models for FAQ Chatbot
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from pydantic import BaseModel, Field

@dataclass
class FAQEntry:
    """
    Data class representing a single FAQ entry with bilingual content.
    
    Attributes:
        question_en (str): The question in English
        question_ar (str): The question in Arabic
        answer_en (str): The answer in English
        answer_ar (str): The answer in Arabic
        embedding (Optional[np.ndarray]): Vector embedding of the FAQ content
    """
    question_en: str
    question_ar: str
    answer_en: str
    answer_ar: str
    embedding: Optional[np.ndarray] = None

class ChatRequest(BaseModel):
    """
    Pydantic model for chat request validation.
    
    Attributes:
        query (str): The user's input query
    """
    query: str

class ChatResponse(BaseModel):
    """
    Pydantic model for chat response validation.
    
    Attributes:
        response (str): The chatbot's response
        language (str): The detected language of the response ('en' or 'ar')
    """
    response: str
    language: str

class ModelIdRequest(BaseModel):
    """
    Request model for operations that only need a model ID.
    
    Attributes:
        model_id (str): ID of the model to operate on
    """
    model_id: str = Field(..., description="ID of the model to operate on")

class HuggingFaceModelRequest(BaseModel):
    """
    Request model for adding a model from HuggingFace.
    
    Attributes:
        repo_id (str): HuggingFace repository ID
        model_name (str): Name to use for the model in Ollama
        display_name (str): Human-readable name for the model
        description (str): Description of the model
        context_length (int): Context window size
        temperature (float): Default temperature
    """
    repo_id: str = Field(..., description="HuggingFace repository ID")
    model_name: str = Field(..., description="Name to use for the model in Ollama")
    display_name: str = Field(..., description="Human-readable name for the model")
    description: str = Field(..., description="Description of the model")
    context_length: int = Field(4096, description="Context window size")
    temperature: float = Field(0.7, description="Default temperature")