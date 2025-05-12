"""
Base models for FAQ Chatbot
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from pydantic import BaseModel

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