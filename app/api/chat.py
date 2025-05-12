"""
Chat API endpoints for FAQ chatbot
"""

import logging
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from app import app
from app.models.base import ChatRequest, ChatResponse
from app.core.chatbot import FAQChatbot

# Configure logging
logger = logging.getLogger(__name__)

# Initialize chatbot instance
chatbot = FAQChatbot()

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Process chat requests and return responses.
    
    Args:
        chat_request (ChatRequest): The chat request containing the user's query
        
    Returns:
        ChatResponse: The formatted response with detected language
        
    Raises:
        HTTPException: If request processing fails
    """
    try:
        query = chat_request.query.strip()
        logger.info(f"Received chat request: {query}")
        
        if not query:
            logger.warning("Empty query received")
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        language = chatbot.detect_language(query)
        logger.info(f"Detected language: {language}")
        
        response = chatbot.get_response(query)
        logger.info(f"Generated response: {response[:100]}...")  # Log first 100 chars
        
        return ChatResponse(response=response, language=language)
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 