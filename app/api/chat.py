"""
Chat API endpoints for FAQ chatbot
"""

import logging
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any

from app import app
from app.models.base import ChatRequest, ChatResponse
from app.core.enhanced_chatbot import EnhancedChatbot

# Configure logging
logger = logging.getLogger(__name__)

# Initialize enhanced chatbot instance
chatbot = EnhancedChatbot()

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
        
        # Use enhanced chatbot response with memory
        response = chatbot.get_enhanced_response(query)
        logger.info(f"Generated response: {response[:100]}...")  # Log first 100 chars
        
        return ChatResponse(response=response, language=language)
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """
    Get list of available models that can be used with the chatbot
    
    Returns:
        List[Dict[str, Any]]: List of available models with details
    """
    try:
        models = chatbot.list_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/switch")
async def switch_model(model_data: Dict[str, str]):
    """
    Switch the active model used by the chatbot
    
    Args:
        model_data (Dict[str, str]): Dictionary containing the model_id
        
    Returns:
        Dict[str, Any]: Result of the model switch operation
    """
    try:
        model_id = model_data.get("model_id")
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id is required")
            
        success = chatbot.configure_model(model_id)
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to switch to model: {model_id}")
            
        return {
            "success": True,
            "message": f"Successfully switched to model: {model_id}",
            "current_model": chatbot.current_model
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 