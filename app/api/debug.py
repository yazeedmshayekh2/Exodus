"""
Debug API endpoints for monitoring and testing
"""

import logging
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from app import app
from app.api.chat import chatbot
from app.utils.server_info import get_server_info

# Configure logging
logger = logging.getLogger(__name__)

@app.get("/api/debug/faqs")
async def get_faq_count():
    """Debug endpoint to check loaded FAQs"""
    try:
        # Get all points from Qdrant
        points = chatbot.qdrant.scroll(
            collection_name=chatbot.collection_name,
            limit=10  # Get first 10 for debugging
        )[0]
        
        # Return detailed info
        return {
            "faq_count": len(points),
            "sample_faqs": [
                {
                    "question_en": p.payload["question_en"],
                    "question_ar": p.payload["question_ar"],
                }
                for p in points[:3]  # Show first 3 FAQs
            ] if points else []
        }
    except Exception as e:
        logger.error(f"Error checking FAQs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/debug/faq/{query}")
async def debug_faq_match(query: str):
    """Debug endpoint to test FAQ matching"""
    try:
        language = chatbot.detect_language(query)
        best_match, similarity = chatbot.find_most_similar_faq(query, language)
        threshold = chatbot.similarity_threshold[language]
        
        return {
            "query": query,
            "language": language,
            "similarity_score": similarity,
            "threshold": threshold,
            "exceeds_threshold": similarity > threshold,
            "best_match": {
                "question_en": best_match.question_en if best_match else None,
                "question_ar": best_match.question_ar if best_match else None,
                "answer_en": best_match.answer_en if best_match else None,
                "answer_ar": best_match.answer_ar if best_match else None
            } if best_match else None
        }
    except Exception as e:
        logger.error(f"Error in debug FAQ match: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/debug/server-info")
async def server_info_endpoint():
    """Get information about the server, including connection details."""
    try:
        return get_server_info()
    except Exception as e:
        logger.error(f"Error getting server info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(405)
async def method_not_allowed_handler(request, exc: HTTPException):
    """Handle method not allowed errors with a helpful message"""
    if request.url.path == "/api/chat":
        return JSONResponse(
            status_code=405,
            content={
                "detail": "This endpoint only accepts POST requests. Please use a POST request with JSON data."
            }
        )
    return JSONResponse(
        status_code=405,
        content={
            "detail": "Method not allowed"
        }
    ) 