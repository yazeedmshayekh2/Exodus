"""
Test script to verify the updated embedding model works correctly.
"""

import os
import logging
import sys
from app.core.chatbot import FAQChatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_chatbot():
    """Test the chatbot with the new model."""
    try:
        # Log the current working directory
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Initialize the chatbot
        logger.info("Initializing chatbot with new embedding model...")
        chatbot = FAQChatbot()
        
        # Test with some sample queries
        test_queries = [
            "What are your business hours?",
            "How do I reset my password?",
            "كيف يمكنني إنشاء حساب جديد؟",  # How can I create a new account?
            "ما هي خدماتكم؟"  # What are your services?
        ]
        
        logger.info("Testing with sample queries...")
        for query in test_queries:
            language = chatbot.detect_language(query)
            logger.info(f"\nQuery: {query} (Language: {language})")
            
            # Find similar FAQs
            faqs, scores = chatbot.find_most_similar_faq(query, language)
            
            if faqs:
                logger.info(f"Found {len(faqs)} matching FAQs:")
                for i, (faq, score) in enumerate(zip(faqs, scores), 1):
                    logger.info(f"Match {i} (score: {score:.4f}):")
                    if language == 'en':
                        logger.info(f"Q: {faq.question_en}")
                        logger.info(f"A: {faq.answer_en[:100]}...")
                    else:
                        logger.info(f"Q: {faq.question_ar}")
                        logger.info(f"A: {faq.answer_ar[:100]}...")
            else:
                logger.info("No matching FAQs found.")
            
            # Test full response generation
            response = chatbot.get_response(query)
            logger.info(f"Response: {response[:150]}...")
            
            logger.info("-" * 50)
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        return False
    
    return True

if __name__ == "__main__":
    success = test_chatbot()
    sys.exit(0 if success else 1) 