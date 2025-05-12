#!/usr/bin/env python3
"""
Test script for the chatbot with focus on Arabic support
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to sys.path to import app modules
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Apply SSL patch before importing other modules
try:
    from app.utils.asyncio_patch import apply_asyncio_patch
    apply_asyncio_patch()
    print("Applied SSL compatibility patch")
except Exception as e:
    print(f"Warning: Could not apply SSL patch: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("chat_test")

# Now import the chatbot
try:
    from app.core.chatbot import FAQChatbot
except ImportError as e:
    logger.error(f"Failed to import FAQChatbot: {e}")
    sys.exit(1)

def test_arabic_response():
    """Test basic Arabic response generation"""
    chatbot = FAQChatbot()
    
    # Test Arabic language detection
    test_text_ar = "كيف حالك؟"
    language = chatbot.detect_language(test_text_ar)
    logger.info(f"Detected language for '{test_text_ar}': {language}")
    assert language == 'ar', "Failed to detect Arabic language"
    
    # Test Arabic greeting
    response = chatbot.get_response(test_text_ar)
    logger.info(f"Response to '{test_text_ar}':\n{response}")
    
    # Test more complex Arabic query
    test_query_ar = "هل يمكنك مساعدتي في معرفة خدماتكم؟"
    response = chatbot.get_response(test_query_ar)
    logger.info(f"Response to '{test_query_ar}':\n{response}")
    
    return True

def test_english_response():
    """Test basic English response generation"""
    chatbot = FAQChatbot()
    
    # Test English language detection
    test_text_en = "How are you today?"
    language = chatbot.detect_language(test_text_en)
    logger.info(f"Detected language for '{test_text_en}': {language}")
    assert language == 'en', "Failed to detect English language"
    
    # Test English greeting
    response = chatbot.get_response(test_text_en)
    logger.info(f"Response to '{test_text_en}':\n{response}")
    
    # Test more complex English query
    test_query_en = "Can you tell me about your services?"
    response = chatbot.get_response(test_query_en)
    logger.info(f"Response to '{test_query_en}':\n{response}")
    
    return True

def test_moderation():
    """Test content moderation for both languages"""
    chatbot = FAQChatbot()
    
    # Test Arabic moderation
    test_bad_ar = "انت كلب وحمار"  # Mild insult in Arabic
    response = chatbot.get_response(test_bad_ar)
    logger.info(f"Moderation response to '{test_bad_ar}':\n{response}")
    
    # Test English moderation
    test_bad_en = "You are stupid and dumb"
    response = chatbot.get_response(test_bad_en)
    logger.info(f"Moderation response to '{test_bad_en}':\n{response}")
    
    return True

def run_all_tests():
    """Run all tests"""
    logger.info("Starting chatbot tests with focus on Arabic support")
    
    tests = [
        ("Arabic Response", test_arabic_response),
        ("English Response", test_english_response),
        ("Moderation", test_moderation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"Running test: {test_name}")
        try:
            result = test_func()
            logger.info(f"Test {test_name}: {'PASSED' if result else 'FAILED'}")
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} raised exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    logger.info("\nTest Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    logger.info(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status} - {test_name}")
    
    return passed == total

if __name__ == "__main__":
    # First, setup Arabic support
    try:
        from app.utils.setup_arabic import setup_arabic_support
        setup_arabic_support()
    except Exception as e:
        logger.warning(f"Failed to run Arabic setup: {e}")
    
    # Run tests
    success = run_all_tests()
    sys.exit(0 if success else 1) 