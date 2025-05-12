"""
Core Chatbot implementation for FAQ retrieval and response generation
"""

import os
import logging
import uuid
import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langdetect import detect
import re

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from app.models.base import FAQEntry
from app.data.database import connect_to_database
from app.core.response import format_response, get_fallback_response
from app.utils.moderation import moderate_content

# Configure logging
logger = logging.getLogger(__name__)

class FAQChatbot:
    """
    Main chatbot class implementing FAQ matching and response generation.
    
    This class handles:
    - FAQ loading and embedding
    - Language detection
    - Semantic search
    - Response formatting
    - Natural language processing
    
    Attributes:
        embedding_model: SentenceTransformer model for semantic embeddings
        llm: Ollama language model for response generation
        qdrant: Vector database client for similarity search
        collection_name (str): Name of the Qdrant collection
        conversation_memory (dict): Storage for conversation history
        conversation_state (dict): Storage for conversation state
        similarity_threshold (dict): Threshold values for FAQ matching
    """
    
    FAQ_QUERY = """
    SELECT 
        q.QuestionEN AS Question_English, 
        q.QuestionAR AS Question_Arabic,
        a.AnswerEN AS Answer_English, 
        a.AnswerAR AS Answer_Arabic
    FROM agencyDB_Live.dbo.tblFAQQuestions AS q
    LEFT JOIN agencyDB_Live.dbo.tblFAQAnswer AS a
        ON q.ID = a.QuestionID
    WHERE q.isDeleted = 0 AND q.isVisible = 1
      AND (a.isDeleted = 0 AND a.isVisible = 1 OR a.QuestionID IS NULL)
    """

    def __init__(self):
        """Initialize the chatbot with required models and configurations."""
        # Set up cache file in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.embeddings_cache_file = os.path.join(script_dir, 'embeddings_cache.npz')
        
        # Initialize embedding model on CPU
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device='cpu')
        
        self.llm = OllamaLLM(
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            model=os.getenv('MODEL_NAME', 'llama3.1:8b'),
            temperature=0.5,
            top_p=0.95,
            num_ctx=2048,
            num_predict=-1
        )
        
        # Initialize Qdrant client in memory
        self.qdrant = QdrantClient(":memory:")
        self.collection_name = "faq_embeddings"
        
        # Create collection and load FAQs
        self.init_qdrant()
        self.load_faqs()
        
        # Intent classification prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Analyze the user's message and determine which type of message it is.
            Be very permissive and ensure common conversational phrases are properly handled.

            User message: {query}

            Guidelines:
            - ALWAYS classify "how are you", "how's it going", "what's up" as CHITCHAT
            - Questions about wellbeing, status, or other casual conversation should be CHITCHAT
            - Only classify as INAPPROPRIATE if it contains explicit profanity or clearly offensive content
            - Classify as GREETING only if it's a pure greeting with no question ("hello", "hi", etc.)
            - Classify as QUESTION if it asks for specific information or help
            
            Examples:
            "hello" -> GREETING
            "how are you" -> CHITCHAT
            "how are you doing" -> CHITCHAT
            "what's up" -> CHITCHAT
            "how is it going" -> CHITCHAT
            "tell me about your services" -> QUESTION
            
            Respond with one of the following categories:
            GREETING - if it's only a greeting or introduction with no question
            QUESTION - if it contains a request for specific information or help
            CHITCHAT - if it's casual conversation like "how are you", wellbeing questions, etc.
            INAPPROPRIATE - if it contains explicit profanity or clearly offensive content (use sparingly)
            
            When in doubt between QUESTION and CHITCHAT, classify as CHITCHAT for better conversation flow.
            
            Response:"""
        )
        
        # Intent classification chain
        self.intent_chain = (
            RunnablePassthrough() |
            self.intent_prompt |
            self.llm |
            StrOutputParser()
        )
        
        # FAQ response prompt
        self.faq_prompt = PromptTemplate(
            input_variables=["question_en", "answer_en", "question_ar", "answer_ar", 
                           "language", "user_query"],
            template="""You are a friendly and professional customer service assistant. Act naturally and respond directly to user queries. ONLY provide information based on the FAQ match provided.

            FAQ Match:
            EN Q: {question_en}
            EN A: {answer_en}
            AR Q: {question_ar}
            AR A: {answer_ar}

            User Query: {user_query}
            Language: {language}

            Instructions:
            1. If FAQ match exists (both question and answer are not empty):
               - Answer directly and naturally using ONLY the FAQ information
               - Keep the original information accurate
               - Use a conversational tone
               - DO NOT make up or invent information that is not in the FAQ
               - If the user's query asks for details not in the FAQ, state clearly that you don't have that specific information
               
            2. Format your response with proper line breaks and spacing:
               - Start with the title/heading
               - Add a blank line after the heading
               - Add a blank line before lists
               - Keep paragraphs concise
               - No email signatures or formal closings

            3. Use markdown formatting:
               - For titles: Use "**Title:**" format
               - For important terms: Use "**term**" with spaces
               - For phone numbers: Use "**1234 5678**" format
               - For lists: Use numbers (1. 2. 3.) or bullet points (*)

            4. Response structure:
               - Title/heading
               - Brief introduction if needed
               - Numbered steps or bullet points
               - Simple closing statement (optional)
               - NO signatures, names, or titles

            5. If NO FAQ match (empty question or answer):
               AR: "عذراً، لا أملك معلومات كافية حول هذا الموضوع. يمكنك التواصل مع فريق خدمة العملاء للحصول على المساعدة المتخصصة."
               EN: "I apologize, I don't have enough information about this topic. You can contact our customer service team for specialized assistance."

            Remember:
            - Be direct and concise
            - No email formatting or signatures
            - No meta-commentary
            - Keep responses focused
            - Always add proper spacing around bold text (** text **)
            - ONLY provide information from the FAQ match, never invent or assume details

            Response:"""
        )
        
        # Chitchat response prompt
        self.chitchat_prompt = PromptTemplate(
            input_variables=["language", "query"],
            template="""Generate a friendly, brief conversational response to the user's casual message.
            
            User message: {query}
            Language: {language}
            
            Guidelines:
            - For English: Be friendly, warm and conversational
            - For Arabic: Be culturally appropriate and friendly
            - ALWAYS respond appropriately to questions like "how are you", "what's up", etc.
            - If asked about wellbeing ("how are you"), RESPOND with your status (e.g., "I'm doing great, thanks for asking!")
            - Keep it simple and natural (1-2 sentences)
            - Be conversational and engaging
            - Don't ask complex questions
            - Don't provide specialized information
            - For Arabic, use formal but friendly Arabic
            
            Examples:
            If user asks "how are you" -> respond with "I'm doing great, thanks for asking! How can I help you today?"
            If user asks "what's up" -> respond with "Not much, just here to assist you! What can I help you with?"
            
            Response:"""
        )
        
        # How are you response prompt (specific for common wellbeing questions)
        self.how_are_you_prompt = PromptTemplate(
            input_variables=["language"],
            template="""Generate a friendly response to "how are you" or similar wellbeing questions.
            
            Language: {language}
            
            Guidelines:
            - Be warm, friendly and enthusiastic
            - ALWAYS respond with your status (e.g. "I'm doing well") 
            - Include a question about how you can help
            - Keep it simple (2 sentences maximum)
            - For Arabic, be culturally appropriate but friendly
            
            Output ONLY the response, nothing else.
            
            Response:"""
        )
        
        # Inappropriate content response prompt
        self.inappropriate_prompt = PromptTemplate(
            input_variables=["language"],
            template="""Generate a polite but firm response declining to engage with inappropriate content.
            
            Language: {language}
            
            Guidelines:
            - Be polite but direct
            - Do not repeat or reference the inappropriate content
            - Redirect to appropriate topics
            - For Arabic, use formal Arabic
            - Keep it brief (1-2 sentences)
            
            Response:"""
        )
        
        # Greeting response prompt
        self.greeting_prompt = PromptTemplate(
            input_variables=["language"],
            template="""Generate a friendly greeting response in the appropriate language.
            
            Language: {language}
            
            Guidelines:
            - For English: Be friendly and professional
            - For Arabic: Be culturally appropriate and professional (use appropriate Islamic greetings if in Arabic)
            - Keep it simple and natural
            - Don't add meta-commentary
            - Don't ask specific questions
            
            Response:"""
        )
        
        # Create chains
        self.faq_chain = (
            RunnablePassthrough() | 
            self.faq_prompt | 
            self.llm | 
            StrOutputParser()
        )
        
        self.greeting_chain = (
            RunnablePassthrough() |
            self.greeting_prompt |
            self.llm |
            StrOutputParser()
        )
        
        self.chitchat_chain = (
            RunnablePassthrough() |
            self.chitchat_prompt |
            self.llm |
            StrOutputParser()
        )
        
        self.inappropriate_chain = (
            RunnablePassthrough() |
            self.inappropriate_prompt |
            self.llm |
            StrOutputParser()
        )
        
        self.how_are_you_chain = (
            RunnablePassthrough() |
            self.how_are_you_prompt |
            self.llm |
            StrOutputParser()
        )
        
        # Initialize state and memory
        self.conversation_memory = {}
        self.conversation_state = {}
        self.similarity_threshold = {
            'ar': 0.20,  # Much lower threshold for Arabic to be more permissive
            'en': 0.25   # Lower threshold for English to increase match rate
        }
    
    def init_qdrant(self):
        """Initialize Qdrant collection in memory"""
        try:
            # Create new collection
            self.qdrant.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,  # Dimension of MPNet embeddings
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created new in-memory Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing in-memory Qdrant: {e}")
            raise
    
    def save_embeddings_cache(self, points):
        """
        Save embeddings and metadata to cache file.
        
        Args:
            points (List[PointStruct]): List of points with embeddings and metadata
        """
        try:
            cache_data = {
                'vectors': np.array([p.vector for p in points]),
                'ids': [p.id for p in points],
                'payloads': [p.payload for p in points]
            }
            np.savez_compressed(self.embeddings_cache_file, **cache_data)
            logger.info(f"Saved {len(points)} embeddings to cache file")
        except Exception as e:
            logger.error(f"Error saving embeddings cache: {e}")
            raise
    
    def load_embeddings_cache(self):
        """
        Load embeddings and metadata from cache file.
        
        Returns:
            List[PointStruct]: List of points with embeddings and metadata
        """
        try:
            if not os.path.exists(self.embeddings_cache_file):
                logger.info("No embeddings cache file found")
                return None
                
            cache_data = np.load(self.embeddings_cache_file, allow_pickle=True)
            points = [
                models.PointStruct(
                    id=str(id_),
                    vector=vector.tolist(),
                    payload=dict(payload)  # Convert numpy object to dict
                )
                for id_, vector, payload in zip(
                    cache_data['ids'],
                    cache_data['vectors'],
                    cache_data['payloads']
                )
            ]
            logger.info(f"Loaded {len(points)} embeddings from cache file")
            return points
        except Exception as e:
            logger.error(f"Error loading embeddings cache: {e}")
            return None
    
    def load_faqs(self):
        """
        Load FAQs from database and store in Qdrant.
        First tries to load from cache, if not available processes from database.
        """
        try:
            # Check if cache file exists and try to load it
            if os.path.exists(self.embeddings_cache_file):
                cached_points = self.load_embeddings_cache()
                if cached_points:
                    # Verify points have required fields
                    valid_points = []
                    for point in cached_points:
                        if all(key in point.payload for key in ['question_en', 'question_ar', 'answer_en', 'answer_ar']):
                            # Normalize vector
                            vector = np.array(point.vector)
                            vector = vector / np.linalg.norm(vector)
                            point.vector = vector.tolist()
                            valid_points.append(point)
                        else:
                            logger.error(f"Point missing required fields: {point.payload.keys()}")
                    
                    if valid_points:
                        # Clear existing collection
                        self.init_qdrant()
                        
                        # Insert all points
                        batch_size = 100
                        for i in range(0, len(valid_points), batch_size):
                            batch = valid_points[i:i + batch_size]
                            self.qdrant.upsert(
                                collection_name=self.collection_name,
                                points=batch
                            )
                        
                        total_points = len(self.qdrant.scroll(
                            collection_name=self.collection_name,
                            limit=1000
                        )[0])
                        
                        logger.info(f"Successfully loaded {total_points} FAQs into Qdrant")
                        return
                    else:
                        logger.error("No valid points found in cache")
                        
            # If we get here, either no cache exists or loading failed
            logger.info("Loading FAQs from database...")
            connection = connect_to_database()
            try:
                cursor = connection.cursor()
                rows = cursor.execute(self.FAQ_QUERY).fetchall()
                logger.info(f"Retrieved {len(rows)} rows from database")
                
                points = []
                for i, row in enumerate(rows, 1):
                    try:
                        if all([row.Question_English, row.Question_Arabic, 
                               row.Answer_English, row.Answer_Arabic]):
                            # Create embeddings
                            en_embedding = self.embedding_model.encode(row.Question_English.strip())
                            ar_embedding = self.embedding_model.encode(row.Question_Arabic.strip())
                            
                            # Average and normalize
                            combined_embedding = (en_embedding + ar_embedding) / 2
                            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
                            
                            points.append(models.PointStruct(
                                id=str(uuid.uuid4()),
                                vector=combined_embedding.tolist(),
                                payload={
                                    'question_en': row.Question_English.strip(),
                                    'question_ar': row.Question_Arabic.strip(),
                                    'answer_en': row.Answer_English.strip(),
                                    'answer_ar': row.Answer_Arabic.strip()
                                }
                            ))
                            if i % 10 == 0:
                                logger.info(f"Processed {i} FAQs...")
                    except Exception as e:
                        logger.error(f"Error processing FAQ {i}: {e}")
                        continue
                
                if points:
                    # Save to cache
                    self.save_embeddings_cache(points)
                    
                    # Load into Qdrant
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    logger.info(f"Successfully loaded {len(points)} FAQs into Qdrant")
                else:
                    logger.warning("No valid FAQs were loaded")
                    
            finally:
                connection.close()
                
        except Exception as e:
            logger.error(f"Error in load_faqs: {e}", exc_info=True)
            raise
            
    def find_most_similar_faq(self, query: str, language: str = 'en') -> Tuple[FAQEntry, float]:
        """
        Find the most similar FAQ to the user's query.
        
        Args:
            query (str): User's input query
            language (str): Language of the query ('en' or 'ar')
            
        Returns:
            Tuple[Optional[FAQEntry], float]: The best matching FAQ and its similarity score
        """
        try:
            if not query:
                return None, 0

            # Generate and normalize query embedding
            query_embedding = self.embedding_model.encode(query)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Debug: Print threshold
            threshold = self.similarity_threshold[language]  # Use instance variable directly
            logger.info(f"Current threshold for {language}: {threshold}")
            
            # Search in Qdrant
            search_result = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=5,  # Increased limit for better matching
                score_threshold=0.0  # Remove score threshold to see all results
            )
            
            if not search_result:
                logger.info("No FAQ matches found in Qdrant")
                return None, 0
            
            # Get the best match
            best_match = search_result[0]
            
            # Debug: Print all matches
            logger.info("=== Search Results ===")
            for i, match in enumerate(search_result):
                logger.info(f"Match {i+1}:")
                logger.info(f"Score: {match.score:.4f}")
                logger.info(f"Q(EN): {match.payload['question_en']}")
                logger.info(f"Q(AR): {match.payload['question_ar']}")
                logger.info("---")
            
            # Create FAQEntry from best match
            faq = FAQEntry(
                question_en=best_match.payload['question_en'],
                question_ar=best_match.payload['question_ar'],
                answer_en=best_match.payload['answer_en'],
                answer_ar=best_match.payload['answer_ar']
            )
            
            logger.info(f"Best match score: {best_match.score} (Threshold: {threshold})")
            logger.info(f"Will use FAQ: {best_match.score > threshold}")
            
            return faq, best_match.score
            
        except Exception as e:
            logger.error(f"Error in find_most_similar_faq: {e}", exc_info=True)
            return None, 0
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            str: Language code ('en' or 'ar')
        """
        try:
            # Clean and prepare text
            text = text.strip()
            if not text:
                return 'en'  # Default to English for empty text
            
            # Arabic character detection (more reliable for Arabic than langdetect)
            # Check for Arabic characters (standard Arabic Unicode range)
            arabic_char_count = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
            
            # Check for additional Arabic-related character ranges
            # Arabic Supplement and Extended-A
            arabic_extended_count = sum(1 for c in text if '\u0750' <= c <= '\u077F' or '\u08A0' <= c <= '\u08FF')
            
            # Arabic Presentation Forms
            arabic_pres_count = sum(1 for c in text if '\uFB50' <= c <= '\uFDFF' or '\uFE70' <= c <= '\uFEFF')
            
            # Total Arabic character count
            total_arabic = arabic_char_count + arabic_extended_count + arabic_pres_count
            
            # If significant Arabic content (at least 20% of characters or at least 2 chars in short texts)
            if total_arabic >= max(0.2 * len(text), 2):
                return 'ar'
            
            # Try langdetect for non-Arabic text
            try:
                detected = detect(text)
                if detected == 'ar':
                    return 'ar'
                elif detected in ['en', 'cy']:  # cy sometimes detected for English
                    return 'en'
            except:
                pass  # Continue to fallback if langdetect fails
            
            # Common Arabic transliterated/Arabizi words (for mixed Arabic/English text)
            arabizi_patterns = [
                r'\b(salam|ahlan|marhaba|shukran|afwan|yalla|habibi|habibti|inshallah|mashallah|wallah)\b',
                r'\b(sabah|masa|alkhair|alnour|assalamu|alaikum|mabrook|mabruk|alhamdulillah)\b',
                r'\b(ana|anta|anti|howa|heya|3an|min|ila|bil|fee|ma3|bas|khalas|akeed|tab|tayeb)\b'
            ]
            
            # Check for Arabizi patterns with Arabic numerals (3=ع, 7=ح, 5=خ, etc.)
            for pattern in arabizi_patterns:
                if re.search(pattern, text.lower()):
                    return 'ar'  # Consider this Arabic if Arabizi detected
            
            # Default to English if nothing else matched
            return 'en'
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}. Defaulting to English.")
            return 'en'
            
    def get_response(self, query: str, user_id: str = "default") -> str:
        """
        Generate a response for the user's query.
        
        Args:
            query (str): User's input query
            user_id (str): Unique identifier for the user
            
        Returns:
            str: Formatted response in the appropriate language
        """
        try:
            language = self.detect_language(query)
            logger.info(f"Processing query: '{query}' (Language: {language})")
            
            # Check for common conversational phrases directly
            lower_query = query.lower().strip()
            
            # How are you phrases (check these first and give priority)
            how_are_you_phrases = [
                "how are you", "how are you doing", "how are you today", "how's it going", 
                "what's up", "how do you do", "how have you been", "how is everything"
            ]
            
            arabic_how_are_you = [
                "كيف حالك", "كيفك", "شلونك", "اخبارك", "كيف الحال", "عامل ايه", "ازيك"
            ]
            
            # Other casual chitchat phrases
            other_chitchat = [
                "nice to meet you", "good to see you", "what can you do", 
                "tell me about yourself", "who are you"
            ]
            
            arabic_other_chitchat = [
                "تشرفنا", "سعيد بلقائك", "ماذا يمكنك أن تفعل", "من أنت", "عرفني بنفسك"
            ]
            
            # Handle "how are you" with dedicated response
            is_how_are_you = False
            if any(phrase in lower_query for phrase in how_are_you_phrases):
                logger.info("Direct detection of 'how are you' phrase")
                is_how_are_you = True
            elif any(phrase in query for phrase in arabic_how_are_you):
                logger.info("Direct detection of Arabic 'how are you' phrase")
                is_how_are_you = True
                
            if is_how_are_you:
                logger.info("Generating 'how are you' response")
                
                # Hardcoded responses for "how are you" to ensure consistent quality
                if language == 'ar':
                    hardcoded_responses = [
                        "أنا بخير، شكراً على سؤالك! كيف يمكنني مساعدتك اليوم؟",
                        "الحمد لله، أنا بخير. سعيد بالتحدث معك. كيف يمكنني خدمتك؟",
                        "بخير والحمد لله! أنا هنا لمساعدتك في أي استفسار."
                    ]
                    response = hardcoded_responses[hash(query) % len(hardcoded_responses)]
                else:
                    hardcoded_responses = [
                        "I'm doing great, thanks for asking! How can I help you today?",
                        "I'm very well, thank you! What can I assist you with?",
                        "I'm doing well! I'm here to help you with any questions you might have."
                    ]
                    response = hardcoded_responses[hash(query) % len(hardcoded_responses)]
                
                return format_response(response, language)
            
            # Handle other chitchat
            is_other_chitchat = False
            if any(phrase in lower_query for phrase in other_chitchat):
                logger.info("Direct detection of other chitchat phrase")
                is_other_chitchat = True
            elif any(phrase in query for phrase in arabic_other_chitchat):
                logger.info("Direct detection of Arabic chitchat phrase")
                is_other_chitchat = True
            
            if is_other_chitchat:
                logger.info("Processing other chitchat response")
                response = self.chitchat_chain.invoke({"language": language, "query": query})
                return format_response(response, language)
            
            # More permissive moderation check - only block highly confident detections
            is_inappropriate, reason = moderate_content(query)
            if is_inappropriate and reason == "profanity":  # Only block explicit profanity
                logger.warning(f"Inappropriate content detected: {reason}")
                response = self.inappropriate_chain.invoke({"language": language})
                return format_response(response, language)
            
            # Determine intent
            intent = self.intent_chain.invoke({"query": query}).strip().upper()
            logger.info(f"Detected intent: {intent}")
            
            # Handle different intent types
            if intent == "GREETING":
                logger.info("Processing greeting response")
                response = self.greeting_chain.invoke({"language": language})
                return format_response(response, language)
            
            elif intent == "CHITCHAT":
                logger.info("Processing chitchat response")
                response = self.chitchat_chain.invoke({"language": language, "query": query})
                return format_response(response, language)
                
            elif intent == "INAPPROPRIATE":
                # Double-check with our moderation before rejecting
                if is_inappropriate:
                    logger.info("Processing response to inappropriate content")
                    response = self.inappropriate_chain.invoke({"language": language})
                    return format_response(response, language)
                else:
                    # Override with question handling if our moderation didn't flag it
                    intent = "QUESTION"
                    logger.info("Overriding INAPPROPRIATE intent to QUESTION")
            
            # Process as FAQ query (QUESTION intent or default)
            logger.info("Searching for FAQ matches...")
            best_match, similarity = self.find_most_similar_faq(query, language)
            threshold = self.similarity_threshold[language]
            
            # Debug logging
            logger.info(f"Query: '{query}'")
            logger.info(f"Language: {language}")
            logger.info(f"Similarity score: {similarity}")
            logger.info(f"Using threshold: {threshold} for language: {language}")
            
            # If we have a match (even a low confidence one), try to use it
            if best_match:
                # For stronger matches, use the FAQ directly
                if similarity > threshold:
                    logger.info("Found good matching FAQ - generating direct response")
                    logger.info(f"Matched Question (EN): {best_match.question_en}")
                    logger.info(f"Matched Question (AR): {best_match.question_ar}")
                    
                    # Use the chain to process the response
                    chain_input = {
                        "question_en": best_match.question_en,
                        "answer_en": best_match.answer_en,
                        "question_ar": best_match.question_ar,
                        "answer_ar": best_match.answer_ar,
                        "language": language,
                        "user_query": query
                    }
                    response = self.faq_chain.invoke(chain_input)
                    logger.info("Generated direct FAQ response")
                    return format_response(response, language)
                
                # For weaker matches, provide a modified response that indicates uncertainty
                elif similarity > (threshold * 0.7):  # Lower secondary threshold
                    logger.info("Found partial FAQ match - generating qualified response")
                    
                    # Generate a "not sure but here's what I found" response
                    uncertain_prompt = PromptTemplate(
                        input_variables=["question_en", "answer_en", "question_ar", "answer_ar", 
                                      "language", "user_query"],
                        template="""You are a helpful customer service assistant. The user's question doesn't exactly match our FAQs,
                        but there is a somewhat related answer that might be useful.
                        
                        Most similar FAQ:
                        EN Q: {question_en}
                        EN A: {answer_en}
                        AR Q: {question_ar}
                        AR A: {answer_ar}
                        
                        User query: {user_query}
                        Language: {language}
                        
                        Provide a response that:
                        1. Acknowledges that you don't have an exact answer to their specific question
                        2. Offers the related information that might be helpful
                        3. Is conversational and helpful, not formal
                        4. Uses the FAQ content but clarifies this isn't an exact match for their question
                        
                        For Arabic, be culturally appropriate.
                        
                        Response:"""
                    )
                    
                    uncertain_chain = (
                        RunnablePassthrough() | 
                        uncertain_prompt | 
                        self.llm | 
                        StrOutputParser()
                    )
                    
                    # Create input with the partial match
                    chain_input = {
                        "question_en": best_match.question_en,
                        "answer_en": best_match.answer_en,
                        "question_ar": best_match.question_ar,
                        "answer_ar": best_match.answer_ar,
                        "language": language,
                        "user_query": query
                    }
                    
                    response = uncertain_chain.invoke(chain_input)
                    logger.info("Generated qualified response from partial match")
                    return format_response(response, language)
            
            # For casual conversation that's not a greeting, provide a conversational response
            if len(query.split()) < 5 or "?" not in query:
                logger.info("Short query without question mark - treating as conversation")
                response = self.chitchat_chain.invoke({"language": language, "query": query})
                return format_response(response, language)
            
            # If we get here, no good match was found
            logger.info(f"No matching FAQ found (score: {similarity} < threshold: {threshold})")
            response = get_fallback_response(language)
            return format_response(response, language)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return format_response(get_fallback_response(language), language) 