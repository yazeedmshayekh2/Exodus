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
            template="""Analyze the user's message and determine if it's a greeting/conversation starter or a question/request.
            
            User message: {query}
            
            Respond with either:
            GREETING - if it's a greeting or conversation starter
            QUESTION - if it's a question or request for information
            
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
            template="""You are a friendly and professional customer service assistant. Act naturally and respond directly to user queries.

            FAQ Match:
            EN Q: {question_en}
            EN A: {answer_en}
            AR Q: {question_ar}
            AR A: {answer_ar}

            User Query: {user_query}
            Language: {language}

            Instructions:
            1. If FAQ match exists (both question and answer are not empty):
               - Answer directly and naturally using the FAQ information
               - Keep the original information accurate
               - Use a conversational tone
               
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
               AR: "عذراً، أقترح التواصل مع فريق خدمة العملاء للحصول على المساعدة المتخصصة."
               EN: "I apologize, I recommend contacting our customer service team for specialized assistance."

            Remember:
            - Be direct and concise
            - No email formatting or signatures
            - No meta-commentary
            - Keep responses focused
            - Always add proper spacing around bold text (** text **)

            Response:"""
        )
        
        # Greeting response prompt
        self.greeting_prompt = PromptTemplate(
            input_variables=["language"],
            template="""Generate a friendly greeting response in the appropriate language.
            
            Language: {language}
            
            Guidelines:
            - For English: Be friendly and professional
            - For Arabic: Be culturally appropriate and professional
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
        
        # Initialize state and memory
        self.conversation_memory = {}
        self.conversation_state = {}
        self.similarity_threshold = {
            'ar': 0.3,
            'en': 0.4
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
            
            # Try langdetect first
            try:
                detected = detect(text)
                if detected == 'ar':
                    return 'ar'
                elif detected in ['en', 'cy']:  # cy sometimes detected for English
                    return 'en'
            except:
                pass  # Continue to fallback if langdetect fails
            
            # Fallback: Check for Arabic characters
            if any('\u0600' <= c <= '\u06FF' for c in text):
                return 'ar'
            
            # Default to English if no Arabic detected
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
            
            # Determine intent
            intent = self.intent_chain.invoke({"query": query}).strip().upper()
            logger.info(f"Detected intent: {intent}")
            
            if intent == "GREETING":
                logger.info("Processing greeting response")
                response = self.greeting_chain.invoke({"language": language})
                return format_response(response, language)
            
            # Process as FAQ query
            logger.info("Searching for FAQ matches...")
            best_match, similarity = self.find_most_similar_faq(query, language)
            threshold = self.similarity_threshold[language]  # Use the correct threshold from instance variable
            
            # Debug logging
            logger.info(f"Query: '{query}'")
            logger.info(f"Language: {language}")
            logger.info(f"Similarity score: {similarity}")
            logger.info(f"Using threshold: {threshold} for language: {language}")
            
            if best_match and similarity > threshold:
                logger.info("Found matching FAQ - generating response")
                logger.info(f"Matched Question (EN): {best_match.question_en}")
                logger.info(f"Matched Question (AR): {best_match.question_ar}")
                logger.info(f"Answer (EN): {best_match.answer_en[:100]}...")  # Log first 100 chars of answer
                
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
                logger.info("Generated FAQ response")
            else:
                logger.info(f"No matching FAQ found (score: {similarity} < threshold: {threshold})")
                # Return fallback response
                response = get_fallback_response(language)
            
            return format_response(response, language)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return format_response(get_fallback_response(language), language) 