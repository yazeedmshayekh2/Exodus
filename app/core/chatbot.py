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
        # Set up cache file in the root directory instead of the script directory
        self.embeddings_cache_file = os.path.join(os.getcwd(), 'embeddings_cache.npz')
        
        # Initialize embedding model on CPU with the new model
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
        
        # Initialize question matcher for exact matching
        from app.core.question_matcher import QuestionMatcher
        self.question_matcher = QuestionMatcher(similarity_threshold={
            'ar': 0.20,  # Much lower threshold for Arabic to be more permissive
            'en': 0.25   # Lower threshold for English to increase match rate
        })
        
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
        
        # Initialize FAQ prompt template
        self.faq_prompt = PromptTemplate(
            input_variables=["language", "context", "query"],
            template="""You are a precise and knowledgeable assistant providing specific information based on the given context.

Language: {language}

Relevant Context:
{context}

User Query: {query}

Response Guidelines:

1. Structure:
   - If the answer involves steps, use numbered steps
   - If listing items, use bullet points
   - For complex information, break it down into clear sections

2. Content Requirements:
   - Use ONLY the information provided in the context
   - If the context doesn't fully answer the query, explicitly state what information is missing
   - Include specific details, numbers, and references from the context
   - For technical terms, provide brief explanations

3. Format Based on Query Type:
   A. For "How to" questions:
      - List prerequisites first
      - Provide numbered steps
      - Include any warnings or notes
      - End with expected outcome

   B. For informational questions:
      - Start with the most relevant information
      - Organize details in logical sections
      - Include any important qualifications or limitations

   C. For requirements or specifications:
      - List all requirements clearly
      - Specify any conditions or exceptions
      - Include deadlines or time constraints if mentioned

4. Clarity:
   - Use clear, direct language
   - Define any technical terms
   - Highlight important warnings or prerequisites
   - For Arabic, use formal Modern Standard Arabic (فصحى)
   - For English, use clear professional language

Remember: Only use information from the provided context. If information is missing or unclear, state this explicitly.

Response:"""
        )
        
        # Chitchat response prompt
        self.chitchat_prompt = PromptTemplate(
            input_variables=["language", "query"],
            template="""Generate a friendly, conversational response to the user's casual message.
            
            User message: {query}
            Language: {language}
            
            Guidelines:
            - For English: Be warm, friendly and conversational with a positive tone
            - For Arabic: Be culturally appropriate, warm and friendly
            - ALWAYS respond appropriately to questions like "how are you", "what's up", etc.
            - If asked about wellbeing ("how are you"), RESPOND with your status and add a friendly question back
            - Keep it natural and engaging (1-3 sentences)
            - Include a helpful suggestion or relevant topic when appropriate
            - Use a friendly, approachable tone that builds rapport
            - For Arabic, use formal but warm and friendly Arabic
            - Where appropriate, offer a suggestion for how you could help further
            
            Examples:
            If user asks "how are you" -> respond with "I'm doing great, thanks for asking! I hope you're having a wonderful day. Is there anything specific I can help you with?"
            If user asks "what's up" -> respond with "Just here ready to assist you! I'm always excited to help with insurance questions or provide information. What's on your mind today?"
            
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
            template="""Generate a warm, engaging greeting response in the appropriate language.
            
            Language: {language}
            
            Guidelines:
            - For English: Be friendly, welcoming and conversational with a personal touch
            - For Arabic: Be culturally appropriate and warm (use appropriate Islamic greetings if in Arabic)
            - Keep it natural and engaging (2-3 sentences)
            - Express enthusiasm about helping the user
            - Include a brief mention of what you can help with
            - Offer a suggestion for how the conversation could begin
            - For Arabic, use appropriate cultural expressions of welcome
            
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
            # Create new collection with the new model's dimension (768 for mpnet-base)
            self.qdrant.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,  # Dimension of paraphrase-multilingual-mpnet-base-v2 embeddings
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
                logger.info(f"Loading embeddings cache from: {self.embeddings_cache_file}")
                cached_points = self.load_embeddings_cache()
                if cached_points:
                    # Verify points have required fields and vector dimension matches
                    valid_points = []
                    expected_dim = 768  # dimension for paraphrase-multilingual-mpnet-base-v2
                    
                    for point in cached_points:
                        # Check required fields
                        if all(key in point.payload for key in ['question_en', 'question_ar', 'answer_en', 'answer_ar']):
                            # Check vector dimensions
                            vector = np.array(point.vector)
                            if len(vector) != expected_dim:
                                logger.error(f"Vector dimension mismatch: expected {expected_dim}, got {len(vector)}")
                                continue
                                
                            # Normalize vector
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
                        
                        logger.info(f"Successfully loaded {total_points} FAQs into Qdrant from cache")
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
            
    def find_most_similar_faq(self, query: str, language: str = 'en') -> Tuple[List[FAQEntry], List[float]]:
        """
        Find the top 5 most similar FAQs to the user's query.
        
        Args:
            query (str): User's input query
            language (str): Language of the query ('en' or 'ar')
            
        Returns:
            Tuple[List[FAQEntry], List[float]]: List of best matching FAQs and their similarity scores
        """
        try:
            if not query:
                return [], []

            # Generate and normalize query embedding
            try:
                query_embedding = self.embedding_model.encode(query)
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
            except Exception as e:
                logger.error(f"Error generating embedding for query: {e}")
                return [], []
                
            # Check dimension
            expected_dim = 768  # dimension for paraphrase-multilingual-mpnet-base-v2
            if len(query_embedding) != expected_dim:
                logger.error(f"Query embedding dimension mismatch: expected {expected_dim}, got {len(query_embedding)}")
                return [], []
            
            # Debug: Print threshold
            threshold = self.similarity_threshold[language]
            logger.info(f"Current threshold for {language}: {threshold}")
            
            # Search in Qdrant with top 5 results
            search_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=5,  # Get top 5 matches
                score_threshold=0.0  # Remove score threshold to see all results
            )
            
            if not search_results:
                logger.info("No FAQ matches found in Qdrant")
                return [], []
            
            # Debug: Print all matches
            logger.info("=== Search Results ===")
            faqs = []
            scores = []
            
            for i, match in enumerate(search_results):
                logger.info(f"Match {i+1}:")
                logger.info(f"Score: {match.score:.4f}")
                logger.info(f"Q(EN): {match.payload['question_en']}")
                logger.info(f"Q(AR): {match.payload['question_ar']}")
                logger.info("---")
                
                # Create FAQEntry from match
                faq = FAQEntry(
                    question_en=match.payload['question_en'],
                    question_ar=match.payload['question_ar'],
                    answer_en=match.payload['answer_en'],
                    answer_ar=match.payload['answer_ar']
                )
                faqs.append(faq)
                scores.append(match.score)
            
            return faqs, scores
            
        except Exception as e:
            logger.error(f"Error in find_most_similar_faq: {e}", exc_info=True)
            return [], []
    
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
            
    def is_asking_for_detail(self, query: str, language: str = 'en') -> bool:
        """
        Detect if the user is asking for more details or elaboration.
        
        Args:
            query (str): User's input query
            language (str): Language of the query
            
        Returns:
            bool: True if user is asking for details, False otherwise
        """
        # English detail request patterns
        en_detail_patterns = [
            r'\bmore detail(s|ed)?\b', 
            r'\belaborate\b', 
            r'\btell me more\b', 
            r'\bexplain (more|further)\b',
            r'\bmore information\b',
            r'\bexpand on\b',
            r'\bin depth\b',
            r'\bdetailed\b',
            r'\bcomprehensive\b',
            r'\bthorough\b'
        ]
        
        # Arabic detail request patterns
        ar_detail_patterns = [
            r'\bتفاصيل (أكثر|اكثر)\b',
            r'\bاشرح (أكثر|اكثر|بالتفصيل)\b',
            r'\bأخبرني (أكثر|اكثر)\b',
            r'\bتوضيح (أكثر|اكثر)\b',
            r'\bمعلومات (إضافية|اضافية)\b',
            r'\bتوسع\b',
            r'\bبالتفصيل\b',
            r'\bمفصل\b',
            r'\bشامل\b'
        ]
        
        # Choose the appropriate patterns based on language
        patterns = en_detail_patterns if language == 'en' else ar_detail_patterns
        
        # Check if any pattern matches the query
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
                
        return False

    def get_response(self, query: str, user_id: str = "default") -> str:
        """
        Get response for user query. The approach is:
        1. Search for exact question match in database
        2. If exact match found:
           - Return only the exact answer without using history
        3. If similar matches found but no exact match:
           - Generate answer using model knowledge base
        4. If no matches:
           - Generate new answer using model knowledge
        
        Args:
            query (str): User's input query
            user_id (str): Unique identifier for the user
            
        Returns:
            str: Response text
        """
        try:
            # Detect language
            language = self.detect_language(query)
            logger.info(f"Detected language: {language}")
            
            # Check if user is asking for more details
            wants_details = self.is_asking_for_detail(query, language)
            logger.info(f"User wants detailed response: {wants_details}")
            
            # Search for similar FAQs
            faqs, scores = self.find_most_similar_faq(query, language)
            threshold = self.similarity_threshold[language]
            
            # Debug logging
            logger.info(f"Query: '{query}'")
            logger.info(f"Language: {language}")
            logger.info(f"Found {len(faqs)} matches")
            
            # If we have matches, process them
            if faqs and scores:
                # Log all matches for debugging
                for i, (faq, score) in enumerate(zip(faqs[:5], scores[:5])):
                    logger.info(f"Match {i+1}:")
                    logger.info(f"Score: {score:.4f}")
                    logger.info(f"Q(EN): {faq.question_en}")
                    logger.info(f"Q(AR): {faq.question_ar}")
                
                # First check for exact matches using question_matcher
                for i, faq in enumerate(faqs[:5]):
                    candidate_q = faq.question_en if language == 'en' else faq.question_ar
                    if self.question_matcher.is_exact_match(query, candidate_q, language):
                        logger.info(f"Found exact match with question: {candidate_q}")
                        exact_answer = faq.answer_en if language == 'en' else faq.answer_ar
                        return format_response(exact_answer, language)
                
                # No exact match but we have a good similarity match (above threshold)
                best_match = faqs[0]
                best_score = scores[0]
                logger.info(f"Best match score: {best_score} (Threshold: {threshold})")
                
                if best_score > threshold:
                    logger.info("Found similar question - generating answer with model knowledge")
                    
                    # Get the best match question and answer for context
                    best_q = best_match.question_en if language == 'en' else best_match.question_ar
                    best_a = best_match.answer_en if language == 'en' else best_match.answer_ar
                    
                    # Create prompt that includes the similar questions but doesn't rely heavily on history
                    if language == 'ar':
                        # Adjust verbosity based on whether user wants details
                        if wants_details:
                            prompt = f"""أنت مساعد ذكي ومفيد وخبير. المستخدم سأل:
{query}

وجدت سؤالاً مشابهاً في قاعدة البيانات:
سؤال: {best_q}
إجابة: {best_a}

المستخدم يطلب تفاصيل أكثر. الرجاء تقديم إجابة شاملة ومفصلة على سؤال المستخدم باستخدام المعلومات من السؤال المشابه والمعرفة الخاصة بك.

تعليمات إضافية:
1. قدم شرحاً مفصلاً ووافياً
2. اشرح المفاهيم الرئيسية بعمق
3. قدم أمثلة وتفاصيل داعمة حيثما أمكن
4. نظم إجابتك في نقاط أو أقسام إذا كانت الإجابة طويلة
5. اختم بخلاصة توضح النقاط الرئيسية"""
                        else:
                            prompt = f"""أنت مساعد ذكي ومفيد. المستخدم سأل:
{query}

وجدت سؤالاً مشابهاً في قاعدة البيانات:
سؤال: {best_q}
إجابة: {best_a}

الرجاء الإجابة على سؤال المستخدم الأصلي باستخدام المعلومات من السؤال المشابه كمرجع. 

تعليمات إضافية:
1. كن مختصراً ومباشراً - استخدم 2-3 جمل فقط
2. تجنب الإطالة أو التكرار - أعط المعلومات الأساسية فقط
3. ركز على النقاط الرئيسية المتعلقة بالسؤال
4. استخدم لغة واضحة ومباشرة"""
                    else:
                        # Adjust verbosity based on whether user wants details
                        if wants_details:
                            prompt = f"""You are an intelligent, helpful expert assistant. The user asked:
{query}

I found a similar question in the database:
Question: {best_q}
Answer: {best_a}

The user is requesting more detailed information. Please provide a comprehensive and detailed answer to the user's original question using both the information from the similar question and your knowledge.

Additional instructions:
1. Provide an in-depth, thorough explanation
2. Explain key concepts comprehensively
3. Include supporting examples and details where possible
4. Organize your answer into sections or bullet points if lengthy
5. Conclude with a summary of the main points"""
                        else:
                            prompt = f"""You are an intelligent, helpful assistant. The user asked:
{query}

I found a similar question in the database:
Question: {best_q}
Answer: {best_a}

Please answer the user's original question using the information from the similar question as reference.

Additional instructions:
1. Be extremely concise - use only 2-3 sentences maximum
2. Avoid wordiness or repetition - provide only essential information
3. Focus solely on the main points related to the question
4. Use clear, direct language
5. Do not include pleasantries or unnecessary explanations"""
                    
                    try:
                        # Generate response using the model
                        response = self.llm.invoke(prompt)
                        return format_response(response, language)
                    except Exception as e:
                        logger.error(f"Error generating LLM response: {e}")
                        return format_response(get_fallback_response(language), language)
            
            # If no matches or no good matches found, generate response using LLM's knowledge
            logger.info("No matching FAQ found - generating new response")
            
            # Create prompt for the model without relying on conversation history
            if language == 'ar':
                # Adjust verbosity based on whether user wants details
                if wants_details:
                    prompt = f"""أنت مساعد ذكي ومفيد وخبير. المستخدم سأل:
{query}

المستخدم يطلب تفاصيل أكثر. الرجاء تقديم إجابة شاملة ومفصلة بناءً على معرفتك.

تعليمات إضافية:
1. قدم شرحاً مفصلاً ووافياً
2. اشرح المفاهيم الرئيسية بعمق
3. قدم أمثلة وتفاصيل داعمة حيثما أمكن
4. نظم إجابتك في نقاط أو أقسام إذا كانت الإجابة طويلة
5. اختم بخلاصة توضح النقاط الرئيسية"""
                else:
                    prompt = f"""أنت مساعد ذكي ومفيد. المستخدم سأل:
{query}

بما أنني لم أجد إجابة مشابهة في قاعدة البيانات، الرجاء تقديم إجابة مفيدة ودقيقة بناءً على معرفتك.

تعليمات إضافية:
1. كن مختصراً ومباشراً - استخدم 2-3 جمل فقط
2. تجنب الإطالة أو التكرار - أعط المعلومات الأساسية فقط
3. ركز على النقاط الرئيسية المتعلقة بالسؤال
4. أضف اقتراحاً مفيداً واحداً فقط إذا كان ضرورياً
5. استخدم لغة واضحة ومباشرة"""
            else:
                # Adjust verbosity based on whether user wants details
                if wants_details:
                    prompt = f"""You are an intelligent, helpful expert assistant. The user asked:
{query}

The user is requesting more detailed information. Please provide a comprehensive and detailed answer based on your knowledge.

Additional instructions:
1. Provide an in-depth, thorough explanation
2. Explain key concepts comprehensively
3. Include supporting examples and details where possible
4. Organize your answer into sections or bullet points if lengthy
5. Conclude with a summary of the main points"""
                else:
                    prompt = f"""You are an intelligent, helpful assistant. The user asked:
{query}

Since I couldn't find a similar answer in the database, please provide a direct and concise response based on your knowledge.

Additional instructions:
1. Be extremely concise - use only 2-3 sentences maximum
2. Avoid wordiness or repetition - provide only essential information
3. Focus solely on the main points related to the question
4. Add only one brief helpful suggestion if absolutely necessary
5. Use clear, direct language
6. Do not include pleasantries or unnecessary explanations"""
            
            try:
                # Generate response using the model
                response = self.llm.invoke(prompt)
                return format_response(response, language)
            except Exception as e:
                logger.error(f"Error generating LLM response: {e}")
                return format_response(get_fallback_response(language), language)
                
        except Exception as e:
            logger.error(f"Error in get_response: {e}", exc_info=True)
            return format_response(get_fallback_response(language), language) 