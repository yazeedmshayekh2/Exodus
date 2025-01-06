"""
Bilingual FAQ Chatbot Server

This project implements a FastAPI-based server for a bilingual FAQ chatbot that supports
English and Arabic languages. It uses semantic search with sentence transformers and
Qdrant vector database for finding relevant FAQ matches from a SQL Server database.

Key components:
- FastAPI server with CORS and security middleware.
- Sentence transformer MPNet-base-v2 for semantic embeddings.
- Local in-memory Qdrant for vector similarity search.
- Ollama Llama 3.1 8B LLM as the brain of the chatbot.
- Markdown formatting for responses.

Built by: Basel Anaya | AI Engineer
Date: 6/1/2025
"""

import logging
import pyodbc
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langdetect import detect
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import uuid
import socket
import re
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging 
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

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
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
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
        self.embeddings_cache_file = os.getenv('EMBEDDINGS_CACHE_FILE', 'embeddings_cache.npz')
        
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
            'ar': 0.5,
            'en': 0.6
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
        # Check if cache file exists and try to load it
        if os.path.exists(self.embeddings_cache_file):
            try:
                cached_points = self.load_embeddings_cache()
                if cached_points:
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=cached_points
                    )
                    logger.info(f"Successfully loaded {len(cached_points)} FAQs from cache into Qdrant")
                    return
            except Exception as e:
                logger.error(f"Error loading cached FAQs into Qdrant: {e}")
                # Only continue to database if cache loading fails
        
        # Only reach here if no cache exists or cache loading failed
        logger.info("No valid cache found. Loading FAQs from database...")
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
                        combined_embedding = (en_embedding + ar_embedding) / 2
                        
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
                        logger.info(f"Processed FAQ {i}: {row.Question_English[:50]}...")
                except Exception as e:
                    logger.error(f"Error processing FAQ {i}: {e}")
                    continue
            
            if points:
                # Save to cache before loading into Qdrant
                self.save_embeddings_cache(points)
                
                # Load into Qdrant
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Successfully loaded {len(points)} FAQs into Qdrant")
            else:
                logger.warning("No valid FAQs were loaded")
                
        except Exception as e:
            logger.error(f"Error loading FAQs: {e}")
            raise
        finally:
            connection.close()
    
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

            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Search in Qdrant
            search_result = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=3
            )
            
            if not search_result:
                logger.info("No FAQ matches found")
                return None, 0
            
            # Get the best match
            best_match = search_result[0]
            
            # Log all candidates for debugging
            for i, match in enumerate(search_result):
                logger.info(f"Match {i+1}: Score={match.score:.4f}")
                logger.info(f"Q(EN): {match.payload['question_en']}")
                logger.info(f"Q(AR): {match.payload['question_ar']}")
            
            # Create FAQEntry from best match
            faq = FAQEntry(
                question_en=best_match.payload['question_en'],
                question_ar=best_match.payload['question_ar'],
                answer_en=best_match.payload['answer_en'],
                answer_ar=best_match.payload['answer_ar']
            )
            
            return faq, best_match.score
            
        except Exception as e:
            logger.error(f"Error in find_most_similar_faq: {e}")
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
            
            # Determine intent
            intent = self.intent_chain.invoke({"query": query}).strip().upper()
            logger.info(f"Detected intent: {intent}")
            
            if intent == "GREETING":
                logger.info("Processing greeting response")
                response = self.greeting_chain.invoke({"language": language})
                return self.format_response(response, language)
            
            # Process as FAQ query
            best_match, similarity = self.find_most_similar_faq(query, language)
            threshold = self.similarity_threshold.get(language, 0.5)
            
            # Debug logging
            logger.info(f"Query: {query}")
            logger.info(f"Language: {language}")
            logger.info(f"Similarity score: {similarity}")
            logger.info(f"Threshold: {threshold}")
            
            if best_match and similarity > threshold:
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
            else:
                # Return fallback response
                response = self.get_fallback_response(language)
            
            return self.format_response(response, language)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return self.format_response(self.get_fallback_response(language), language)

    def format_response(self, response: str, language: str) -> str:
        """
        Format the response based on language.
        
        Args:
            response (str): Raw response text
            language (str): Language code ('en' or 'ar')
            
        Returns:
            str: Formatted response with proper markdown and styling
        """
        if language == 'ar':
            return self.format_arabic_response(response)
        return self.format_english_response(response)

    def format_english_response(self, response: str) -> str:
        """
        Format English response with markdown styling.
        
        Args:
            response (str): Raw English response
            
        Returns:
            str: Formatted response with proper markdown and styling
        """
        # Format phone numbers first
        response = self.format_phone_numbers_en(response)
        
        # Pre-process section titles and ensure proper spacing around bold text
        response = re.sub(r'\*\*\s*(.*?)\s*\*\*', r'**\1**', response)
        
        # Convert bullet points to markdown
        lines = response.split('\n')
        formatted_lines = []
        
        in_list = False
        list_indent = "  "  # Standard indentation for list items
        
        for line in lines:
            line = line.strip()
            
            # Skip multiple empty lines
            if not line:
                if not formatted_lines or formatted_lines[-1] != "":
                    formatted_lines.append("")
                continue
            
            # Handle section titles (enclosed in **)
            if re.match(r'^\*\*.*\*\*$', line):
                if formatted_lines and formatted_lines[-1] != "":
                    formatted_lines.append("")
                formatted_lines.append(line)
                formatted_lines.append("")
                continue
            
            # Handle numbered lists
            if re.match(r'^\d+\.', line):
                number, content = line.split('.', 1)
                formatted_lines.append(f"{number.strip()}. {content.strip()}")
                in_list = True
                continue
            
            # Handle bullet points
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                content = line.lstrip('•- *').strip()
                formatted_lines.append(f"{list_indent}* {content}")
                in_list = True
                continue
            
            # Handle continuation of list items
            if in_list and not line.startswith(('*', '-', '•', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                if line.strip():
                    formatted_lines.append(f"{list_indent}  {line}")
                else:
                    in_list = False
                continue
            
            # Regular text (not part of a list)
            in_list = False
            formatted_lines.append(line)
        
        response = '\n'.join(formatted_lines)
        
        # Clean up formatting
        response = re.sub(r'\s+:', ':', response)
        response = re.sub(r'\s+,', ',', response)
        response = re.sub(r'\s+\.', '.', response)
        response = re.sub(r'\s+\)', ')', response)
        response = re.sub(r'\(\s+', '(', response)
        
        # Add proper spacing after punctuation
        response = re.sub(r'([.,!?:])(?!\s|$)', r'\1 ', response)
        
        # Ensure proper spacing around bold text
        response = re.sub(r'(?<!\*)\*\*(?!\s)', '** ', response)  # Add space after **
        response = re.sub(r'(?<!\s)\*\*(?!\*)', ' **', response)  # Add space before **
        
        # Clean up multiple spaces and newlines
        response = re.sub(r' {2,}', ' ', response)
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        # Clean up markdown
        response = re.sub(r'\*{3,}', '**', response)  # Fix multiple asterisks
        response = re.sub(r'\*\*\s+\*\*', '', response)  # Remove empty bold tags
        
        # Remove any remaining email-like formatting
        response = re.sub(r'\n\s*Best regards,?\s*\n', '\n', response, flags=re.IGNORECASE)
        response = re.sub(r'\n\s*\[.*?\]\s*\n', '\n', response)
        response = re.sub(r'\n\s*Customer Service.*?\n', '\n', response, flags=re.IGNORECASE)
        
        return response.strip()

    def format_phone_numbers_en(self, text: str) -> str:
        """Enhanced phone number formatting"""
        import re
        
        # Format different phone number patterns
        patterns = [
            (r'\b\d{8}\b', lambda m: f"{m.group(0)[:4]} {m.group(0)[4:]}"),  # 12345678 -> 1234 5678
            (r'\b\d{4}\s?\d{4}\b', lambda m: m.group(0).replace(" ", "") [:4] + " " + m.group(0).replace(" ", "") [4:]),  # Standardize 8-digit format
            (r'\+\d{3}\s?\d{8}\b', lambda m: f"+{m.group(0).replace(' ', '')[1:4]} {m.group(0).replace(' ', '')[4:8]} {m.group(0).replace(' ', '')[8:]}"),  # International format
        ]
        
        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
        
        return result

    def format_arabic_response(self, response: str) -> str:
        """
        Format Arabic response with markdown styling.
        
        Args:
            response (str): Raw Arabic response
            
        Returns:
            str: Formatted response with proper markdown and RTL styling
        """
        # Convert Western numbers to Arabic
        western_to_arabic = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')
        response = response.translate(western_to_arabic)
        
        # Pre-process section titles
        response = re.sub(r'\*\*\s*(.*?)\*\*\s*', r'\n**\1**\n\n', response)
        
        lines = response.split('\n')
        formatted_lines = []
        
        in_list = False
        list_indent = "  "  # Standard indentation for list items
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_section:
                    formatted_lines.append("")
                continue
            
            # Handle section titles (enclosed in **)
            if re.match(r'^\*\*.*\*\*$', line):
                current_section = line
                formatted_lines.extend(["", line, ""])
                continue
            
            # Handle headings (Arabic style)
            if line.endswith(':') and len(line.split()) <= 4:
                formatted_lines.extend(["", f"**{line}**", ""])
                continue
            
            # Handle numbered lists (RTL format)
            if any(line.startswith(f'{i}.') for i in range(10)):
                number, content = line.split('.', 1)
                formatted_lines.append(f"{number.strip()}. {content.strip()}")
                in_list = True
                continue
            
            # Handle bullet points
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                content = line.lstrip('•- *').strip()
                # If content starts with bold text, handle it specially
                if content.startswith('**'):
                    content = content.replace('**', '', 2)  # Remove first pair of **
                formatted_lines.append(f"{list_indent}* {content}")
                in_list = True
                continue
            
            # Handle continuation of list items
            if in_list and not line.startswith(('*', '-', '•')):
                if line.strip():  # Only append non-empty continuation lines
                    formatted_lines.append(f"{list_indent}  {line}")
                else:
                    in_list = False  # End list on empty line
                continue
            
            # Handle bold text
            if '**' in line:
                line = re.sub(r'(?<!\*)\*\*(?!\*)', ' **', line)  # Add space before **
                line = re.sub(r'\*\*(?!\s)', '** ', line)  # Add space after **
                line = re.sub(r'\s+\*\*\s+', ' **', line)  # Clean up extra spaces
            
            # Regular text
            in_list = False
            formatted_lines.append(line)
        
        response = '\n'.join(formatted_lines)
        
        # Clean up formatting
        response = re.sub(r'\s+:', ':', response)
        response = re.sub(r'\s+،', '،', response)  # Arabic comma
        response = re.sub(r'\s+\.', '.', response)
        response = re.sub(r'\s+\)', ')', response)
        response = re.sub(r'\(\s+', '(', response)
        
        # Add proper spacing after Arabic punctuation
        response = re.sub(r'([.،؛؟!:])(?!\s|$)', r'\1 ', response)
        
        # Clean up multiple spaces and newlines
        response = re.sub(r'\s{2,}', ' ', response)
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)  # Fix multiple newlines
        
        # Clean up markdown
        response = re.sub(r'\*{4,}', '**', response)
        response = re.sub(r'\*\*\s+\*\*', '', response)
        
        # Add RTL mark and additional formatting for Arabic
        response = '\u200F' + response  # RTL mark
        
        return response.strip()

    def get_fallback_response(self, language: str) -> str:
        """Enhanced fallback response with customer service referral"""
        if language == 'ar':
            return """
                    عذراً، أقترح التواصل مع فريق خدمة العملاء للحصول على المساعدة المتخصصة.
                    """
        return """
                I apologize, I recommend contacting our customer service team for specialized assistance.
                """

def connect_to_database():
    """
    Create a connection to the SQL Server database.
    
    Returns:
        pyodbc.Connection: Database connection object
        
    Raises:
        pyodbc.Error: If connection fails
    """
    conn_str = (
        'DRIVER={ODBC Driver 18 for SQL Server};'
        'SERVER=192.168.3.120;'
        'DATABASE=agencyDB_Live;'
        'UID=sa;'
        'PWD=P@ssw0rdSQL;'
        'TrustServerCertificate=yes;'
        'Encrypt=no;'  
    )
   
    try:
        connection = pyodbc.connect(conn_str)
        logging.info("Successfully connected to the database!")
        return connection
    except pyodbc.Error as e:
        logging.error(f"Error connecting to the database: {e}")
        raise
    
# Create FastAPI app with additional configuration
app = FastAPI(
    title="FAQ Chatbot API",
    description="Production-ready FAQ Chatbot API with multilingual support",
    version="2.0.0",
    docs_url="/api/docs",  # Secure Swagger UI location
    redoc_url="/api/redoc"  # Secure ReDoc location
)

# Initialize chatbot instance
chatbot = FAQChatbot()

# Add logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming HTTP requests."""
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    return response

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Configure CORS settings
def get_allowed_origins() -> List[str]:
    """
    Get allowed origins for CORS configuration.
    
    Returns:
        List[str]: List of allowed origin URLs
    """
    origins_env = os.getenv("ALLOWED_ORIGINS")
    if origins_env:
        return [origin.strip() for origin in origins_env.split(",")]
    return [
        "http://localhost:8000",
        "http://localhost:3000",
    ]

ALLOWED_ORIGINS = get_allowed_origins()

ALLOWED_METHODS = ["GET", "POST", "OPTIONS"]

ALLOWED_HEADERS = [
    "Content-Type",
    "Authorization",
    "Accept",
    "Origin",
    "X-Requested-With",
]

# Add CORS middleware with secure configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS,
    max_age=3600,  # Cache preflight requests for 1 hour
    expose_headers=["Content-Length"],  # Headers that can be exposed to the browser
)

# Add a custom exception handler for 405 Method Not Allowed
@app.exception_handler(405)
async def method_not_allowed_handler(request: Request, exc: HTTPException):
    if request.url.path == "/chat/":
        return JSONResponse(
            status_code=405,
            content={
                "detail": "This endpoint only accepts POST requests. Please use the chat interface or send a POST request with JSON data."
            }
        )
    return JSONResponse(
        status_code=405,
        content={"detail": "Method not allowed"}
    )

# Mount templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the chat interface."""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat/", response_model=ChatResponse)
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

@app.get("/debug/faqs")
async def get_faq_count():
    """Debug endpoint to check loaded FAQs"""
    try:
        count = len(chatbot.qdrant.scroll(
            collection_name=chatbot.collection_name,
            limit=1
        )[0])
        return {"faq_count": count}
    except Exception as e:
        logger.error(f"Error checking FAQ count: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        port = int(os.getenv('PORT', 8000))
        host = os.getenv('HOST', '0.0.0.0')
        workers = int(os.getenv('WORKERS', 4))
        
        if os.getenv('ENVIRONMENT') == 'production':
            # Production server setup
            import gunicorn.app.base

            class StandaloneApplication(gunicorn.app.base.BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()

                def load_config(self):
                    for key, value in self.options.items():
                        self.cfg.set(key.lower(), value)

                def load(self):
                    return self.application

            options = {
                'bind': f'{host}:{port}',
                'workers': workers,
                'worker_class': 'uvicorn.workers.UvicornWorker',
                'timeout': 120,
                'keepalive': 5,
                'errorlog': 'error.log',
                'accesslog': 'access.log',
                'capture_output': True,
                'loglevel': os.getenv('LOG_LEVEL', 'info'),
            }

            StandaloneApplication(app, options).run()
        else:
            # Development server setup
            uvicorn.run(
                "main:app",
                host=host,
                port=port,
                reload=True,
                log_level=os.getenv('LOG_LEVEL', 'info').lower()
            )
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
    