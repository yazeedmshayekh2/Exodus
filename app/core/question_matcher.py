"""
Question matcher utility for finding exact or similar questions in the database
"""

import logging
import re
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class QuestionMatcher:
    """
    Utility for finding exact or similar questions in the database.
    
    This class provides methods to:
    - Find exact question matches in the database
    - Compare questions semantically
    - Pre-process questions for better matching
    """
    
    def __init__(self, similarity_threshold: Dict[str, float] = None):
        """
        Initialize the question matcher.
        
        Args:
            similarity_threshold: Thresholds for similarity matching per language
        """
        self.similarity_threshold = similarity_threshold or {
            'ar': 0.20,
            'en': 0.25
        }
        
        # Common question prefixes to normalize
        self.question_prefixes = [
            'what is', 'how do i', 'how can i', 'can you', 'tell me about',
            'explain', 'how does', 'when is', 'where is', 'why is', 'who is',
            'ما هو', 'كيف يمكنني', 'هل يمكنك', 'اخبرني عن', 'اشرح', 'متى', 'اين', 'لماذا', 'من هو'
        ]
        
    def normalize_question(self, question: str) -> str:
        """
        Normalize a question by:
        - Converting to lowercase
        - Removing punctuation
        - Removing extra whitespace
        - Removing common question prefixes
        
        Args:
            question: The question to normalize
            
        Returns:
            Normalized question
        """
        if not question:
            return ""
            
        # Convert to lowercase
        normalized = question.lower()
        
        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove common question prefixes
        for prefix in self.question_prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                break
                
        return normalized
        
    def is_exact_match(self, query: str, candidate: str, language: str = 'en') -> bool:
        """
        Check if a query is an exact match for a candidate question.
        
        Args:
            query: User query
            candidate: Candidate question from database
            language: Language of the question
            
        Returns:
            True if exact match, False otherwise
        """
        # Normalize both questions
        norm_query = self.normalize_question(query)
        norm_candidate = self.normalize_question(candidate)
        
        # Check for exact match (case insensitive)
        exact_match = norm_query == norm_candidate
        
        # Check for near-exact match (small variations)
        fuzzy_match = False
        if not exact_match and len(norm_query) > 6 and len(norm_candidate) > 6:
            # If lengths are similar
            if abs(len(norm_query) - len(norm_candidate)) <= 3:
                # And Levenshtein distance is small (under 10% of length)
                edits = self.levenshtein_distance(norm_query, norm_candidate)
                max_len = max(len(norm_query), len(norm_candidate))
                fuzzy_match = edits <= max(2, int(max_len * 0.1))
        
        return exact_match or fuzzy_match
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate the Levenshtein distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance as an integer
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
    
    def score_similarity(self, query_vector: np.ndarray, candidate_vector: np.ndarray) -> float:
        """
        Calculate cosine similarity between query and candidate embeddings.
        
        Args:
            query_vector: Query embedding
            candidate_vector: Candidate embedding
            
        Returns:
            Similarity score (0-1)
        """
        # Ensure vectors are normalized
        query_norm = query_vector / np.linalg.norm(query_vector)
        candidate_norm = candidate_vector / np.linalg.norm(candidate_vector)
        
        # Calculate cosine similarity
        similarity = np.dot(query_norm, candidate_norm)
        
        return float(similarity)
    
    def is_similar_enough(self, similarity_score: float, language: str = 'en') -> bool:
        """
        Check if a similarity score exceeds the threshold for the given language.
        
        Args:
            similarity_score: Similarity score (0-1)
            language: Language code
            
        Returns:
            True if similar enough, False otherwise
        """
        threshold = self.similarity_threshold.get(language, 0.25)
        return similarity_score > threshold 