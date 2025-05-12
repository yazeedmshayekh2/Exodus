"""
Chat memory management for storing conversation history
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class Message:
    """
    Single message in a conversation
    
    Attributes:
        role (str): 'user' or 'assistant'
        content (str): The content of the message
        timestamp (float): Unix timestamp when the message was created
    """
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class ConversationHistory:
    """
    Conversation history for a specific user
    
    Attributes:
        user_id (str): Unique identifier for the user
        messages (List[Message]): List of messages in the conversation
        metadata (Dict[str, Any]): Additional metadata about the conversation
        max_messages (int): Maximum number of messages to keep
    """
    user_id: str
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_messages: int = field(default=20)
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history
        
        Args:
            role (str): 'user' or 'assistant'
            content (str): The content of the message
        """
        if role not in ['user', 'assistant']:
            logger.warning(f"Invalid role: {role}. Using 'user' instead.")
            role = 'user'
            
        self.messages.append(Message(role=role, content=content))
        
        # Trim history if it exceeds max_messages
        if len(self.messages) > self.max_messages:
            excess = len(self.messages) - self.max_messages
            self.messages = self.messages[excess:]
            logger.debug(f"Trimmed {excess} old messages from conversation history")
    
    def get_recent_messages(self, count: int = 5) -> List[Message]:
        """
        Get the most recent messages
        
        Args:
            count (int): Number of recent messages to retrieve
            
        Returns:
            List[Message]: List of recent messages
        """
        return self.messages[-count:] if count < len(self.messages) else self.messages
    
    def get_formatted_history(self, count: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get formatted conversation history for LLM context
        
        Args:
            count (Optional[int]): Number of recent messages to include, or None for all
            
        Returns:
            List[Dict[str, str]]: List of message dictionaries with role and content
        """
        messages = self.messages
        if count is not None:
            messages = self.messages[-count:] if count < len(self.messages) else self.messages
            
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    def clear(self) -> None:
        """Clear the conversation history"""
        self.messages = []
        logger.info(f"Cleared conversation history for user {self.user_id}")

class ConversationMemory:
    """
    Manager for all user conversation histories
    """
    def __init__(self, max_users: int = 1000, max_messages_per_user: int = 20):
        """
        Initialize conversation memory
        
        Args:
            max_users (int): Maximum number of users to keep in memory
            max_messages_per_user (int): Maximum number of messages to keep per user
        """
        self.conversations: Dict[str, ConversationHistory] = {}
        self.max_users = max_users
        self.max_messages_per_user = max_messages_per_user
        logger.info(f"Initialized conversation memory with max_users={max_users}, max_messages_per_user={max_messages_per_user}")
    
    def get_user_history(self, user_id: str) -> ConversationHistory:
        """
        Get conversation history for a user, creating it if it doesn't exist
        
        Args:
            user_id (str): Unique identifier for the user
            
        Returns:
            ConversationHistory: The user's conversation history
        """
        if user_id not in self.conversations:
            self.conversations[user_id] = ConversationHistory(
                user_id=user_id,
                max_messages=self.max_messages_per_user
            )
            logger.debug(f"Created new conversation history for user {user_id}")
            
            # If we've exceeded the maximum number of users, remove the oldest
            if len(self.conversations) > self.max_users:
                oldest_user = next(iter(self.conversations))
                del self.conversations[oldest_user]
                logger.info(f"Removed oldest conversation history for user {oldest_user}")
                
        return self.conversations[user_id]
    
    def add_message(self, user_id: str, role: str, content: str) -> None:
        """
        Add a message to a user's conversation history
        
        Args:
            user_id (str): Unique identifier for the user
            role (str): 'user' or 'assistant'
            content (str): The content of the message
        """
        history = self.get_user_history(user_id)
        history.add_message(role=role, content=content)
        logger.debug(f"Added {role} message to history for user {user_id}")
    
    def get_formatted_history(self, user_id: str, count: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get formatted conversation history for a user
        
        Args:
            user_id (str): Unique identifier for the user
            count (Optional[int]): Number of recent messages to include, or None for all
            
        Returns:
            List[Dict[str, str]]: List of message dictionaries with role and content
        """
        history = self.get_user_history(user_id)
        return history.get_formatted_history(count)
    
    def clear_user_history(self, user_id: str) -> None:
        """
        Clear conversation history for a user
        
        Args:
            user_id (str): Unique identifier for the user
        """
        if user_id in self.conversations:
            self.conversations[user_id].clear()
            logger.info(f"Cleared conversation history for user {user_id}")
    
    def clear_all(self) -> None:
        """Clear all conversation histories"""
        self.conversations = {}
        logger.info("Cleared all conversation histories") 