"""Services module"""
from .ai_agent import chat_with_agent, BreastCancerCompanionAgent, SessionManager
from .knowledge_base import KnowledgeBaseService, get_knowledge_base, EmbeddingService

__all__ = [
    'chat_with_agent',
    'BreastCancerCompanionAgent',
    'SessionManager',
    'KnowledgeBaseService',
    'get_knowledge_base',
    'EmbeddingService'
]

