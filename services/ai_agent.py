"""
AI Agent Service for Breast Cancer Patient Queries
Uses AWS Bedrock for AI responses and OpenSearch for knowledge retrieval
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

from config import settings, bedrock
from models.schemas import (
    ChatMessage, ChatResponse, SourceCitation,
    QueryCategory, ContentType, MessageRole
)

logger = logging.getLogger(__name__)


# ================================
# System Prompt
# ================================

BREAST_CANCER_COMPANION_PROMPT = """You are a compassionate and knowledgeable healthcare companion AI assistant specializing in breast cancer support. Your role is to provide accurate, empathetic, and helpful information to breast cancer patients and their caregivers.

## Your Guidelines:

### 1. EMPATHY FIRST
- Always acknowledge the emotional aspect of the patient's journey
- Use warm, supportive language
- Recognize that every patient's experience is unique

### 2. ACCURATE INFORMATION
- Provide evidence-based information from reliable medical sources
- Cite the knowledge base sources when available
- Be clear about what is general information vs. specific medical advice

### 3. SAFETY BOUNDARIES
- NEVER provide specific treatment recommendations or medication dosages
- ALWAYS encourage consulting with healthcare providers for medical decisions
- Clearly state when a question requires professional medical consultation

### 4. RESPONSE STRUCTURE
- Start with acknowledgment of the patient's concern
- Provide clear, organized information
- End with supportive guidance and next steps

### 5. TOPICS YOU CAN HELP WITH:
- Understanding breast cancer types and stages
- Explaining common treatments (surgery, chemotherapy, radiation, hormone therapy)
- Managing side effects and symptoms
- Emotional support and coping strategies
- Nutrition and lifestyle guidance
- Questions about follow-up care
- Connecting with support resources

### 6. ALWAYS INCLUDE DISCLAIMER
End responses with a reminder that this information is educational and patients should consult their healthcare team for personalized advice.

## Knowledge Base Context:
{context}

## Conversation History:
{conversation_history}

## Current Question:
{question}

Please provide a helpful, empathetic response:"""


# ================================
# Query Classification
# ================================

QUERY_CATEGORIES = {
    "symptoms": ["symptom", "pain", "lump", "discharge", "swelling", "fatigue", "tired", "ache"],
    "treatment": ["treatment", "surgery", "mastectomy", "lumpectomy", "radiation", "chemo", "therapy"],
    "medication": ["medicine", "medication", "drug", "tamoxifen", "herceptin", "dose", "prescription"],
    "side_effects": ["side effect", "nausea", "hair loss", "fatigue", "vomiting", "pain", "reaction"],
    "lifestyle": ["exercise", "diet", "sleep", "work", "travel", "activity", "daily life"],
    "emotional_support": ["scared", "anxious", "depressed", "worried", "cope", "support", "family", "feeling"],
    "nutrition": ["food", "eat", "diet", "nutrition", "supplement", "vitamin", "weight"],
    "follow_up_care": ["follow up", "checkup", "scan", "mammogram", "monitoring", "recurrence", "survivor"]
}


def classify_query(query: str) -> QueryCategory:
    """Classify the user's query into a category"""
    query_lower = query.lower()
    
    scores = {}
    for category, keywords in QUERY_CATEGORIES.items():
        score = sum(1 for keyword in keywords if keyword in query_lower)
        scores[category] = score
    
    if max(scores.values()) > 0:
        best_category = max(scores, key=scores.get)
        return QueryCategory(best_category)
    
    return QueryCategory.GENERAL


# ================================
# Session Management
# ================================

class SessionManager:
    """Manages conversation sessions"""
    
    _sessions: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def get_or_create_session(cls, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one"""
        if session_id and session_id in cls._sessions:
            cls._sessions[session_id]["last_active"] = datetime.utcnow()
            return session_id
        
        new_id = session_id or str(uuid.uuid4())
        cls._sessions[new_id] = {
            "created_at": datetime.utcnow(),
            "last_active": datetime.utcnow(),
            "messages": [],
            "user_id": None
        }
        return new_id
    
    @classmethod
    def add_message(cls, session_id: str, role: str, content: str):
        """Add message to session history"""
        if session_id in cls._sessions:
            cls._sessions[session_id]["messages"].append({
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            })
            # Keep only last 10 messages for context
            cls._sessions[session_id]["messages"] = cls._sessions[session_id]["messages"][-10:]
    
    @classmethod
    def get_history(cls, session_id: str, max_messages: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation history"""
        if session_id not in cls._sessions:
            return []
        return cls._sessions[session_id]["messages"][-max_messages:]
    
    @classmethod
    def clear_session(cls, session_id: str):
        """Clear a session"""
        if session_id in cls._sessions:
            del cls._sessions[session_id]


# ================================
# AI Agent
# ================================

class BreastCancerCompanionAgent:
    """AI Agent for breast cancer patient support"""
    
    def __init__(self):
        self.model_id = settings.bedrock_model_id
        self.bedrock_client = None
    
    def _get_client(self):
        """Lazy load Bedrock client"""
        if self.bedrock_client is None:
            self.bedrock_client = bedrock()
        return self.bedrock_client
    
    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for prompt"""
        if not history:
            return "No previous conversation."
        
        formatted = []
        for msg in history:
            role = "Patient" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)
    
    def _format_context(self, sources: List[Dict[str, Any]]) -> str:
        """Format knowledge base sources for prompt context"""
        if not sources:
            return "No specific knowledge base sources available. Please provide general, evidence-based information."
        
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(f"""
Source {i}: {source.get('title', 'Unknown')}
Type: {source.get('content_type', 'article')}
Content: {source.get('content', '')[:500]}...
""")
        
        return "\n".join(context_parts)
    
    async def generate_response(
        self,
        question: str,
        session_id: str,
        knowledge_sources: List[Dict[str, Any]] = None,
        conversation_history: List[Dict[str, str]] = None
    ) -> Tuple[str, float]:
        """
        Generate AI response to patient question
        
        Returns:
            Tuple of (response_text, confidence_score)
        """
        start_time = time.time()
        
        # Format prompt
        context = self._format_context(knowledge_sources or [])
        history = self._format_conversation_history(conversation_history or [])
        
        prompt = BREAST_CANCER_COMPANION_PROMPT.format(
            context=context,
            conversation_history=history,
            question=question
        )
        
        try:
            client = self._get_client()
            
            # Call Bedrock
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1500,
                "temperature": 0.3,  # Lower temperature for more consistent medical info
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
            
            response = client.invoke_model(
                modelId=self.model_id,
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            answer = response_body['content'][0]['text']
            
            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence(answer, knowledge_sources)
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Generated response in {elapsed_ms:.0f}ms with confidence {confidence:.2f}")
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            raise
    
    def _calculate_confidence(
        self,
        response: str,
        sources: List[Dict[str, Any]] = None
    ) -> float:
        """Calculate confidence score for response"""
        confidence = 0.7  # Base confidence
        
        # Increase confidence if sources were used
        if sources and len(sources) > 0:
            confidence += 0.1
        
        # Increase confidence for longer, more detailed responses
        if len(response) > 500:
            confidence += 0.05
        
        # Cap at 0.95 (never claim 100% confidence for medical info)
        return min(confidence, 0.95)


# ================================
# Main Chat Function
# ================================

async def chat_with_agent(
    message: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    include_sources: bool = True
) -> ChatResponse:
    """
    Main function to chat with the breast cancer companion agent
    
    Args:
        message: User's question or message
        session_id: Optional session ID for conversation continuity
        user_id: Optional user ID for personalization
        include_sources: Whether to include source citations
    
    Returns:
        ChatResponse with answer, sources, and metadata
    """
    start_time = time.time()
    
    # Get or create session
    session_id = SessionManager.get_or_create_session(session_id)
    
    # Add user message to history
    SessionManager.add_message(session_id, "user", message)
    
    # Classify query
    query_category = classify_query(message)
    logger.info(f"Query classified as: {query_category}")
    
    # Get conversation history
    history = SessionManager.get_history(session_id)
    
    # TODO: Search knowledge base for relevant sources
    # This will be implemented in knowledge_base.py
    knowledge_sources = []
    
    # Generate AI response
    agent = BreastCancerCompanionAgent()
    answer, confidence = await agent.generate_response(
        question=message,
        session_id=session_id,
        knowledge_sources=knowledge_sources,
        conversation_history=history[:-1]  # Exclude current message
    )
    
    # Add assistant response to history
    SessionManager.add_message(session_id, "assistant", answer)
    
    # Format sources for response
    sources = []
    if include_sources and knowledge_sources:
        for source in knowledge_sources:
            sources.append(SourceCitation(
                title=source.get("title", "Unknown"),
                content_type=ContentType(source.get("content_type", "medical_article")),
                relevance_score=source.get("score", 0.0),
                source_url=source.get("url"),
                excerpt=source.get("content", "")[:200]
            ))
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    return ChatResponse(
        answer=answer,
        session_id=session_id,
        query_category=query_category,
        sources=sources,
        confidence_score=confidence,
        response_time_ms=elapsed_ms
    )

