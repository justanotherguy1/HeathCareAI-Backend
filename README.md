# 🏥 Healthcare Companion AI Backend

An AI-powered healthcare companion application designed to provide supportive, accurate information for breast cancer patients. Built with FastAPI and powered by AWS services.

## 🌟 Features

- **💬 Intelligent Chat**: Empathetic AI assistant specialized in breast cancer support
- **📚 Knowledge Base**: Medical information search with semantic understanding
- **🔒 Safe & Reliable**: Evidence-based responses with appropriate disclaimers
- **📱 Multi-Platform**: Supports iOS, Android, and Web clients
- **☁️ AWS-Powered**: Leverages Bedrock, OpenSearch, and S3

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                       │
│         iOS (Swift)  │  Android (Kotlin)  │  Web (React)    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Chat API   │  │ Knowledge   │  │   Health    │         │
│  │  Endpoint   │  │   Search    │  │   Checks    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   AWS Bedrock   │  │ AWS OpenSearch  │  │     AWS S3      │
│   (Claude AI)   │  │  (Vector DB)    │  │   (Documents)   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- AWS Account with access to:
  - Bedrock (Claude models)
  - OpenSearch Serverless
  - S3

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/amulyatayal/HeathCareAI-Backend.git
   cd HeathCareAI-Backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env with your AWS credentials and endpoints
   ```

5. **Run the server**
   ```bash
   python main.py
   ```

   The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📁 Project Structure

```
HeathCareAI-Backend/
├── api/                    # API routes and endpoints
│   ├── __init__.py
│   └── routes.py
├── config/                 # Configuration and AWS clients
│   ├── __init__.py
│   ├── settings.py
│   └── aws.py
├── models/                 # Pydantic schemas
│   ├── __init__.py
│   └── schemas.py
├── services/               # Business logic
│   ├── __init__.py
│   ├── ai_agent.py        # AI chat agent
│   └── knowledge_base.py  # Knowledge base operations
├── knowledge_base/         # KB management utilities
├── utils/                  # Helper functions
├── data/                   # Sample data and documents
├── logs/                   # Application logs
├── main.py                 # FastAPI application entry
├── requirements.txt        # Python dependencies
├── env.example            # Environment variables template
└── README.md
```

## 🔌 API Endpoints

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/chat/` | Send a message to the AI companion |
| DELETE | `/api/v1/chat/session/{session_id}` | Clear chat session |

### Knowledge Base

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/knowledge/search` | Search the knowledge base |
| POST | `/api/v1/knowledge/document` | Add a document |
| DELETE | `/api/v1/knowledge/document/{id}` | Delete a document |
| GET | `/api/v1/knowledge/stats` | Get KB statistics |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health/` | Full health check |
| GET | `/api/v1/health/ping` | Simple ping |

## 💬 Example Chat Request

```bash
curl -X POST "http://localhost:8000/api/v1/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are common side effects of chemotherapy?",
    "include_sources": true
  }'
```

**Response:**
```json
{
  "answer": "I understand you're asking about chemotherapy side effects...",
  "session_id": "abc123",
  "query_category": "side_effects",
  "sources": [...],
  "confidence_score": 0.85,
  "response_time_ms": 1250.5,
  "disclaimer": "This information is for educational purposes only..."
}
```

## 🏥 Query Categories

The AI agent categorizes queries to provide relevant context:

| Category | Description |
|----------|-------------|
| `symptoms` | Physical symptoms and concerns |
| `treatment` | Treatment options and procedures |
| `medication` | Medications and prescriptions |
| `side_effects` | Managing treatment side effects |
| `lifestyle` | Daily life and activities |
| `emotional_support` | Mental health and coping |
| `nutrition` | Diet and nutrition |
| `follow_up_care` | Post-treatment monitoring |
| `general` | General inquiries |

## ☁️ AWS Setup

### Bedrock

1. Enable Claude model access in AWS Bedrock console
2. Recommended models:
   - Chat: `anthropic.claude-3-haiku-20240307-v1:0`
   - Embeddings: `amazon.titan-embed-text-v2:0`

### OpenSearch Serverless

1. Create a collection for vector search
2. Configure IAM permissions
3. Create index with the provided mapping

### S3

1. Create a bucket for document storage
2. Enable versioning (recommended)
3. Configure appropriate bucket policies

## 🔐 Security Considerations

- All medical information includes appropriate disclaimers
- Rate limiting to prevent abuse
- CORS configuration for allowed origins
- No storage of personal health information (PHI) by default
- Secure API authentication (implement as needed)

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_REGION` | AWS region | `us-east-1` |
| `OPENSEARCH_ENDPOINT` | OpenSearch URL | - |
| `BEDROCK_MODEL_ID` | Chat model ID | Claude Haiku |
| `S3_BUCKET_NAME` | Document bucket | - |
| `API_PORT` | Server port | `8000` |
| `DEBUG` | Debug mode | `true` |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## ⚠️ Disclaimer

This application provides educational information only and should not replace professional medical advice. Always consult healthcare providers for medical decisions.

## 📄 License

MIT License - see LICENSE file for details

---

Built with ❤️ for breast cancer patients and their families

