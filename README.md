# Report Analyst

A modern document analysis tool built with LangChain for analyzing corporate reports and documents.

## Features

- 📄 PDF and document processing
- 🤖 Advanced document analysis using LLMs
- 🔍 Customizable question & answer system
- 📊 Structured report generation
- 🎯 Modular prompt system
- 🚀 FastAPI backend
- ⚡ High performance document processing

## Project Structure

```
report-analyst/
├── app/                    # Main application code
│   ├── api/               # FastAPI routes and endpoints
│   ├── core/              # Core business logic
│   ├── models/            # Pydantic models
│   └── services/          # Service layer
├── prompts/               # Modular prompt templates
│   ├── analysis/         # Document analysis prompts
│   └── qa/               # Q&A prompts
├── config/                # Configuration files
├── data/                  # Data directory
│   ├── input/            # Input documents
│   └── output/           # Generated outputs
└── tests/                # Test suite
```

## Question Set Naming Convention

This project uses a **mountain peak naming convention** for question sets to enable scalable versioning:

### Current Question Sets:
- **`everest_questions.yaml`** - Comprehensive sustainability labeling framework (35 questions, prefix: `ev_`)
- **`tcfd_questions.yaml`** - TCFD climate disclosure questions (prefix: `tcfd_`)
- **`s4m_questions.yaml`** - Score4More sustainability questions (prefix: `s4m_`)
- **`lucia_questions.yaml`** - Lucia sustainability analysis questions (prefix: `lucia_`)

### Future Naming Pattern:
| Version | Mountain | File | Prefix |
|---------|----------|------|--------|
| 1 | **Everest** | `everest_questions.yaml` | `ev_` |
| 2 | **Kilimanjaro** | `kilimanjaro_questions.yaml` | `ki_` |
| 3 | **Denali** | `denali_questions.yaml` | `de_` |
| 4 | **Matterhorn** | `matterhorn_questions.yaml` | `ma_` |
| 5 | **Fuji** | `fuji_questions.yaml` | `fu_` |

This system provides hundreds of unique names for future question set iterations while maintaining memorable, professional naming.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_ORGANIZATION=your_organization_id  # Optional
OPENAI_API_MODEL=gpt-4o-mini             # Default model

# For Gemini models
GOOGLE_API_KEY=your_google_api_key_here  # Required for Gemini models
```

4. Run the application:
```bash
uvicorn app.main:app --reload
# Or use the Streamlit interface
streamlit run app/streamlit_app.py
```

## Available LLM Models

The application supports multiple LLM providers:

### OpenAI Models (requires OPENAI_API_KEY)
- gpt-4o-mini
- gpt-4o
- gpt-3.5-turbo

### Google Gemini Models (requires GOOGLE_API_KEY)
- gemini-flash-2.0
- gemini-pro

Models will only be available in the UI if you have the corresponding API key configured.

## Usage

1. Place your documents in the `data/input` directory
2. Use the API endpoints to:
   - Analyze documents
   - Ask questions about documents
   - Generate structured reports

## Customizing Prompts

The `prompts` directory contains modular prompt templates that can be customized for different use cases. Each prompt is a separate file that can be modified without affecting the core functionality.

## License

MIT License 