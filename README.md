# DocuRAG - Intelligent Document Analysis System

![image](https://github.com/user-attachments/assets/318f28ec-1cdd-40d4-9c85-9e19a0c646ed)

![image](https://github.com/user-attachments/assets/704109b3-707b-4191-bc60-5d5cc19163b6)



A powerful document analysis tool combining RAG (Retrieval-Augmented Generation) with multi-modal processing for PDF/text analysis.

## Features

- **Document Processing**
  - PDF/TXT file ingestion
  - Language detection
  - Domain classification (Education, Health, Technology, etc.)
  - Key term extraction & visualization
  
- **AI Analysis**
  - Context-aware summarization
  - Interactive Q&A interface
  - Semantic search capabilities
  - Notes preservation system

- **Technical Highlights**
  - Hybrid retrieval (TF-IDF + Sentence Transformers)
  - Quantized LLMs for efficient inference
  - GPU/CPU compatible architecture
  - Secure API key management

## Architercture 
![image](https://github.com/user-attachments/assets/e4f82f6d-a510-442d-8dc2-c437c7620ffc)

![image](https://github.com/user-attachments/assets/5e8104b8-9c7e-4f98-afae-438a7b1af1c2)


## Demo




## Installation

1. **Clone Repository**
```bash
git clone https://github.com/salam-alkaissi/ChatBot_V3.git
cd docurag


python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows


pip install -r requirements.txt
python -m nltk.downloader punkt


cp config/api_keys.env.example config/api_keys.env
# Add  API keys to config/api_keys.env

# Start the application
python app.py



docurag/
├── src/
│   ├── document_processing.py
│   ├── generation.py
│   ├── graph_summary.py
│   ├── hybrid_retrieval.py
│   ├── topic_visualization.py
│   ├── wiki_utils.py
│   └── rag_pipeline.py
├── config/
│   └── api_keys.env
├── outputs/
├── app.py
└── requirements.txt
