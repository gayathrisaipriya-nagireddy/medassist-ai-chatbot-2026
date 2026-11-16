# MedAssist AI – Expanded starter package

## What's included
- `MedAssist_Expanded_Medical_Dataset.pdf`: 30 concise, paraphrased medical topic summaries.
- `data/medical_knowledge_base.json`: structured source dataset (one record per topic).
- `ingest.py`: token chunking (600 tokens, 100 overlap), embeddings and ChromaDB indexing.
- `chatbot.py`: retrieval-augmented Gemini response with source metadata and safety constraints.
- `app.py`: Streamlit chat UI.
- `requirements.txt` and `.env.example`.

## Coverage
Global information plus India-relevant topics, including TB, dengue, malaria, chikungunya, Japanese encephalitis, rabies, snakebite, maternal health, child health and immunization.

## Setup
1. Extract this ZIP.
2. Create a virtual environment (Python 3.10–3.12 recommended for smoother package compatibility).
3. Install packages: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your own Gemini API key.
5. Run `python ingest.py` to build the chunks and ChromaDB.
6. Run `streamlit run app.py`.

## Data integrity
This is a curated starter dataset of paraphrased summaries with links to original official sources, not a bulk copy of source websites and not a 500-page medical reference. Verify each page's latest content, publication date and reuse terms before expanding or redistributing. For a larger corpus, download permitted official PDFs/guidelines and add each file with its own title, publisher, date, URL and page metadata.

## Safety
For educational use only. Not a diagnostic or clinical decision-support tool. Do not use it to diagnose, prescribe or recommend individualized treatment. Urgent or severe symptoms require professional/emergency care.
