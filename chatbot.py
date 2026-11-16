import os
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai
from google.genai import errors


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found. "
        "Please check your .env file."
    )


# ==========================================
# 2. INITIALIZE GEMINI
# ==========================================

client_ai = genai.Client(api_key=api_key)


# ==========================================
# 3. INITIALIZE PROJECT DIRECTORY
# ==========================================

ROOT = Path(__file__).parent


# ==========================================
# 4. INITIALIZE CHROMADB
# ==========================================

db = chromadb.PersistentClient(
    path=str(ROOT / "vectorstore" / "chroma_db")
)

collection = db.get_or_create_collection(
    name="medassist_health_education"
)


# ==========================================
# 5. LOAD SENTENCE TRANSFORMER
# ==========================================

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)


# ==========================================
# 6. MEDICAL ASSISTANT RULES
# ==========================================

RULES = """
You are MedAssist, an educational medical information assistant.

Use only the retrieved excerpts for factual claims.

Explain information in simple and understandable language.

Cite factual sections using:
- Topic title
- Publisher
- Source URL

If retrieved information is insufficient,
clearly say so.

Do not diagnose diseases.
Do not prescribe medicines.
Do not recommend individual medication doses.
Do not provide individualized treatment plans.

For possible emergencies or severe symptoms,
advise the user to seek immediate professional
or emergency medical care.

Be respectful and non-stigmatizing.

This assistant provides general education,
not medical advice.
"""


# ==========================================
# 7. MAIN ANSWER FUNCTION
# ==========================================

def answer(question: str) -> str:

    try:

        # Generate question embedding
        qvec = embedder.encode(
            [question],
            normalize_embeddings=True
        ).tolist()[0]


        # Check knowledge base
        if collection.count() == 0:

            return (
                "The medical knowledge base is empty. "
                "Please add medical documents before "
                "asking questions."
            )


        # Retrieve relevant documents
        result = collection.query(
            query_embeddings=[qvec],
            n_results=min(5, collection.count())
        )


        docs = (result.get("documents") or [[]])[0]

        metas = (result.get("metadatas") or [[]])[0]


        # Check retrieved documents
        if not docs:

            return (
                "No relevant information was found "
                "in the medical knowledge base."
            )


        # Prepare retrieved context
        context = "\n\n".join(

            f"""
EXCERPT {i+1}

TITLE: {m.get('title')}

PUBLISHER: {m.get('publisher')}

REGION: {m.get('region_focus')}

SOURCE: {m.get('source_url')}

TEXT: {d}
"""

            for i, (d, m) in enumerate(zip(docs, metas))

        )


        # Prepare Gemini prompt
        prompt = f"""
{RULES}

RETRIEVED MATERIAL:

{context}

USER QUESTION:

{question}

Provide a clear, educational answer based only
on the retrieved material.
"""


        # ==================================
        # 8. GEMINI RESPONSE WITH RETRIES
        # ==================================

        for attempt in range(3):

            try:

                response = client_ai.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )


                if response.text:

                    return response.text


                return (
                    "No response was received. "
                    "Please try again."
                )


            except errors.ServerError as e:

                print("GEMINI SERVER ERROR:", repr(e))

                if attempt < 2:

                    wait_time = 3 * (attempt + 1)

                    print(
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                else:

                    return (
                        "Gemini servers are temporarily busy. "
                        "Please try again later."
                    )


            except errors.ClientError as e:

                print("GEMINI CLIENT ERROR:", repr(e))

                return f"Gemini API error: {e}"


            except Exception as e:

                print("GEMINI CONNECTION ERROR:", repr(e))

                return (
                    f"Gemini connection error: "
                    f"{type(e).__name__}: {e}"
                )


    except Exception as e:

        print("CHATBOT ERROR:", repr(e))

        return (
            f"Unexpected error: {type(e).__name__}: {e}"
        )


# ==========================================
# 9. RUN CHATBOT IN TERMINAL
# ==========================================

if __name__ == "__main__":

    while True:

        question = input(
            "\nAsk a general health-information question "
            "(or type exit): "
        ).strip()


        if question.lower() in {"exit", "quit"}:

            print("MedAssist stopped.")

            break


        if not question:

            continue


        print("\nMedAssist:", answer(question))