import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import pandas
from dotenv import load_dotenv
from pathlib import Path

# from plotly.graph_objs.icicle import Pathbar
# from app.main import faqs_path

load_dotenv()

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

chroma_client = chromadb.Client()
groq_client = Groq()
collection_name_faq = 'faqs'
# faqs_path = Path(__file__).parent / "resources/final.csv"

def ingest_faq_data(path):
    if collection_name_faq not in [c.name for c in chroma_client.list_collections()]:
        print("Ingesting FAQ data into Chromadb...")
        collection = chroma_client.create_collection(
            name=collection_name_faq,
            embedding_function=ef
        )
        df = pandas.read_csv(path)
        docs = df['question'].to_list()
        metadata = [{'answer': ans} for ans in df['answer'].to_list()]
        ids = [f"id_{i}" for i in range(len(docs))]
        collection.add(
            documents=docs,
            metadatas=metadata,
            ids=ids
        )
        print(f"FAQ Data successfully ingested into Chroma collection: {collection_name_faq}")
    else:
        print(f"Collection: {collection_name_faq} already exist")


def get_relevant_qa(query):
    collection = chroma_client.get_collection(
        name=collection_name_faq,
        embedding_function=ef
    )
    result = collection.query(
        query_texts=[query],
        n_results=1
    )
    return result


def generate_answer(query, reference):
    prompt = f'''
You are a knowledgeable health assistant.

Answer the QUESTION using:

- Your general knowledge as the main source
- The REFERENCE only if it directly matches diseases, symptoms, treatments, or health tips
- No mention of the REFERENCE’s amount or quality of info
- Logic and understanding as the basis, referencing REFERENCE only when relevant
- Prefer short, clear answers; use bullet points if helpful to engage the user

REFERENCE:
{reference}

QUESTION:
{query}

Answer concisely:
'''

    completion = groq_client.chat.completions.create(
        model=os.environ['GROQ_MODEL'],
        messages=[
            {
                'role': 'user',
                'content': prompt.strip()
            }
        ],
        max_completion_tokens=1024
    )
    return completion.choices[0].message.content


def faq_chain(query):
    result = get_relevant_qa(query)
    refference = "".join([r.get('answer') for r in result['metadatas'][0]])
    # print("refference:", refference)
    answer = generate_answer(query, refference)
    return answer


if __name__ == '__main__':
    ingest_faq_data(faqs_path)
    query = "Suggest some effective home remedies for managing mild headaches."
    # query = "How often should I exercise?"
    result = get_relevant_qa(query)
    answer = faq_chain(query)
    print("Answer:", answer)
