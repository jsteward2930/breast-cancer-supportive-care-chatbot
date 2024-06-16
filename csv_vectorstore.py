import os
import openai
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Retrieve environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Debugging: Print environment variables to ensure they are loaded correctly
print(f"PINECONE_API_KEY: {PINECONE_API_KEY}")
print(f"PINECONE_ENVIRONMENT: {PINECONE_ENVIRONMENT}")
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")

# Check if any of the environment variables are None
if None in [PINECONE_API_KEY, PINECONE_ENVIRONMENT, OPENAI_API_KEY]:
    raise ValueError("One or more environment variables are not set correctly. Please check your .env file.")

INDEX_NAME = "breast-cancer-rag"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

try:
    # Check if the index exists, and create it if it doesn't
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=PINECONE_ENVIRONMENT
            )
        )
except Exception as e:
    print(f"Error during Pinecone initialization or index creation: {e}")
    exit()

# Connect to the index
index = pc.Index(INDEX_NAME)

# Initialize OpenAI client with API key
openai.api_key = OPENAI_API_KEY

# Function to generate embeddings using the new OpenAI API
def get_embeddings(text):
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return response['data'][0]['embedding']

# Function to create a valid ASCII ID
def create_ascii_id(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)[:50]

# Function to ingest embeddings into Pinecone
def ingest_embeddings():
    PATH_TO_RESOURCES = "/Users/josephsteward/Nvidia/breast-nonprofit-chatbot/breast-cancer-resources"
    embeddings = []
    for file in os.listdir(PATH_TO_RESOURCES):
        if file.endswith('.csv'):
            csv_path = os.path.join(PATH_TO_RESOURCES, file)
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                text = ' '.join(str(value) for value in row.values)
                vector_id = create_ascii_id(text)
                vector = get_embeddings(text)
                metadata = {'text': text}
                embeddings.append((vector_id, vector, metadata))

    index.upsert(vectors=embeddings)
    print("Embeddings ingested into Pinecone.")
    
if __name__ == "__main__":
    ingest_embeddings()