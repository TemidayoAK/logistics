import pathlib
import pandas as pd
import chromadb
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry

# Configure Gemini API with direct key usage
genai.configure(api_key="AIzaSyDd2aIl3jxN_aT4OdPcbANBhliz2lkkLS0")

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, is_document=True):
        self.document_mode = is_document

    def __call__(self, input: Documents) -> Embeddings:
        embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"
        
        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}
        
        # Prepare embeddings for each input document
        embeddings = []
        for doc in input:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=doc,
                task_type=embedding_task,
                request_options=retry_policy,
            )
            embeddings.append(response["embedding"])
        
        return embeddings

# Get current directory
current_dir = pathlib.Path().resolve()
chroma_db_path = f"{current_dir}/chroma_store"

# Read the geocoded carrier data
combined_df = pd.read_csv(r"Data\pdf_folder\geocoded_data.csv", encoding='ISO-8859-1')

# Prepare documents and metadata
documents = []
metadatas = []

for index, row in combined_df.iterrows():
    # Create a combined text representation of the carrier information
    document_text = f"{row['carrier_name']} {row['state']} {row['location']} {row['address']} {row['full_address']}"
    
    # Create metadata
    metadata = {
        "carrier_name": row['carrier_name'],
        "state": row['state'],
        "location": row['location'],
        "address": row['address'],
        "full_address": row['full_address'],
        "lat": row['lat'],
        "long": row['long']
    }
    
    documents.append(document_text)
    metadatas.append(metadata)

# Initialize Gemini embedding function
embed_fn = GeminiEmbeddingFunction(is_document=True)

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Create or get the collection
DB_NAME = "carrier_embeddings"
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

# Add documents to the collection
db.add(
    documents=documents, 
    metadatas=metadatas, 
    ids=[str(i) for i in range(len(documents))]
)

print(f"Total documents embedded: {len(documents)}")
print(f"Embedding collection '{DB_NAME}' created successfully")