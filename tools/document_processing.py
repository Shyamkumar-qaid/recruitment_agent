from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# import pinecone # Remove this line
from pinecone import Pinecone, ServerlessSpec # Add this line
import os
from dotenv import load_dotenv

load_dotenv()

class DocumentProcessor:
    def __init__(self):
        # Replace the old initialization block with the new one
        # pinecone.init(
        #     # api_key=os.getenv('PINECONE_API_KEY'),
        #     api_key="",
        #     environment=os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')
        # )
        # self.index_name = os.getenv('PINECONE_INDEX_NAME', 'candidate-embeddings')
        # self.vector_db = pinecone.Index(self.index_name)
        
        # New initialization
        api_key = os.getenv('PINECONE_API_KEY')
        environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east1-gcp')
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'candidate-embeddings')

        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set.")
        if not environment:
             raise ValueError("PINECONE_ENVIRONMENT environment variable not set.")

        pc = Pinecone(api_key=api_key, environment=environment)
        index_exists = self.index_name in [idx.name for idx in pc.list_indexes()]
        correct_dimension = 384

        if index_exists:
            index_description = pc.describe_index(self.index_name)
            if index_description.dimension != correct_dimension:
                print(f"Index '{self.index_name}' found with incorrect dimension {index_description.dimension}. Deleting and recreating with dimension {correct_dimension}.")
                pc.delete_index(self.index_name)
                index_exists = False # Mark as non-existent to trigger creation below
            else:
                print(f"Index '{self.index_name}' found with correct dimension {correct_dimension}.")

        if not index_exists:
            print(f"Creating index '{self.index_name}' with dimension {correct_dimension}...")
            pc.create_index(
                name=self.index_name,
                dimension=correct_dimension,  # Use the variable
                metric="cosine",
                spec=ServerlessSpec(cloud="gcp", region="us-east-1")  # adjust to your need
            )
            print(f"Index '{self.index_name}' created successfully.")

        self.vector_db = pc.Index(self.index_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"  # or your LLaMA-compatible model
        )

    
    def process_resume(self, file_path):
        """Process resume file and create embeddings"""
        try:
            # Load document based on file type
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(('.doc', '.docx')):
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Load and split the document
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", ",", " ", ""]
            )
            texts = text_splitter.split_documents(pages)
            
            # Create embeddings and store in Pinecone
            embeddings = [self.embeddings.embed_query(text.page_content) for text in texts]
            metadata = [{
                'text': text.page_content,
                'source': file_path,
                'page': text.metadata.get('page', 0)
            } for text in texts]
            
            # Generate unique vector IDs
            vector_ids = [f"doc_{i}_{os.path.basename(file_path)}" for i in range(len(texts))]
            
            # Upsert to Pinecone
            self.vector_db.upsert(vectors=zip(vector_ids, embeddings, metadata))
            
            return {
                'status': 'success',
                'num_chunks': len(texts),
                'vector_ids': vector_ids
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def search_similar_documents(self, query, top_k=5):
        """Search for similar documents using the query"""
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.vector_db.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            return results
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }