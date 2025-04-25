from crewai import Agent
from tools.document_processing import DocumentProcessor
from tools.llm_tools import LLMTools
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

load_dotenv()

class ResumeAgent:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.llm_tools = LLMTools()
        self.agent = Agent(
            role='Resume Processing Expert',
            goal='Extract and analyze resume information',
            backstory='Specialized in parsing and understanding resumes',
            verbose=True,
            allow_delegation=False,
            # Ensure Ollama server is running and the model is pulled (e.g. ollama run phi3)
            llm=Ollama(model="phi3") # Replace with your desired Ollama model
        )
    
    def process(self, file_path, job_context):
        try:
            # Process document and create embeddings
            doc_result = self.doc_processor.process_resume(file_path)
            if doc_result['status'] == 'error':
                return doc_result
            
            # Extract text from the processed document
            similar_docs = self.doc_processor.search_similar_documents(
                query="Extract all relevant information from this resume",
                top_k=doc_result['num_chunks']
            )
            
            # Combine all text chunks
            full_text = ' '.join([doc['metadata']['text'] for doc in similar_docs['matches']])
            
            # Extract structured information
            entities_result = self.llm_tools.extract_entities(full_text)
            if entities_result.get('status') != 'success':
                # Propagate the error status and details
                return entities_result
            
            # Extract data from the 'data' key upon success
            entities_data = entities_result.get('data', {})
            
            return {
                'status': 'success',
                'skills': entities_data.get('skills', []),
                'experience': entities_data.get('experience', []),
                'education': entities_data.get('education', []),
                'contact_info': entities_data.get('contact_info', {}),
                'embeddings_ref': ','.join(doc_result['vector_ids'])
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

