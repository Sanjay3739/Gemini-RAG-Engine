import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import chromadb
import logging
from dotenv import load_dotenv
import os

load_dotenv()

# Setup logging for the process
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    logger.info(f"Extracting text from PDF: {pdf_path}")
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    logger.info("Text extraction complete.")
    return text

# Function to chunk text into smaller pieces for better retrieval
def chunk_text(text):
    logger.info("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  
        chunk_overlap=50,  
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"Text split into {len(chunks)} chunks.")
    return chunks    

# Set up Gemini API for embedding generation
genai.configure(api_key=os.environ["GEMINI_API_KEY"])  # Replace with your API key

# Function to generate embeddings for the text chunks
def generate_embeddings(texts):
    logger.info("Generating embeddings for text chunks...")
    embeddings = []
    for i, text in enumerate(texts):
        logger.info(f"Generating embedding for chunk {i + 1}/{len(texts)}...")
        result = genai.embed_content(
            model="models/text-embedding-004",  # You can change the model
            content=text
        )
        embeddings.append(result['embedding'])
    logger.info("Embeddings generated.")
    return embeddings

# Function to store embeddings in ChromaDB
def store_embeddings_in_chromadb(chunks, chunk_embeddings):
    logger.info("Storing embeddings in ChromaDB...")
    client = chromadb.Client()

    # Change the collection name to your preferred name
    collection = client.get_or_create_collection(name="insurance_chunks")

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "pdf"}] * len(chunks)  # Add metadata if needed
    collection.add(
        documents=chunks, 
        embeddings=chunk_embeddings, 
        metadatas=metadatas, 
        ids=ids  # Unique ID for each document
    )
    logger.info("Embeddings stored in ChromaDB.")
    return collection

# Function to rewrite a query to make it more specific and detailed
def rewrite_query(original_query):
    logger.info("Rewriting query...")
    
    query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system.
    Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.
    Original query: {original_query}
    Rewritten query:"""

    # Use Gemini to rewrite the query
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Generate and return our response
    response = model.generate_content(query_rewrite_template.format(original_query=original_query))
    logger.info("Query rewritten.")
    return response.text

# HyDERetriever class for generating hypothetical documents and performing retrieval
class HyDERetriever:
    def __init__(self, collection, chunk_size=500):
        self.collection = collection
        self.chunk_size = chunk_size

    # Function to generate a hypothetical document for the query
    def generate_hypothetical_document(self, query):
        logger.info("Generating hypothetical document...")
        hyde_prompt = """Given the question '{query}', generate a hypothetical document that directly answers this question. The document should be detailed and in-depth.
        The document size should be approximately {chunk_size} characters."""

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(hyde_prompt.format(query=query, chunk_size=self.chunk_size))
        logger.info("Hypothetical document generated.")
        return response.text

    # Function to retrieve relevant documents based on the query
    def retrieve(self, query, k=3):
        logger.info("Retrieving relevant documents using HyDE...")
        hypothetical_doc = self.generate_hypothetical_document(query)
        hypothetical_embedding = generate_embeddings([hypothetical_doc])[0]
        results = self.collection.query(
            query_embeddings=[hypothetical_embedding],
            n_results=k
        )
        similar_docs = results["documents"][0] 
        logger.info(f"Retrieved {len(similar_docs)} relevant documents.")
        return similar_docs, hypothetical_doc

# Function to generate a response based on the query and context
def generate_response(query, context):
    logger.info("Generating response...")
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    logger.info("Response generated.")
    return response.text

# Main function to orchestrate the entire process
if __name__ == "__main__":
    try:
        # Step 1: Extract and chunk text
        pdf_path = r"D:\ai_projects\rag\Insurance_Handbook_20103.pdf"  # Path to your PDF file
        logger.info(f"Starting process for PDF: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)

        # Step 2: Generate embeddings
        chunk_embeddings = generate_embeddings(chunks)

        # Step 3: Store embeddings in ChromaDB
        collection = store_embeddings_in_chromadb(chunks, chunk_embeddings)

        # Step 4: Rewrite the query
        original_query = "What is residual markets in insurance?"
        rewritten_query = rewrite_query(original_query)
        logger.info(f"Rewritten Query: {rewritten_query}")

        # Step 5: Retrieve relevant documents using HyDE
        hyde_retriever = HyDERetriever(collection)
        similar_docs, hypothetical_doc = hyde_retriever.retrieve(rewritten_query)
        logger.info(f"Hypothetical Document: {hypothetical_doc}")
        logger.info(f"Similar Documents: {similar_docs}")

        # Step 6: Generate a response
        context = " ".join(similar_docs)
        response = generate_response(original_query, context)
        logger.info(f"Generated Response: {response}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
