import os
from dotenv import load_dotenv
from classes.insert_embedding import SentenceTransformerInserter, LangChainInserter
from classes.vector_search import SentenceTransformerVectorSearch,LangChainVectorSearch,LangChainOpenAIEmbeddingVectorSearch
from classes.chatbot import Chatbot
import warnings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import MongoDBAtlasVectorSearch



# Suppress all warnings from urllib3
warnings.filterwarnings("ignore", module="urllib3")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

def insert_documents_SentenceTransformer(username,password,mongo_uri,database_name,collection_name):
    """
    Insert documents and embeddings into MongoDB using SentenceTransformer.
    """
    
    # Insert statements
    insert = SentenceTransformerInserter(mongo_uri, database_name, collection_name)
    insert.connect()

     # Insert your original document sets into MongoDB
    statements = [
            "Artificial intelligence is transforming industries.",
            "Machine learning is a subset of AI.",
            "AI applications are growing rapidly.",
            "Natural language processing is a key area of AI.",
            "Deep learning models are used for image recognition."
        ]
    
    insert.insert_statements(statements)
    insert.close_connection()

def insert_documents_langchain(username,password,mongo_uri,database_name,collection_name,openai_api_key):
    """
    Insert documents into MongoDB Atlas using LangChain and VectorStores.
    """
    # Directory containing text files to be loaded
    directory_path = "./sample_files"  # Path to the directory containing text files

   # Initialize LangChainInserter
    insert = LangChainInserter(mongo_uri, database_name, collection_name, openai_api_key, directory_path)
    insert.connect()
    
    # Insert documents into MongoDB Atlas VectorStore
    insert.insert_documents()


def llm_function(query, search_results,openai_api_key,method):
    """
    Perform LLM-based processing for the query using Retrieval-Augmented langchaination (RAG).
    :param query: The user query.
    :param vector_store: The vector store used for retrieval.
    :return: LLM response as a formatted string.
    """
    #return("enter LLM")

    # Initialize OpenAI LLM
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)

    # Format search results for LLM
    try:
        print("Search Results:", search_results)

        # Conditional logic for key access based on vector search type
        if method == "langchain":
            formatted_results = "\n".join([doc["text"] for doc in search_results])  # Access 'text' key
        elif method  == "sentence_transformer":
            formatted_results = "\n".join([doc["statement"] for doc in search_results])  # Access 'statement' key
        else:
            print(f"Error: Unsupported vector search type '{method}'.")
            return "Error processing search results."
    except KeyError:
        print("Error: 'page_content' key not found in search results.")
        return "Error processing search results."

    formatted_input = f"Query: {query}\nContext:\n{formatted_results}"
    print(f"Formatted input for LLM:\n{formatted_input}\n")

    # Generate LLM response
    try:
        response = llm.invoke(formatted_input)  # Use invoke instead of __call__
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return "Error generating LLM response."

    return response

def main():
    """
    Main function to execute chatbot with different methods.
    """
    method = input("Choose method (sentence_transformer/langchain): ").strip().lower()

    if method not in ["sentence_transformer", "langchain"]:
        print("Invalid method selected. Please choose 'sentence_transformer' or 'langchain'.")
        return

    # MongoDB connection details
    username = os.getenv("MONGO_USERNAME")
    password = os.getenv("MONGO_PASSWORD")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not username or not password:
        raise ValueError("MongoDB username and password must be set in environment variables.")
    
    mongo_uri = f"mongodb+srv://{username}:{password}@cluster0.2ndpb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    database_name = "VectorDB"
    

    # Insert documents and initialize vector store based on the selected embedding method
    if method == "sentence_transformer":
        # Initialize SentenceTransformerInserter
        collection_name = "sentence_transformer_collection"
        #insert_documents_SentenceTransformer(username,password,mongo_uri,database_name,collection_name)

        # Initialize vector store for SentenceTransformer
        vector_store = SentenceTransformerVectorSearch(mongo_uri, database_name, collection_name)
        vector_store.connect()

        # Precompute search results
        search_results_function = lambda query: vector_store.search(query)  # Precompute search results

        chatbot = Chatbot(
          vector_search_function=search_results_function,  # Pass precomputed search results function
          llm_function=lambda query: llm_function(query, search_results_function(query),openai_api_key,method)  # Pass search results to llm_function
        )
    elif method == "langchain":
        # Initialize LangChainInserter
        collection_name = "langchain_collection"
        #insert_documents_langchain(username,password,mongo_uri,database_name,collection_name,openai_api_key)
        
        # Initialize vector store for SentenceTransformer
        #vector_store = LangChainVectorSearch(mongo_uri, database_name, collection_name,openai_api_key)
        vector_store = LangChainOpenAIEmbeddingVectorSearch(mongo_uri, database_name, collection_name,openai_api_key)
        vector_store.connect()
        
        # Precompute search results
        #search_results_function = lambda query: vector_store.search(query, k=2)  # Precompute search results
        search_results_function = lambda query: vector_store.search(query) 

        chatbot = Chatbot(
          vector_search_function=search_results_function,  # Pass precomputed search results function
          llm_function=lambda query: llm_function(query, search_results_function(query),openai_api_key,method)  # Pass search results to llm_function
          #llm_function=lambda query: llm_function(query, "not running at this time",openai_api_key,method)   
        )

    # Launch chatbot
    chatbot.launch()

if __name__ == "__main__":
    main()

  