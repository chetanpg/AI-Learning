import os
from sentence_transformers import SentenceTransformer
from classes.mongodb_connector import MongoDBConnector
from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader

class SentenceTransformerVectorSearch:
    def __init__(self, mongo_uri, database_name, collection_name):
        self.connection_string = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.mongo_connector = MongoDBConnector(mongo_uri, database_name, max_pool_size=10)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def connect(self):
        self.mongo_connector.connect()
        #print("Connection successful!")
    
    def search(self, query, num_candidates=10, limit=2):
        """
        Perform vector search using either knnBeta or $vectorSearch.
        :param query: The search query.
        :param num_candidates: Number of nearest neighbors to retrieve.
        :param limit: Number of results to return.
        :param use_knn_beta: If True, use knnBeta; otherwise, use $vectorSearch.
        """
        # Get the collection
        collection = self.mongo_connector.get_collection(self.collection_name)
        if collection is None:
            raise ValueError(f"Collection '{self.collection_name}' does not exist.")

        # Generate embedding for the query
        query_embedding = self.model.encode(query).tolist()

        # Define the aggregation pipeline based on the search method
        pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index_1",  # Replace with your vector index name
                        "path": "embedding",  # Field containing the embeddings
                        "queryVector": query_embedding,  # Query vector
                        "numCandidates": num_candidates,  # Number of candidates to consider
                        "limit": limit  # Number of results to return
                    }
                },
                {
                    "$project": {
                        "_id": 0,  # Exclude the _id field from the results
                        "statement": 1,
                        "score": {"$meta": "vectorSearchScore"}  # Include the score
                    }
                }
            ]

        # Execute the aggregation pipeline and convert the cursor to a list
        cursor = collection.aggregate(pipeline)
        results = list(cursor)  # Convert CommandCursor to a list
        if not results:
            print("No results found.")
            return []
       
        # Print the results
        for result in results:
           print(result)
        return results

    def close_connection(self):
        self.mongo_connector.close_connection()

class LangChainVectorSearch:
    def __init__(self, mongo_uri, database_name, collection_name, openai_api_key):
        """
        Initialize LangChain-based vector search.
        :param connection_string: MongoDB connection string.
        :param database_name: Name of the database.
        :param collection_name: Name of the collection.
        :param openai_api_key: OpenAI API key for embeddings.
        """
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.openai_api_key = openai_api_key

        # Initialize MongoDB client and collection 
        self.mongo_connector = MongoDBConnector(mongo_uri, database_name, max_pool_size=10)
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


    def connect(self):
        """
        Connect to MongoDB.
        """
        self.mongo_connector.connect()
        print("Connected to MongoDB using LangChain!")
        # Inspect embeddings after connection
        self.inspect_embeddings()

    def inspect_embeddings(self):
        """
        Inspect stored embeddings in MongoDB.
        """
        print("inspect embedding in collection")    
        collection = self.mongo_connector.get_collection(self.collection_name)
        documents = collection.find()
        for doc in documents:
            print(doc)  # Print each document to verify embeddings

    def search(self, query, k=1):
        """
        Perform vector search using LangChain's VectorStore.
        :param query: The search query.
        :param k: Number of most similar documents to retrieve.
        :return: The most similar documents.
        """
        # Get the collection
        self.collection = self.mongo_connector.get_collection(self.collection_name)
        if self.collection is None:
            raise ValueError(f"Collection '{self.collection_name}' does not exist.")
        
        # Initialize the VectorStore
        self.vector_store = MongoDBAtlasVectorSearch(self.collection, self.embeddings)
        if self.vector_store is None: 
            raise ValueError("Failed to initialize MongoDBAtlasVectorSearch. Check your connection and collection.")
        print("VectorStore initialized successfully!")

        # Perform similarity search
        print(f"Performing similarity search for query: '{query}'")
        docs = self.vector_store.similarity_search(query, k=k)

        # Check if any documents were found
        if not docs:
          print("No results found. Debugging similarity search...")
          print(f"Query: {query}")
          print(f"Query Embedding: {self.embeddings.embed_query(query)}")
          print(f"Collection: {self.collection_name}")
          return []
        else:
          print(f"Results: {docs}")

        # Extract page content from the documents
        results = [doc.page_content for doc in docs]
        #print(f"Search results: {results}")
        return results
    
class LangChainOpenAIEmbeddingVectorSearch:
    def __init__(self, mongo_uri, database_name, collection_name, openai_api_key):
        """
        Initialize LangChain-based vector search using OpenAI embeddings.
        :param mongo_uri: MongoDB connection string.
        :param database_name: Name of the database.
        :param collection_name: Name of the collection.
        :param openai_api_key: OpenAI API key for embeddings.
        """
        self.connection_string = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.openai_api_key = openai_api_key
        self.mongo_connector = MongoDBConnector(mongo_uri, database_name, max_pool_size=10)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    def connect(self):
        """
        Connect to MongoDB.
        """
        self.mongo_connector.connect()
        self.collection = self.mongo_connector.get_collection(self.collection_name)
        if self.collection is None:
            raise ValueError(f"Collection '{self.collection_name}' does not exist.")
        print("Connected to MongoDB using LangChain with OpenAI embeddings!")

    def search(self, query, num_candidates=10, limit=2):
        """
        Perform vector search using MongoDB aggregation pipeline.
        :param query: The search query.
        :param k: Number of most similar documents to retrieve.
        :return: The most similar documents.
        """
        if self.collection is None:
            raise ValueError("MongoDB collection is not initialized. Ensure connect() was called successfully.")

        # Embed the query using OpenAI embeddings
        query_embedding = self.embeddings.embed_query(query)
        
        # Define the aggregation pipeline for vector search
        # Define the aggregation pipeline based on the search method
        pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",  # Replace with your vector index name
                        "path": "embedding",  # Field containing the embeddings
                        "queryVector": query_embedding,  # Query vector
                        "numCandidates": num_candidates,  # Number of candidates to consider
                        "limit": limit  # Number of results to return
                    }
                },
                {
                    "$project": {
                        "_id": 0,  # Exclude the _id field from the results
                        "text": 1,
                        "score": {"$meta": "vectorSearchScore"}  # Include the score
                    }
                }
            ]

        # Execute the aggregation pipeline and convert the cursor to a list
        cursor = self.collection.aggregate(pipeline)
        results = list(cursor)  # Convert CommandCursor to a list
        if not results:
            print("No results found.")
            return []
       
    
        return results
    
    def close_connection(self):
        self.mongo_connector.close_connection()
