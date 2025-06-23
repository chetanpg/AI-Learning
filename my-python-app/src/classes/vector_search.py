import os
from sentence_transformers import SentenceTransformer
from classes.mongodb_connector import MongoDBConnector
from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

#Embedding used is SentenceTransformer('all-MiniLM-L6-v2') for SentenceTransformerVectorSearch
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
    
#Vector search is using raw MongoDB $vectorsearch pipeline
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

#Using langchain vector search methos instead of raw $vectorsearch in the mongodb pipeline 
class LangChainOpenAIEmbeddingVectorSearchnNew:
    def __init__(self, mongo_uri, database_name, collection_name, openai_api_key, vector_index_name):
        """
        Initialize LangChain-based vector search with RAG pipeline.
        :param mongo_uri: MongoDB connection string.
        :param database_name: Name of the database.
        :param collection_name: Name of the collection.
        :param openai_api_key: OpenAI API key for embeddings and LLM.
        :param vector_index_name: Name of the vector index in MongoDB.
        """
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.openai_api_key = openai_api_key
        self.vector_index_name = vector_index_name
        self.vector_store = None
        self.retriever = None
        self.rag_chain = None

    def connect(self):
        """
        Connect to MongoDB Atlas and initialize the vector store, retriever, and RAG chain.
        """
        # Step 1: Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

        # Step 2: Create a vector store in MongoDB Atlas
        self.vector_store = MongoDBAtlasVectorSearch.from_connection_string(
            self.mongo_uri,
            f"{self.database_name}.{self.collection_name}",
            embeddings,
            index_name=self.vector_index_name
        )
        if self.vector_store is None:
            raise ValueError("Failed to initialize MongoDBAtlasVectorSearch. Check your connection and collection.")
        else:
            print("Vector store initialized successfully!")
        # Ensure the vector store is connected

        # Step 3: Define a retriever for searching relevant data in the vector store
        self.retriever = self.vector_store.as_retriever(search_type="similarity", k=2)
        print("Retriever initialized successfully.")

        # Step 4: Build a Retrieval-Augmented Generation (RAG) pipeline
        llm = OpenAI(openai_api_key=self.openai_api_key, temperature=0)
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.retriever,
            return_source_documents=True
        )
        print("RAG pipeline initialized successfully.")

    def search(self, query):
        """
        Perform a search using the RAG pipeline.
        :param query: The user query.
        :return: The response generated by the RAG pipeline.
        """
        if self.rag_chain is None:
            raise ValueError("RAG pipeline is not initialized. Ensure connect() was called successfully.")

        # Step 5: Execute the RAG pipeline
        print(f"Executing RAG pipeline for query: '{query}'")
        result = self.rag_chain.invoke(query)

        # Step 6: Return the response
        print(f"RAG pipeline response: {result['result']}")
        return result