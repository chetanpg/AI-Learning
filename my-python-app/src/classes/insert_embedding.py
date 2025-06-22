import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from classes.mongodb_connector import MongoDBConnector
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader


class SentenceTransformerInserter:
    def __init__(self, mongo_uri, database_name, collection_name):
        self.connection_string = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.mongo_connector = MongoDBConnector(mongo_uri, database_name, max_pool_size=10)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def connect(self):
        self.mongo_connector.connect()
        print("Connection successful!")

    def insert_statements(self, statements):
        # Create or get the collection
        collection = self.mongo_connector.get_collection(self.collection_name)
        if collection is None:
            print(f"Creating new collection: {self.collection_name}")
            collection = self.mongo_connector.db.create_collection(self.collection_name)

        # Generate embeddings and insert documents
        documents = []
        for statement in statements:
            embedding = self.model.encode(statement).tolist()  # Convert embedding to list for MongoDB compatibility
            documents.append({"statement": statement, "embedding": embedding})

        # Insert documents into the collection
        result = collection.insert_many(documents)
        print(f"Inserted {len(result.inserted_ids)} documents into the collection '{self.collection_name}'.")

    def close_connection(self):
        self.mongo_connector.close_connection()

class LangChainInserter:
    def __init__(self, mongo_uri, database_name, collection_name, openai_api_key, directory_path):
        """
        Initialize the LangChainInserter class.
        :param mongo_uri: MongoDB connection string.
        :param database_name: Name of the database.
        :param collection_name: Name of the collection.
        :param openai_api_key: OpenAI API key for embeddings.
        :param directory_path: Path to the directory containing documents.
        """
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.openai_api_key = openai_api_key
        self.directory_path = directory_path
        self.mongo_connector = MongoDBConnector(mongo_uri, database_name, max_pool_size=10)
        
    def connect(self):
        self.mongo_connector.connect()
        print("Connection successful!")

    def insert_documents(self):
        """
        Load documents, generate embeddings, and insert them into MongoDB Atlas using LangChain.
        """
        # Load documents from the specified directory
        collection = self.mongo_connector.get_collection(self.collection_name)
        if collection is None:
            print(f"Creating new collection: {self.collection_name}")
            collection = self.mongo_connector.db.create_collection(self.collection_name)

        loader = DirectoryLoader(self.directory_path, glob="./*.txt", show_progress=True)
        try:
          data = loader.load()
          print(f"Loaded {len(data)} documents.")
        except Exception as e:
          print(f"Error loading documents: {e}")
          return
        
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key,
                                       model="text-embedding-3-small",
                                       max_retries=3,
                                       request_timeout=60)
        if not embeddings:
            raise ValueError("OpenAI embeddings could not be initialized. Check your API key.")

        # Initialize the VectorStore and insert documents
        vector_store = MongoDBAtlasVectorSearch.from_documents(data, embeddings, collection=collection)
        if not vector_store:
            raise ValueError("VectorStore could not be created. Check your MongoDB connection and collection.")