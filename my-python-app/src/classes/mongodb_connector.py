from pymongo import MongoClient

class MongoDBConnector:
    def __init__(self, connection_string, database_name, max_pool_size=10):
        """
        Initialize the MongoDBConnector with connection pooling.
        :param connection_string: MongoDB Atlas connection string.
        :param database_name: Name of the database to connect to.
        :param max_pool_size: Maximum number of connections in the pool.
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.max_pool_size = max_pool_size
        self.client = None
        self.db = None

    def connect(self):
        """
        Connect to MongoDB Atlas using connection pooling.
        """
        try:
            self.client = MongoClient(self.connection_string, maxPoolSize=self.max_pool_size)
            self.db = self.client[self.database_name]
            #print(f"Connected to database: {self.database_name} with connection pooling (maxPoolSize={self.max_pool_size})")
        except Exception as e:
            print(f"Failed to connect to MongoDB Atlas: {e}")

    def get_collection(self, collection_name):
       """
       Get a collection from the connected database.
       """
       try:
          return self.db[collection_name]
       except AttributeError:
          print("Database connection is not established.")
          return None

    def close_connection(self):
        """
        Close the connection to MongoDB Atlas.
        """
        if self.client:
            self.client.close()
            print("Connection to MongoDB Atlas closed.")