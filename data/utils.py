from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


def test_mongo_connection():
    try:
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        # Force a connection to check if the server is available
        client.admin.command('ping')
        return
    except ConnectionFailure as e:
        msg = ("MongoDB connection failed, but FiftyOne needs one. Start the service with: \n"
              "sudo systemctl start mongod")
        raise ConnectionError(msg)