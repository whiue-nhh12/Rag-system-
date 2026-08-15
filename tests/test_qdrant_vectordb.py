import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag_models.vectordb.qdrantvectordb import QdrantVectorDB


class QdrantVectorDBTests(unittest.TestCase):
    def test_add_documents_uses_provided_embeddings(self):
        client = MagicMock()
        client.collection_exists.return_value = False
        client.create_collection.return_value = None
        client.upsert.return_value = None

        with patch("rag_models.vectordb.qdrantvectordb.QdrantClient", return_value=client):
            db = QdrantVectorDB(collection_name="demo", url="http://localhost:6333")
            documents = [Document(page_content="hello world", metadata={"source": "doc-1"})]

            count = db.add_documents(documents, embeddings=[[0.1, 0.2]], metadata=[{"source": "doc-1"}])

            self.assertEqual(count, 1)
            client.upsert.assert_called_once()


if __name__ == "__main__":
    unittest.main()
