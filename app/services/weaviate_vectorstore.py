import os

import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.auth import Auth
from weaviate.classes.config import Configure, DataType, Property


class WeaviateService:
    def __init__(self, index_name, embeddings):
        self.client = self.get_client()
        self.index_name = index_name
        self.embeddings = embeddings

    def get_client(self) -> weaviate.WeaviateClient:

        if (url := os.environ.get("WEAVIATE_URL", "")) and (
            api_key := os.environ.get("WEAVIATE_API_KEY", "")
        ):
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=Auth.api_key(api_key),
            )
        else:
            client = weaviate.connect_to_local(
                host=os.environ.get("WEAVIATE_HOST", "localhost"),
                port=int(os.environ.get("WEAVIATE_PORT", "8080")),
                grpc_port=int(os.environ.get("WEAVIATE_GRPC_PORT", "50051")),
            )

        return client

    def ensure_collection(self):
        self.client.collections.delete(name=self.index_name)

        self.client.collections.create(
            name=self.index_name,
            vector_config=Configure.Vectors.self_provided(),
            properties=[
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),
                Property(name="tags", data_type=DataType.TEXT_ARRAY),
            ],
        )

    def get_vectorstore(self) -> WeaviateVectorStore:
        return WeaviateVectorStore(
            client=self.client,
            index_name=self.index_name,
            embedding=self.embeddings,
            text_key="text",
        )
