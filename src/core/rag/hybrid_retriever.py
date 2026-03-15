import abc
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import faiss
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class BaseRetrieverStrategy(abc.ABC):
    """
    Abstract base class for retrieval strategies.
    Defines the contract for implementing various retrieval algorithms.
    """
    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve top_k documents for a given query.
        
        Args:
            query (str): The search query.
            top_k (int): Number of documents to retrieve.
            
        Returns:
            List[Document]: List of retrieved documents.
        """
        pass

class VectorRetriever(BaseRetrieverStrategy):
    """
    Vector-based retriever using FAISS for dense similarity search.
    """
    def __init__(self, embeddings: Embeddings, documents: List[Document]):
        """
        Initialize the VectorRetriever.
        
        Args:
            embeddings (Embeddings): The embedding model to use.
            documents (List[Document]): The documents to index.
        """
        self.embeddings = embeddings
        self.documents = documents
        self.index = self._build_index()

    def _build_index(self) -> faiss.IndexFlatL2:
        texts = [doc.page_content for doc in self.documents]
        vectors = self.embeddings.embed_documents(texts)
        dimension = len(vectors[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(vectors, dtype=np.float32))
        return index

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)
        distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.documents):
                results.append(self.documents[idx])
        return results

class KeywordRetriever(BaseRetrieverStrategy):
    """
    Keyword-based retriever using BM25 for sparse retrieval.
    """
    def __init__(self, documents: List[Document]):
        """
        Initialize the KeywordRetriever.
        
        Args:
            documents (List[Document]): The documents to index.
        """
        self.documents = documents
        corpus = [doc.page_content.split() for doc in documents]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        tokenized_query = query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0:
                results.append(self.documents[idx])
        return results

class HybridRetriever:
    """
    Hybrid retriever that combines Vector Search and Keyword Search 
    using Reciprocal Rank Fusion (RRF).
    """
    def __init__(
        self, 
        vector_retriever: BaseRetrieverStrategy, 
        keyword_retriever: BaseRetrieverStrategy,
        rrf_k: int = 60
    ):
        """
        Initialize the HybridRetriever.
        
        Args:
            vector_retriever (BaseRetrieverStrategy): The vector search instance.
            keyword_retriever (BaseRetrieverStrategy): The keyword search instance.
            rrf_k (int): The constant used in Reciprocal Rank Fusion. Defaults to 60.
        """
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve documents using both strategies and fuse the results.
        
        Args:
            query (str): The search query.
            top_k (int): Final number of documents to return.
            
        Returns:
            List[Document]: The top fused documents.
        """
        vector_results = self.vector_retriever.retrieve(query, top_k=top_k * 2)
        keyword_results = self.keyword_retriever.retrieve(query, top_k=top_k * 2)
        
        fused_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        # Process vector results
        for rank, doc in enumerate(vector_results):
            doc_id = doc.page_content  # Using content as ID for simplicity if metadata lacks ID
            doc_map[doc_id] = doc
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1 / (rank + self.rrf_k)
            
        # Process keyword results
        for rank, doc in enumerate(keyword_results):
            doc_id = doc.page_content
            doc_map[doc_id] = doc
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1 / (rank + self.rrf_k)
            
        # Sort by fused score
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top_k documents
        return [doc_map[doc_id] for doc_id, _ in sorted_docs[:top_k]]
