import os
from typing import List, Optional, Union
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from src.core.rag.hybrid_retriever import HybridRetriever, VectorRetriever, KeywordRetriever

class RAGEngine:
    """
    Advanced RAG Engine for context-augmented LLM generation.
    Supports hybrid search (Vector + Keyword), FAISS indexing, and modular prompt injection.
    """

    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.documents: List[Document] = []
        self.vector_store: Optional[FAISS] = None

    def ingest_documents(self, texts: List[str]) -> None:
        """
        Chunks and indexes raw text into the internal document store and FAISS.
        """
        new_docs = self.text_splitter.create_documents(texts)
        self.documents.extend(new_docs)
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(new_docs, self.embeddings)
        else:
            self.vector_store.add_documents(new_docs)

    def _get_hybrid_retriever(self, k: int = 5) -> HybridRetriever:
        """
        Constructs and returns a HybridRetriever instance.
        """
        if not self.documents:
            raise ValueError("No documents ingested. Ingest documents first.")
        
        vector_strategy = VectorRetriever(self.embeddings, self.documents)
        keyword_strategy = KeywordRetriever(self.documents)
        
        return HybridRetriever(vector_strategy, keyword_strategy)

    def query(self, question: str, use_hybrid: bool = True) -> str:
        """
        Executes a RAG query: Retrieval -> Augmentation -> Generation.
        """
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        if use_hybrid:
            retriever = self._get_hybrid_retriever()
            # Custom retrieval for hybrid since it doesn't follow LangChain's BaseRetriever exactly
            docs = retriever.retrieve(question)
            context = "\n\n".join([d.page_content for d in docs])
            
            chain = prompt | self.llm | StrOutputParser()
            return chain.invoke({"context": context, "question": question})
        else:
            if not self.vector_store:
                raise ValueError("Vector store not initialized.")
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            return chain.invoke(question)

    def rerank_and_query(self, question: str, top_n: int = 3) -> str:
        """
        Placeholder for advanced reranking logic (e.g., using Cohere or Cross-Encoders).
        Retrieves top_k, then narrows down to top_n using a reranker.
        """
        # Logic for reranking would be implemented here.
        # Currently defaults to standard RAG query.
        return self.query(question)

if __name__ == "__main__":
    # Example usage
    engine = RAGEngine()
    sample_texts = [
        "The GenAI Orchestrator is a robust framework for LLM management.",
        "RAG allows LLMs to access external knowledge without retraining.",
        "FAISS is an efficient library for similarity search and clustering of dense vectors."
    ]
    engine.ingest_documents(sample_texts)
    response = engine.query("What is FAISS used for?")
    print(f"RAG Response: {response}")
