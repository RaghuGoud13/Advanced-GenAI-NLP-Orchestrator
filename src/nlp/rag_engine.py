import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class RAGEngine:
    """
    Advanced RAG Engine for context-augmented LLM generation.
    Supports semantic search, FAISS indexing, and modular prompt injection.
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
        self.vector_store: Optional[FAISS] = None

    def ingest_documents(self, texts: List[str]) -> None:
        """
        Chunks and indexes raw text into the FAISS vector store.
        """
        docs = self.text_splitter.create_documents(texts)
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vector_store.add_documents(docs)

    def _get_retriever(self, k: int = 5):
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Ingest documents first.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def query(self, question: str) -> str:
        """
        Executes a RAG query: Retrieval -> Augmentation -> Generation.
        """
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        retriever = self._get_retriever()

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
