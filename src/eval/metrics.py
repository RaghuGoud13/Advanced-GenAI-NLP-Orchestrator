from typing import List, Dict, Any
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

class RagasEvaluator:
    """
    Automated evaluation metrics using Ragas-inspired logic.
    Provides metrics for Faithfulness, Answer Relevance, Context Precision, and Context Recall.
    """
    def __init__(self, llm: BaseChatModel, embeddings: Embeddings):
        """
        Initialize the RagasEvaluator.
        
        Args:
            llm (BaseChatModel): The LLM to use for evaluation.
            embeddings (Embeddings): The embedding model for vector-based metrics.
        """
        self.llm = llm
        self.embeddings = embeddings
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]

    def evaluate_results(
        self, 
        questions: List[str], 
        answers: List[str], 
        contexts: List[List[str]], 
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate a set of RAG results.
        
        Args:
            questions (List[str]): List of user questions.
            answers (List[str]): List of generated answers.
            contexts (List[List[str]]): List of lists of retrieved contexts.
            ground_truths (List[str]): List of ground truth answers.
            
        Returns:
            Dict[str, float]: Calculated metrics scores.
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)
        
        result = evaluate(
            dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )
        
        return result
