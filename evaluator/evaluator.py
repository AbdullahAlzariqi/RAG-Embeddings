from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import statistics

class RetrieverEvaluator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the evaluator with a sentence transformer model."""
        self.encoder = SentenceTransformer(model_name)
    
    def evaluate_retrieval(self, 
                          data: List[Dict[str, Any]],
                          k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Evaluate retrieval results for data in the format:
        [
            {
                "question": str,
                "retrieved-k": [{"text": str, "context": str}, ...]
            },
            ...
        ]
        """
        questions = [item["question"] for item in data]
        retrieved_texts = [[r["text"] for r in item["retrieved-k"]] for item in data]
        retrieved_contexts = [[r["context"] for r in item["retrieved-k"]] for item in data]
        
        results = {}
        
        # 1. Text-Context Coherence
        results["text_context_coherence"] = self._calculate_text_context_coherence(
            retrieved_texts, retrieved_contexts
        )
        
        # 2. Query-Text Relevance
        results["query_text_relevance"] = self._calculate_query_relevance(
            questions, retrieved_texts
        )
        
        # 3. Query-Context Relevance
        results["query_context_relevance"] = self._calculate_query_relevance(
            questions, retrieved_contexts
        )
        
        # 4. Retrieval Diversity
        results["text_diversity"] = self._calculate_diversity(retrieved_texts)
        results["context_diversity"] = self._calculate_diversity(retrieved_contexts)
        
        # 5. Context Window Statistics
        results["avg_context_length"] = self._calculate_avg_length(retrieved_contexts)
        results["context_length_std"] = self._calculate_length_std(retrieved_contexts)
        
        # 6. Hit Ratio at different K values
        for k in k_values:
            if k <= max(len(r) for r in retrieved_texts):
                results[f"hit_ratio_at_{k}"] = self._calculate_hit_ratio_at_k(
                    retrieved_texts, k
                )
        
        return results
    
    def _calculate_text_context_coherence(self, 
                                        texts: List[List[str]], 
                                        contexts: List[List[str]]) -> float:
        """Calculate semantic coherence between retrieved texts and their contexts."""
        coherence_scores = []
        
        for query_texts, query_contexts in zip(texts, contexts):
            if not query_texts or not query_contexts:
                continue
                
            text_embeddings = self.encoder.encode(query_texts)
            context_embeddings = self.encoder.encode(query_contexts)
            
            # Calculate similarities between corresponding text-context pairs
            similarities = [
                cosine_similarity([t], [c])[0][0] 
                for t, c in zip(text_embeddings, context_embeddings)
            ]
            coherence_scores.append(np.mean(similarities))
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_query_relevance(self, 
                                 questions: List[str], 
                                 retrieved: List[List[str]]) -> float:
        """Calculate semantic similarity between queries and retrieved content."""
        relevance_scores = []
        
        for question, items in zip(questions, retrieved):
            if not items:
                continue
                
            q_embedding = self.encoder.encode([question])[0]
            item_embeddings = self.encoder.encode(items)
            
            # Calculate similarities and take the mean
            similarities = cosine_similarity([q_embedding], item_embeddings)[0]
            relevance_scores.append(np.mean(similarities))
        
        return np.mean(relevance_scores) if relevance_scores else 0.0
    
    def _calculate_diversity(self, retrieved: List[List[str]]) -> float:
        """Calculate diversity of retrieved content using semantic similarity."""
        diversity_scores = []
        
        for items in retrieved:
            if len(items) < 2:
                continue
                
            embeddings = self.encoder.encode(items)
            sim_matrix = cosine_similarity(embeddings)
            
            # Calculate average pairwise similarity (excluding self-similarity)
            n = len(sim_matrix)
            similarities = []
            for i in range(n):
                for j in range(i + 1, n):
                    similarities.append(sim_matrix[i][j])
            
            # Convert similarity to diversity (1 - similarity)
            diversity_scores.append(1 - np.mean(similarities))
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _calculate_avg_length(self, contexts: List[List[str]]) -> float:
        """Calculate average length of context windows."""
        lengths = [len(ctx) for query_contexts in contexts for ctx in query_contexts]
        return np.mean(lengths) if lengths else 0.0
    
    def _calculate_length_std(self, contexts: List[List[str]]) -> float:
        """Calculate standard deviation of context lengths."""
        lengths = [len(ctx) for query_contexts in contexts for ctx in query_contexts]
        return np.std(lengths) if lengths else 0.0
    
    def _calculate_hit_ratio_at_k(self, retrieved: List[List[str]], k: int) -> float:
        """Calculate hit ratio at k (proportion of queries with at least k results)."""
        queries_with_k_results = sum(1 for items in retrieved if len(items) >= k)
        return queries_with_k_results / len(retrieved) if retrieved else 0.0

def compare_models(model_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    """
    Compare results across different embedding models and provide rankings.
    
    Args:
        model_results: Dictionary where:
            - keys are model names
            - values are dictionaries of metrics
            
    Returns:
        Dictionary with comparison metrics and rankings
    """
    comparison = {}
    
    # Get all metrics from the first model
    if model_results:
        metrics = set(next(iter(model_results.values())).keys())
        
        # For each metric, rank the models
        for metric in metrics:
            scores = {
                model: results[metric]
                for model, results in model_results.items()
                if metric in results and isinstance(results[metric], (int, float))
            }
            
            if scores:  # Only create rankings if we have valid scores
                # Sort models by score
                ranked_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                comparison[metric] = {
                    "best_model": ranked_models[0][0],
                    "best_score": ranked_models[0][1],
                    "rankings": ranked_models
                }
    
    return comparison

# Example usage:
"""
model_results = {
    'model1': {
        'text_context_coherence': 0.55330795,
        'query_text_relevance': 0.5745262,
        # ... other metrics
    },
    'model2': {
        'text_context_coherence': 0.5505707,
        'query_text_relevance': 0.58018255,
        # ... other metrics
    }
}

comparison = compare_models(model_results)

# Print results nicely
for metric, result in comparison.items():
    print(f"\n{metric}:")
    print(f"Best model: {result['best_model']} (score: {result['best_score']:.3f})")
    print("Rankings:")
    for model, score in result['rankings']:
        print(f"  {model}: {score:.3f}")
"""