from evaluator import RetrieverEvaluator,compare_models
import json

# Initialize evaluator
evaluator = RetrieverEvaluator()



# Evaluate a single model

with open('/Users/abdullah/Documents/Embedding-Test/RAG-Embeddings/results_json/cohere_results_comp.json', 'r') as file:
    cohere_data = json.load(file)
cohere_results = evaluator.evaluate_retrieval(cohere_data)


with open('/Users/abdullah/Documents/Embedding-Test/RAG-Embeddings/results_json/voyage_ai_results_comp.json', 'r') as file:
    voyage_data = json.load(file)
voyage_results = evaluator.evaluate_retrieval(voyage_data)

# `data_dict` is now a Python dictionary



# Compare multiple models
model_results = {
    "model1": cohere_results,
    "model2": voyage_results,
    # ... more models
}
comparison = compare_models(model_results)