import json
import re
import string
from collections import Counter
from typing import List, Dict, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer, util # New imports
from sklearn.metrics.pairwise import cosine_similarity # New imports

class RAGEvaluator:
    """
    Evaluator for RAG (Retrieval-Augmented Generation) systems
    Based on SQuAD evaluation metrics: Answer Recall, Exact Match, and F1 Score
    Enhanced with Semantic Similarity using Sentence Embeddings.
    """

    def __init__(self):
        """
        Initializes the evaluator and loads a pre-trained sentence transformer model
        for semantic similarity calculations.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2') # Load a pre-trained model
        print("SentenceTransformer model 'all-MiniLM-L6-v2' loaded for semantic similarity.")

    def normalize_answer(self, s: str) -> str:
        """
        Normalize answer text for comparison
        - Convert to lowercase
        - Remove punctuation
        - Remove articles (a, an, the)
        - Remove extra whitespace
        """
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_tokens(self, s: str) -> List[str]:
        """Convert text to list of normalized tokens"""
        if not s:
            return []
        return self.normalize_answer(s).split()

    def compute_exact_match(self, prediction: str, ground_truths: List[str]) -> float:
        """
        Compute Exact Match score
        Returns 1.0 if prediction exactly matches any ground truth, 0.0 otherwise
        """
        normalized_prediction = self.normalize_answer(prediction)
        for ground_truth in ground_truths:
            if normalized_prediction == self.normalize_answer(ground_truth):
                return 1.0
        return 0.0

    def compute_f1_score(self, prediction: str, ground_truths: List[str]) -> float:
        """
        Compute F1 score (harmonic mean of precision and recall)
        Based on token overlap between prediction and ground truths
        """
        def _compute_f1(prediction_tokens, ground_truth_tokens):
            if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
                return 1.0
            if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
                return 0.0

            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        prediction_tokens = self.get_tokens(prediction)
        f1_scores = []

        for ground_truth in ground_truths:
            ground_truth_tokens = self.get_tokens(ground_truth)
            f1_scores.append(_compute_f1(prediction_tokens, ground_truth_tokens))

        return max(f1_scores) if f1_scores else 0.0

    def compute_answer_recall(self, prediction: str, ground_truths: List[str]) -> float:
        """
        Compute Answer Recall
        Measures how many ground truth tokens are present in the prediction
        """
        prediction_tokens = set(self.get_tokens(prediction))
        recall_scores = []

        for ground_truth in ground_truths:
            ground_truth_tokens = self.get_tokens(ground_truth)
            if not ground_truth_tokens:
                recall_scores.append(1.0 if not prediction_tokens else 0.0)
                continue

            common_tokens = sum(1 for token in ground_truth_tokens if token in prediction_tokens)
            recall = common_tokens / len(ground_truth_tokens)
            recall_scores.append(recall)

        return max(recall_scores) if recall_scores else 0.0

    def compute_semantic_similarity(self, prediction: str, ground_truths: List[str]) -> float:
        """
        Compute semantic similarity using sentence embeddings and cosine similarity.
        Returns the maximum cosine similarity between the prediction and any ground truth.
        """
        if not prediction or not ground_truths:
            return 0.0

        # Encode the prediction and all ground truths
        prediction_embedding = self.model.encode(prediction, convert_to_tensor=True)
        ground_truth_embeddings = self.model.encode(ground_truths, convert_to_tensor=True)

        # Compute cosine similarity between prediction and each ground truth
        # util.cos_sim returns a tensor, so we convert to numpy and flatten
        similarities = util.cos_sim(prediction_embedding, ground_truth_embeddings).cpu().numpy().flatten()

        # Return the maximum similarity score
        return np.max(similarities)

    def evaluate_single(self, prediction: str, ground_truths: List[str]) -> Dict[str, float]:
        """
        Evaluate a single prediction against ground truth answers
        Returns dictionary with EM, F1, Answer Recall, and Semantic Similarity scores
        """
        if not isinstance(ground_truths, list):
            ground_truths = [ground_truths]

        em_score = self.compute_exact_match(prediction, ground_truths)
        f1_score = self.compute_f1_score(prediction, ground_truths)
        recall_score = self.compute_answer_recall(prediction, ground_truths)
        semantic_sim_score = self.compute_semantic_similarity(prediction, ground_truths) # New metric

        return {
            'exact_match': em_score,
            'f1': f1_score,
            'answer_recall': recall_score,
            'semantic_similarity': semantic_sim_score # Add new metric
        }

    def evaluate_dataset(self, predictions: List[str], ground_truths: List[List[str]]) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Evaluate entire dataset
        Returns average scores across all examples and detailed results
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions must match number of ground truth sets")

        total_em = 0.0
        total_f1 = 0.0
        total_recall = 0.0
        total_semantic_sim = 0.0 # New total

        detailed_results = []

        for i, (pred, gt_list) in enumerate(zip(predictions, ground_truths)):
            scores = self.evaluate_single(pred, gt_list)
            detailed_results.append({
                'index': i,
                'prediction': pred,
                'ground_truths': gt_list,
                'scores': scores
            })

            total_em += scores['exact_match']
            total_f1 += scores['f1']
            total_recall += scores['answer_recall']
            total_semantic_sim += scores['semantic_similarity'] # Accumulate new metric

        n = len(predictions)
        avg_scores = {
            'exact_match': total_em / n,
            'f1': total_f1 / n,
            'answer_recall': total_recall / n,
            'semantic_similarity': total_semantic_sim / n, # Average new metric
            'total_examples': n
        }

        return avg_scores, detailed_results

    def evaluate_from_files(self, system_output_file: str, ground_truth_file: str) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Evaluate from system output and ground truth text files.
        - system_output_file: Each 'Answer:' line is considered a prediction.
        - ground_truth_file: Each line is a ground truth for the corresponding query.
        """
        predictions = []
        with open(system_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("Answer:"):
                    answer = line.split("Answer:")[1].strip()
                    predictions.append(answer)

        ground_truths = []
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            for line in f:
                ground_truths.append([line.strip()]) # Each line is a single ground truth for now

        if len(predictions) != len(ground_truths):
            raise ValueError(f"Number of predictions ({len(predictions)}) does not match number of ground truths ({len(ground_truths)}).")

        return self.evaluate_dataset(predictions, ground_truths)

    def print_evaluation_report(self, avg_scores: Dict[str, float], detailed_results: List[Dict] = None,
                                show_details: bool = False, num_examples: int = 5, output_file: str = None):
        """
        Print formatted evaluation report and optionally save to a file.
        """
        report = "=" * 60 + "\n"
        report += "RAG SYSTEM EVALUATION REPORT\n"
        report += "=" * 60 + "\n"
        report += f"Total Examples: {avg_scores['total_examples']}\n"
        report += f"Exact Match:      {avg_scores['exact_match']:.4f} ({avg_scores['exact_match']*100:.2f}%)\n"
        report += f"F1 Score:         {avg_scores['f1']:.4f} ({avg_scores['f1']*100:.2f}%)\n"
        report += f"Answer Recall:    {avg_scores['answer_recall']:.4f} ({avg_scores['answer_recall']*100:.2f}%)\n"
        report += f"Semantic Similarity: {avg_scores['semantic_similarity']:.4f} ({avg_scores['semantic_similarity']*100:.2f}%)\n" # Display new metric
        report += "=" * 60 + "\n"

        if show_details and detailed_results:
            report += f"\nDETAILED RESULTS (First {num_examples} examples):\n"
            report += "-" * 60 + "\n"
            for i, result in enumerate(detailed_results[:num_examples]):
                report += f"Example {i+1}:\n"
                report += f"  Prediction: {result['prediction']}\n"
                report += f"  Ground Truth: {result['ground_truths']}\n"
                report += f"  EM: {result['scores']['exact_match']:.3f}, "
                report += f"  F1: {result['scores']['f1']:.3f}, "
                report += f"  Recall: {result['scores']['answer_recall']:.3f}, "
                report += f"  Semantic Sim: {result['scores']['semantic_similarity']:.3f}\n" # Display new metric
                report += "\n"

        print(report)  # Print to console

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nEvaluation report saved to: {output_file}")

def main():
    """
    Example usage of RAGEvaluator for evaluating from files and saving results.
    """
    evaluator = RAGEvaluator()
    system_output_file = "/workspaces/End-to-End-NLP-System/system_outputs/result-final.txt"
    ground_truth_file = "/workspaces/End-to-End-NLP-System/data/QnA/test/test_references.txt"
    output_file = "eval-final.txt"

    try:
        avg_scores, detailed_results = evaluator.evaluate_from_files(system_output_file, ground_truth_file)
        evaluator.print_evaluation_report(avg_scores, detailed_results, show_details=True, num_examples=50, output_file=output_file)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
