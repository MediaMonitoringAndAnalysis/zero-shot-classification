import nltk
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from tqdm import tqdm
from llm_multiprocessing_inference.inference import get_answers
import os
from dotenv import load_dotenv

load_dotenv()

# Download NLTK data for sentence tokenization
nltk.download('punkt')

# Prompt for classification using LLM
DETAILED_PROMPT_TEMPLATE = """
You are tasked with classifying the following text based on the provided tags. \n
The text is: "{entry}" \n
The tags to consider are: {tags} \n
Please provide a confidence score for each tag based on the relevance of the text. The score should be a 2 decimal float between 0 and 1. \n
The output should be a JSON object with the following format: \n
{{
    "tag1": score1,
    "tag2": score2,
    ...
}}
"""

class ZeroShotClassifier:
    """
    A two-stage zero-shot classifier that uses a smaller model for initial filtering
    and a larger LLM for more precise classification.
    """
    
    def __init__(
        self, 
        first_pass_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        second_pass_model: str = "gpt-4o-mini",
        first_pass_threshold: float = 0.25,
        batch_size: int = 8,
        api_key: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the zero-shot classifier.
        
        Args:
            first_pass_model (str): Model to use for the first pass classification
            second_pass_model (str): Model to use for the second pass classification
            first_pass_threshold (float): Threshold for filtering tags in the first pass
            batch_size (int): Batch size for processing sentences
            api_key (str, optional): API key for the second pass model (defaults to env variable)
            device (str, optional): Device to run models on (defaults to CUDA if available)
        """
        self.first_pass_model = first_pass_model
        self.second_pass_model = second_pass_model
        self.first_pass_threshold = first_pass_threshold
        self.batch_size = batch_size
        self.api_key = api_key or os.getenv("openai_api_key")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
    def first_pass_classification(
        self,
        entry: str,
        tags: List[str],
    ) -> Dict[str, float]:
        """
        First pass classification using a smaller multilingual model.

        Args:
            entry (str): Input text to classify
            tags (List[str]): List of possible labels/tags
            
        Returns:
            Dict[str, float]: Dictionary of tags and their confidence scores
        """
        print(f"Running first pass classification with {len(tags)} tags...")
        device = torch.device(self.device)

        classifier = pipeline("zero-shot-classification", model=self.first_pass_model, device=device)

        sentences = nltk.sent_tokenize(entry)

        # Process in batches
        scores = {}
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            results = classifier(batch, tags, multi_label=True)

            # Aggregate results
            for result in results:
                for label, score in zip(result['labels'], result['scores']):
                    if label not in scores:
                        scores[label] = []
                    scores[label].append(score)

        # Calculate the maximum score for each label
        aggregated_scores = {
            label: max(scores_list) 
            for label, scores_list in scores.items() 
            if max(scores_list) > self.first_pass_threshold
        }
        
        return aggregated_scores

    def second_pass_classification(
        self,
        entry: str,
        filtered_tags: List[str],
    ) -> Dict[str, float]:
        """
        Second pass classification using a larger LLM model with batch processing.
        
        Args:
            entry (str): Input text to classify
            filtered_tags (List[str]): List of tags from first pass
            
        Returns:
            Dict[str, float]: Dictionary of tags and their confidence scores
        """
        print(f"Running second pass classification with {len(filtered_tags)} filtered tags...")
        
        sentences = nltk.sent_tokenize(entry)
        
        # Create default_response with all tags set to 0.0
        default_response = {tag: 0.0 for tag in filtered_tags}
        default_response_str = str(default_response).replace("'", '"')
        
        # Generate prompts for each batch
        all_prompts = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            prompt_content = DETAILED_PROMPT_TEMPLATE.format(entry=" ".join(batch), tags=', '.join(filtered_tags))
            all_prompts.append([{"role": "user", "content": prompt_content}])
        
        
        # Make a single call to get_answers with all prompts
        results = get_answers(
            prompts=all_prompts,
            default_response=default_response_str,
            response_type="structured",
            api_pipeline="OpenAI",
            model=self.second_pass_model,
            api_key=self.api_key,
            show_progress_bar=True
        )

        scores = {}
        
        # Aggregate results
        for result in results:
            for label, score in result.items():
                if label not in scores:
                    scores[label] = []
                # Convert score to float if it's not already
                score_value = float(score) if not isinstance(score, float) else score
                scores[label].append(score_value)

        # Maximum score for each label
        aggregated_scores = {label: max(scores_list) for label, scores_list in scores.items() if scores_list}
        
        return aggregated_scores

    def __call__(
        self, 
        entry: str, 
        tags: List[str]
    ) -> Dict[str, float]:
        """
        Run the complete classification pipeline.
        
        Args:
            entry (str): Input text to classify
            tags (List[str]): List of possible labels/tags
            
        Returns:
            Dict[str, float]: Final classification results (from second pass)
        """
        print("===== STARTING CLASSIFICATION PIPELINE =====")
        
        # First pass
        first_pass_results = self.first_pass_classification(
            entry=entry,
            tags=tags
        )
        
        # Get filtered tags for second pass
        filtered_tags = list(first_pass_results.keys())
        
        # Second pass
        if filtered_tags:
            second_pass_results = self.second_pass_classification(
                entry=entry,
                filtered_tags=filtered_tags
            )
        else:
            print("No tags passed the first pass threshold. Returning empty results.")
            second_pass_results = {}
            
        print("===== CLASSIFICATION COMPLETE =====")
        return second_pass_results


def main():
    # Example usage with a sample text about a humanitarian situation report
    sample_entry = (
        "The humanitarian crisis in Gaza has reached unprecedented levels, with thousands of families displaced due to ongoing conflict. "
        "Access to basic necessities such as food, water, and medical supplies remains severely limited. "
        "International aid organizations are working tirelessly to provide relief, but the situation is exacerbated by restricted access and security concerns. "
        "In Ukraine, the conflict has led to a significant increase in the number of internally displaced persons, with many seeking refuge in neighboring countries. "
        "Efforts to deliver humanitarian aid are ongoing, but challenges persist due to infrastructure damage and volatile security conditions. "
        "The international community continues to call for a peaceful resolution to these conflicts, emphasizing the need for dialogue and cooperation. "
        "Meanwhile, in other parts of the world, similar humanitarian challenges are unfolding. "
        "In Yemen, years of conflict have devastated the country's infrastructure, leading to widespread famine and disease. "
        "The United Nations has described the situation as the world's worst humanitarian crisis, with millions in need of urgent assistance. "
        "In the Horn of Africa, prolonged droughts have compounded the effects of conflict, leaving millions without access to food and water. "
        "Efforts to address these crises are ongoing, but the scale of the challenges requires a coordinated international response. "
        "The role of technology in humanitarian aid is also evolving, with new tools being developed to improve the efficiency and effectiveness of relief efforts. "
        "From drones delivering medical supplies to remote areas, to blockchain technology ensuring the transparency of aid distribution, innovation is playing a crucial role in addressing these complex challenges. "
        "As the world continues to grapple with these issues, the importance of global cooperation and solidarity cannot be overstated. "
        "Only through collective action can we hope to address the root causes of these crises and build a more sustainable and equitable future for all."
    )
    sample_tags = [
        "Humanitarian Crisis",
        "Displacement",
        "Conflict",
        "Aid Organizations",
        "Cooperation",
        "Chicken",
        "Cooking",
        "Sport",
        "Iceland",
        "Palestine"
    ]
    
    classifier = ZeroShotClassifier()
    final_results = classifier(sample_entry, sample_tags)
    
    print("\n===== FINAL CLASSIFICATION RESULTS =====")
    print(final_results)

if __name__ == "__main__":
    main() 