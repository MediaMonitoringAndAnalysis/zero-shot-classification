import json
import nltk
from typing import List, Dict, Tuple, Optional, Union
from transformers import pipeline
import torch
from tqdm import tqdm
from llm_multiprocessing_inference import get_answers
import os
from dotenv import load_dotenv

load_dotenv()

# Download NLTK data for sentence tokenization
nltk.download('punkt')

# Prompt for classification using LLM
DETAILED_PROMPT_TEMPLATE = """
You are tasked with classifying the following text based on the provided tags. \n
The tags to consider are: {tags} \n
The task is to do am multi-label classification of the text based on the tags (each tag is a label, output is 0, 1 or multiple tags). \n
The output should be a JSON list of tags that are relevant to the text. 
"""

def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class MultiStepZeroShotClassifier:
    """
    A two-stage zero-shot classifier that uses a smaller model for initial filtering
    and a larger LLM for more precise classification.
    """
    
    def __init__(
        self, 
        second_pass_model: str,
        second_pass_pipeline: str,
        second_pass_api_key: Optional[str] = None,
        first_pass_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        first_pass_threshold: float = 0.25,
        batch_size: int = 8,
        device: Optional[str] = None,
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
        self.first_pass_threshold = first_pass_threshold
        self.batch_size = batch_size
        self.second_pass_model = second_pass_model
        self.second_pass_api_key = second_pass_api_key
        self.second_pass_pipeline = second_pass_pipeline
        self.device = device or _get_device()
        self.device = torch.device(self.device)
        self.first_pass_classifier = pipeline("zero-shot-classification", model=self.first_pass_model, device=self.device)
        
    def _one_entry_first_pass_classification(
        self,
        entry: str,
        tags: List[str],
    ) -> List[str]:
        """
        Perform first pass classification on a single entry.
        
        Args:
            entry (str): The text to classify
            tags (List[str]): List of possible labels/tags
        
        Returns:
            List[str]: List of tags that are relevant to the text
        """
        sentences = nltk.sent_tokenize(entry)

        # Process in batches
        scores = {}
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            results = self.first_pass_classifier(batch, tags, multi_label=True)

            # Aggregate results
            for result in results:
                for label, score in zip(result['labels'], result['scores']):
                    if label not in scores:
                        scores[label] = []
                    scores[label].append(score)

        # Calculate the maximum score for each label
        keep_tags = [
            label
            for label, scores_list in scores.items() 
            if max(scores_list) > self.first_pass_threshold
        ]
        
        return keep_tags
        
    def first_pass_classification(
        self,
        entries: List[str],
        tags: List[str],
    ) -> List[List[str]]:
        """
        First pass classification using a smaller multilingual model.

        Args:
            entry (str): Input text to classify
            tags (List[str]): List of possible labels/tags

        Returns:
            List[List[str]]: List of tags that are relevant to the text
            
        """
        keep_tags_list = []
        for entry in tqdm(entries, desc="First pass classification"):
            keep_tags = self._one_entry_first_pass_classification(entry, tags)
            keep_tags_list.append(keep_tags)

        return keep_tags_list


    def second_pass_classification(
        self,
        entries: List[str],
        filtered_tags: List[List[str]],
    ) -> List[List[str]]:
        """
        Second pass classification using a larger LLM model with batch processing.
        
        Args:
            entry (str): Input text to classify
            filtered_tags (List[str]): List of tags from first pass
            
        Returns:
            Dict[str, float]: Dictionary of tags and their confidence scores
        """
        # Create default_response with all tags set to 0.0
        default_response_str = "[]"
        
        all_prompts = []
        entries_to_process = []
        results = []
        
        n_entries = len(entries)
        for i in range(0, n_entries):
            entry = entries[i]
            filtered_tags_one_entry = filtered_tags[i]
            
            # Skip entries with empty filtered tags and assign default response
            if not filtered_tags_one_entry:
                results.append(json.loads(default_response_str))
                continue
                
            prompt_content = DETAILED_PROMPT_TEMPLATE.format(tags=', '.join(filtered_tags_one_entry))
            all_prompts.append([{"role": "system", "content": prompt_content}, {"role": "user", "content": entry}])
            entries_to_process.append(i)
        
        # Only make API call if there are entries to process
        if all_prompts:
            api_results = get_answers(
                prompts=all_prompts,
                default_response=default_response_str,
                response_type="structured",
                api_pipeline=self.second_pass_pipeline,
                model=self.second_pass_model,
                api_key=self.second_pass_api_key,
                show_progress_bar=True,
                additional_progress_bar_description="Second pass classification"
            )
            
            # Place API results in the correct positions
            full_results = [None] * n_entries
            for idx, result in zip(entries_to_process, api_results):
                full_results[idx] = result
                
            # Fill in default responses for skipped entries
            for i in range(n_entries):
                if full_results[i] is None:
                    full_results[i] = json.loads(default_response_str)
            
            return full_results
        
        # If no entries to process, return default responses for all
        return [json.loads(default_response_str) for _ in range(n_entries)]

    def __call__(
        self, 
        entries: List[str], 
        tags: List[str]
    ) -> Dict[str, float]:
        """
        Run the complete classification pipeline.
        
        Args:
            entries (List[str]): Input text to classify
            tags (List[str]): List of possible labels/tags
            
        Returns:
            Dict[str, float]: Final classification results (from second pass)
        """
        print("===== STARTING CLASSIFICATION PIPELINE =====")
        
        # First pass
        first_pass_results = self.first_pass_classification(
            entries=entries,
            tags=tags
        )
        
        
        # Second pass
        second_pass_results = self.second_pass_classification(
            entries=entries,
            filtered_tags=first_pass_results
        )
            
        return second_pass_results

