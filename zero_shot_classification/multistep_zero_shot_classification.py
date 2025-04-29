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
You are tasked with classifying the following text based on the provided tags.
The tags to consider are: {tags} .
The task is to do a multi-label classification of the text based on the tags (each tag is a label, output is 0, 1 or multiple tags).
If unsure about a tag, return it anyways. High recall is crucial in this task.
The output is a JSON list of tags that are relevant to the text. Do not return any other text than the JSON list.
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
        first_pass_model: str = "MoritzLaurer/bge-m3-zeroshot-v2.0",
        first_pass_threshold: Optional[float] = 0.25,
        do_second_pass: bool = True,
        second_pass_model: str = None,
        second_pass_pipeline: str = None,
        second_pass_api_key: Optional[str] = None,
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
        self.do_second_pass = do_second_pass
        if do_second_pass:
            self.second_pass_model = second_pass_model
            self.second_pass_api_key = second_pass_api_key
            self.second_pass_pipeline = second_pass_pipeline
        else:
            self.second_pass_model = None
            self.second_pass_api_key = None
            self.second_pass_pipeline = None

        self.device = device or _get_device()
        self.device = torch.device(self.device)
        self.first_pass_classifier = pipeline("zero-shot-classification", model=self.first_pass_model, device=self.device)
        
    def _one_entry_first_pass_classification(
        self,
        entries: List[str],
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
        hypothesis_template = "This text contains relevant information about {}"
        results = []
        for i in tqdm(range (0, len(entries), self.batch_size), desc="First pass classification"):
            batch_entries = entries[i:i+self.batch_size]
            batch_results = self.first_pass_classifier(batch_entries, tags, hypothesis_template=hypothesis_template, multi_label=True, batch_size=self.batch_size)
            results.extend(batch_results)
        return results


        
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
        split_entries = [nltk.sent_tokenize(entry) for entry in entries]
        # now i want to keep the indices all sentnces respectively before flattening the list so i can retreieve the entry-wise results
        entries_indices = []
        final_inputs = []
        
        for i, sentences in enumerate(split_entries):
            for j, one_sentence in enumerate(sentences):
                entries_indices.append(i)
                final_inputs.append(one_sentence)
                
        results = self._one_entry_first_pass_classification(final_inputs, tags)
        
        per_entry_results = [[] for _ in range(len(entries))]
        for entry_idx, sent_idx in enumerate(entries_indices):
            per_entry_results[sent_idx].append(results[entry_idx])
            
        final_results = []
        for i, one_entry_results in enumerate(per_entry_results):
            scores = {}
            for result in one_entry_results:
                for label, score in zip(result['labels'], result['scores']):
                    if label not in scores:
                        scores[label] = []
                    scores[label].append(score)
            scores = {k: max(v) for k, v in scores.items()}
            final_results.append(scores)
            
        if self.first_pass_threshold is not None:
            # Calculate the maximum score for each label
            keep_tags = [[
                label
                for label, scores_list in final_results[i].items() 
                if max(scores_list) > self.first_pass_threshold
            ] for i in range(len(final_results))]
            return keep_tags
        else:
            return final_results
                    
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
            List[List[str]]: List of tags that are relevant to the text
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
    ) -> Union[List[List[str]], List[Dict[str, float]]]:
        """
        Run the complete classification pipeline.
        
        Args:
            entries (List[str]): Input text to classify
            tags (List[str]): List of possible labels/tags
            
        Returns:
            List[List[str]]: Final classification results (from second pass)
        """
        print("===== STARTING CLASSIFICATION PIPELINE =====")
        
        # First pass
        first_pass_results = self.first_pass_classification(
            entries=entries,
            tags=tags
        )
        
        if self.do_second_pass:
            # Second pass
            second_pass_results = self.second_pass_classification(
                entries=entries,
                filtered_tags=first_pass_results
            )
            
            return second_pass_results
        else:
            return first_pass_results

