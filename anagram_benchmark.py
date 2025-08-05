import os
import json
import random
import time
from typing import List, Dict, Tuple, Optional
from collections import Counter
import requests
from dotenv import load_dotenv
import nltk
from nltk.corpus import words
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
from check_openrouter_limits import get_openrouter_rate_limits

# Load environment variables from .env.local
env_path = os.path.join(os.path.dirname(__file__), '.env.local')
load_dotenv(env_path)

# Download NLTK data if not already present
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Constants
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
print(f"OPENROUTER_API_KEY loaded: {bool(OPENROUTER_API_KEY)}")
if OPENROUTER_API_KEY:
    print(f"API Key starts with: {OPENROUTER_API_KEY[:10]}...")
else:
    print("WARNING: OPENROUTER_API_KEY is None!")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Output folder configuration
OUTPUT_FOLDER = "benchmark_outputs"

# Dictionary validation configuration
CHECK_DICTIONARY = True  # Set to False to allow non-dictionary words in anagrams

# Top models from OpenRouter this week
MODELS = [
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-flash",
    "google/gemini-2.0-flash-001",
    "deepseek/deepseek-chat-v3-0324",
    "google/gemini-2.5-pro",
    "qwen/qwen3-coder",
    "anthropic/claude-3.7-sonnet",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-r1-0528",
    "openrouter/horizon-alpha",
    "google/gemini-flash-1.5",
    "openai/gpt-4o-mini",
    "mistralai/mistral-nemo",
    "openai/gpt-4.1",
    "google/gemini-2.5-flash-lite-preview-06-17",
    "openai/gpt-4.1-mini",
    "anthropic/claude-opus-4",
    "moonshotai/kimi-k2"
]

# Word lengths to test (e.g., 8-letter words up to 16-letter words)
MIN_WORD_LENGTH = 5
MAX_WORD_LENGTH = 15
WORDS_PER_LENGTH = 20

# Get rate limits from OpenRouter
RATE_LIMIT_REQUESTS, RATE_LIMIT_INTERVAL = get_openrouter_rate_limits()
if RATE_LIMIT_REQUESTS and RATE_LIMIT_INTERVAL:
    # Use 90% of the rate limit to be safe
    MAX_CONCURRENT_REQUESTS = int(RATE_LIMIT_REQUESTS * 0.9)
    print(f"OpenRouter rate limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_INTERVAL}")
    print(f"Using 90% of limit: {MAX_CONCURRENT_REQUESTS} concurrent requests")
else:
    # Fallback if we can't get rate limits
    MAX_CONCURRENT_REQUESTS = 5
    print("Could not fetch rate limits, using default of 5 concurrent requests")


class AnagramBenchmark:
    def __init__(self, check_dictionary=True):
        """Initialize the benchmark with word dictionary and results storage.
        
        Args:
            check_dictionary: If True, validate that anagram words exist in dictionary.
                            If False, only check that letters match.
        """
        self.word_list = [w.lower() for w in words.words() if w.isalpha() and w.islower()]
        self.check_dictionary = check_dictionary
        self.results = []
        self.word_samples = {}
        self.semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.results_lock = threading.Lock()
        
        # Create timestamped output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = os.path.join(OUTPUT_FOLDER, f"run_{timestamp}")
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"Output folder: {os.path.abspath(self.output_folder)}")
        
    def generate_word_samples(self) -> Dict[int, List[str]]:
        """Generate random word samples for each word length."""
        print("Generating word samples...")
        
        for length in range(MIN_WORD_LENGTH, MAX_WORD_LENGTH + 1):
            # Filter words by length
            words_of_length = [w for w in self.word_list if len(w) == length]
            
            # Sample words randomly
            if len(words_of_length) >= WORDS_PER_LENGTH:
                self.word_samples[length] = random.sample(words_of_length, WORDS_PER_LENGTH)
            else:
                self.word_samples[length] = words_of_length[:WORDS_PER_LENGTH]
                
            print(f"  Length {length}: {self.word_samples[length]}")
            
        return self.word_samples
    
    def is_valid_anagram(self, original: str, anagram: str) -> bool:
        """Check if the provided word is a valid anagram of the original."""
        # Store the original anagram for word validation
        anagram_with_spaces = anagram.lower()
        
        # Remove spaces and convert to lowercase for character comparison
        original_clean = original.lower().replace(" ", "")
        anagram_clean = anagram.lower().replace(" ", "")
        
        # Check if it's the same word (not a valid anagram)
        if original_clean == anagram_clean:
            return False
            
        # Check if they have the same character counts
        original_counter = Counter(original_clean)
        anagram_counter = Counter(anagram_clean)
        
        if original_counter != anagram_counter:
            return False
        
        # If dictionary checking is enabled, verify all words are real English words
        if self.check_dictionary:
            anagram_words = anagram_with_spaces.split()
            word_set = set(self.word_list)  # Convert to set for O(1) lookup
            
            for word in anagram_words:
                if word not in word_set:
                    return False
        
        return True
    
    def create_prompt(self, word: str) -> str:
        """Create a prompt for the AI model to generate an anagram."""
        prompt = f"""Generate an anagram of the word: "{word}"

An anagram is a word or phrase formed by rearranging the letters of the original word, using all letters exactly once.

IMPORTANT: You must respond ONLY with valid JSON in this exact format:
{{
  "anagram": "your_anagram_here"
}}

The anagram must:
1. Use all letters from "{word}" exactly once
2. Be different from the original word
3. Be a real word or phrase

Respond with ONLY the JSON object, no other text."""
        
        return prompt
    
    def call_openrouter_api(self, model: str, word: str) -> Tuple[Optional[str], float, bool]:
        """
        Call OpenRouter API to generate an anagram.
        Returns: (anagram, response_time, is_valid_json)
        """
        # Debug: Check if API key is present
        if not OPENROUTER_API_KEY:
            print(f"\n  ERROR: OPENROUTER_API_KEY is None or empty when calling API!")
            print(f"  Current env value: {os.getenv('OPENROUTER_API_KEY')}")
            return None, 0, False
            
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "anagram-benchmark",
            "X-Title": "Anagram Benchmark",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": self.create_prompt(word)
                }
            ],
            "response_format": {
                "type": "json_object"
            }
        }
        
        try:
            start_time = time.time()
            response = requests.post(OPENROUTER_BASE_URL, headers=headers, data=json.dumps(data))
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                try:
                    # Parse JSON response
                    json_content = json.loads(content)
                    anagram = json_content.get('anagram', '')
                    return anagram, response_time, True
                except json.JSONDecodeError:
                    print(f"  JSON decode error for {model}: {content}")
                    return content, response_time, False
            else:
                print(f"  API error for {model}: {response.status_code} - {response.text}")
                return None, response_time, False
                
        except Exception as e:
            print(f"  Exception for {model}: {str(e)}")
            return None, 0, False
    
    def test_single_word(self, model: str, word: str, length: int) -> dict:
        """Test a single word with a model and return the result."""
        with self.semaphore:
            anagram, response_time, is_json_compliant = self.call_openrouter_api(model, word)
        
        is_valid = False
        if anagram:
            is_valid = self.is_valid_anagram(word, anagram)
        
        return {
            'word': word,
            'anagram': anagram,
            'is_valid': is_valid,
            'response_time': response_time,
            'is_json_compliant': is_json_compliant,
            'length': length
        }
    
    def test_single_model(self, model: str, model_idx: int):
        """Test a single model on all words in parallel."""
        print(f"\nTesting Model {model_idx}/{len(MODELS)}: {model}")
        print("-" * 40)
        
        model_results = {
            'model': model,
            'results_by_length': {},
            'total_correct': 0,
            'total_attempts': 0,
            'avg_response_time': 0,
            'json_compliance_rate': 0
        }
        
        # Collect all words to test
        all_words = []
        for length in range(MIN_WORD_LENGTH, MAX_WORD_LENGTH + 1):
            for word in self.word_samples[length]:
                all_words.append((word, length))
        
        print(f"  Testing {len(all_words)} words in parallel (max {MAX_CONCURRENT_REQUESTS} concurrent)...")
        
        # Test all words in parallel
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            futures = []
            for word, length in all_words:
                future = executor.submit(self.test_single_word, model, word, length)
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"\n  Error testing word: {str(e)}")
        
        # Process results
        response_times = []
        json_compliant_count = 0
        
        # Initialize results by length
        for length in range(MIN_WORD_LENGTH, MAX_WORD_LENGTH + 1):
            model_results['results_by_length'][length] = {
                'correct': 0,
                'total': 0,
                'examples': []
            }
        
        # Process each result
        for result in results:
            length = result['length']
            response_times.append(result['response_time'])
            
            if result['is_json_compliant']:
                json_compliant_count += 1
            
            length_results = model_results['results_by_length'][length]
            length_results['total'] += 1
            
            if result['is_valid']:
                length_results['correct'] += 1
                model_results['total_correct'] += 1
            
            model_results['total_attempts'] += 1
            
            # Store example
            length_results['examples'].append({
                'word': result['word'],
                'anagram': result['anagram'],
                'is_valid': result['is_valid'],
                'response_time': result['response_time'],
                'is_json_compliant': result['is_json_compliant']
            })
        
        # Print results summary by length
        print("\n  Results by word length:")
        for length in range(MIN_WORD_LENGTH, MAX_WORD_LENGTH + 1):
            length_data = model_results['results_by_length'][length]
            if length_data['total'] > 0:
                accuracy = length_data['correct'] / length_data['total'] * 100
                print(f"    {length}-letter words: {length_data['correct']}/{length_data['total']} ({accuracy:.1f}%)")
        
        # Calculate summary statistics
        model_results['avg_response_time'] = np.mean(response_times) if response_times else 0
        model_results['json_compliance_rate'] = json_compliant_count / model_results['total_attempts'] if model_results['total_attempts'] > 0 else 0
        
        # Thread-safe append to results
        with self.results_lock:
            self.results.append(model_results)
        
        print(f"\nModel Summary: {model_results['total_correct']}/{model_results['total_attempts']} correct "
              f"({model_results['total_correct']/model_results['total_attempts']*100:.1f}%)")
    
    def run_benchmark(self):
        """Run the complete benchmark across all models and words in parallel."""
        # Generate word samples first
        self.generate_word_samples()
        
        print("\n" + "="*60)
        print("Starting Anagram Benchmark (Parallel Mode)")
        print(f"Rate limit: {MAX_CONCURRENT_REQUESTS} concurrent requests")
        total_words = sum(len(words) for words in self.word_samples.values())
        print(f"Testing {len(MODELS)} models with {total_words} words each = {len(MODELS) * total_words} total API calls")
        print("="*60 + "\n")
        
        # Clear results in case of re-run
        self.results = []
        
        # Create thread pool for parallel execution
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            # Submit all models for testing
            futures = []
            for model_idx, model in enumerate(MODELS, 1):
                future = executor.submit(self.test_single_model, model, model_idx)
                futures.append((future, model))
            
            # Wait for all models to complete
            print("\nAll models submitted. Waiting for completion...\n")
            
            completed = 0
            for future, model in futures:
                try:
                    future.result()  # Wait for completion
                    completed += 1
                    print(f"Progress: {completed}/{len(MODELS)} models completed")
                except Exception as e:
                    print(f"Error testing model {model}: {str(e)}")
        
        # Sort results by model order (since they may complete out of order)
        self.results.sort(key=lambda x: MODELS.index(x['model']))
    
    def save_to_excel(self, filename: str = None):
        """Save all results and examples to an Excel file."""
        if filename is None:
            filename = "anagram_benchmark_results.xlsx"
        
        # Save to output folder
        filepath = os.path.join(self.output_folder, filename)
        
        print(f"\nSaving results to {filepath}...")
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for result in self.results:
                row = {
                    'Model': result['model'],
                    'Total Correct': result['total_correct'],
                    'Total Attempts': result['total_attempts'],
                    'Accuracy (%)': result['total_correct'] / result['total_attempts'] * 100 if result['total_attempts'] > 0 else 0,
                    'Avg Response Time (s)': result['avg_response_time'],
                    'JSON Compliance (%)': result['json_compliance_rate'] * 100
                }
                
                # Add accuracy by word length
                for length in range(MIN_WORD_LENGTH, MAX_WORD_LENGTH + 1):
                    if length in result['results_by_length']:
                        length_data = result['results_by_length'][length]
                        accuracy = length_data['correct'] / length_data['total'] * 100 if length_data['total'] > 0 else 0
                        row[f'{length}-letter accuracy (%)'] = accuracy
                
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Examples sheet
            examples_data = []
            for result in self.results:
                for length, length_results in result['results_by_length'].items():
                    for example in length_results['examples']:
                        examples_data.append({
                            'Model': result['model'],
                            'Word Length': length,
                            'Original Word': example['word'],
                            'Generated Anagram': example['anagram'],
                            'Is Valid': example['is_valid'],
                            'Is JSON Compliant': example['is_json_compliant'],
                            'Response Time (s)': example['response_time']
                        })
            
            examples_df = pd.DataFrame(examples_data)
            examples_df.to_excel(writer, sheet_name='All Examples', index=False)
            
        print(f"Results saved to {filepath}")
        return filepath
    
    def create_heatmap(self, save_path: str = None):
        """Create a heatmap visualization of model performance by word length."""
        if save_path is None:
            save_path = "anagram_benchmark_heatmap.png"
        
        # Save to output folder
        save_path = os.path.join(self.output_folder, save_path)
        
        print(f"\nCreating heatmap visualization...")
        
        # Prepare data for heatmap
        models = []
        heatmap_data = []
        
        for result in self.results:
            models.append(result['model'].split('/')[-1])  # Get just the model name
            row = []
            
            for length in range(MIN_WORD_LENGTH, MAX_WORD_LENGTH + 1):
                if length in result['results_by_length']:
                    length_data = result['results_by_length'][length]
                    accuracy = length_data['correct'] / length_data['total'] * 100 if length_data['total'] > 0 else 0
                    row.append(accuracy)
                else:
                    row.append(0)
            
            heatmap_data.append(row)
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        # Convert to numpy array
        data_array = np.array(heatmap_data)
        
        # Create custom colormap (red to blue, where blue is higher)
        cmap = plt.cm.RdBu
        
        # Create heatmap
        sns.heatmap(data_array, 
                    annot=True, 
                    fmt='.1f',
                    cmap=cmap,
                    xticklabels=[f'{i}-letter' for i in range(MIN_WORD_LENGTH, MAX_WORD_LENGTH + 1)],
                    yticklabels=models,
                    cbar_kws={'label': 'Accuracy (%)'},
                    vmin=0,
                    vmax=100)
        
        plt.title('Anagram Generation Accuracy by Model and Word Length', fontsize=16, pad=20)
        plt.xlabel('Word Length', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Heatmap saved to {save_path}")
        return save_path


def main():
    """Main function to run the anagram benchmark."""
    # Check for API key
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenRouter API key.")
        return
    
    # Create and run benchmark
    benchmark = AnagramBenchmark(check_dictionary=CHECK_DICTIONARY)
    print(f"Dictionary checking: {'ENABLED' if CHECK_DICTIONARY else 'DISABLED'}")
    print(f"All outputs will be saved to: {os.path.abspath(benchmark.output_folder)}/\n")
    
    try:
        # Run the benchmark
        benchmark.run_benchmark()
        
        # Save results
        excel_file = benchmark.save_to_excel()
        
        # Create visualization
        heatmap_file = benchmark.create_heatmap()
        
        print("\n" + "="*60)
        print("Benchmark Complete!")
        print(f"Results saved to: {excel_file}")
        print(f"Heatmap saved to: {heatmap_file}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n\nError during benchmark: {str(e)}")
        raise


if __name__ == "__main__":
    main()