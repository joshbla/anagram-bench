"""
Quick test script to verify the anagram benchmark is working.
Tests only 3 models with 2 words per length for faster results.
"""

import os
from dotenv import load_dotenv
from anagram_benchmark import AnagramBenchmark

# Load environment variables from .env.local
env_path = os.path.join(os.path.dirname(__file__), '.env.local')
load_dotenv(env_path)

# Configuration flag for dictionary checking
CHECK_DICTIONARY = True  # Set to False to allow non-dictionary words in anagrams

# Override constants for quick testing
import anagram_benchmark
anagram_benchmark.MODELS = [
    "anthropic/claude-opus-4.1"
]
anagram_benchmark.MIN_WORD_LENGTH = 19
anagram_benchmark.MAX_WORD_LENGTH = 20
anagram_benchmark.WORDS_PER_LENGTH = 1


def main():
    """Run a quick test of the anagram benchmark."""
    # Check for API key
    env_key = os.getenv('OPENROUTER_API_KEY')
    if not env_key:
        print("Error: OPENROUTER_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenRouter API key.")
        return
    
    print("Running quick test with 1 model and fewer words...")
    print(f"Dictionary checking: {'ENABLED' if CHECK_DICTIONARY else 'DISABLED'}")
    print("This is just to verify everything is working correctly.\n")
    
    # Create and run benchmark
    benchmark = AnagramBenchmark(check_dictionary=CHECK_DICTIONARY)
    
    try:
        # Run the benchmark
        benchmark.run_benchmark()
        
        # Save results with quick_test prefix
        excel_file = benchmark.save_to_excel("quick_test_results.xlsx")
        
        # Create visualization with quick_test prefix
        heatmap_file = benchmark.create_heatmap("quick_test_heatmap.png")
        
        print("\n" + "="*60)
        print("Quick Test Complete!")
        print(f"Results saved to: {excel_file}")
        print(f"Heatmap saved to: {heatmap_file}")
        print("All anagrams have been saved to individual sheets in the Excel file.")
        print("="*60)
        
    except Exception as e:
        print(f"\n\nError during quick test: {str(e)}")
        raise


if __name__ == "__main__":
    main()