# Anagram Benchmark for Language Models

This project benchmarks various language models on their ability to generate valid anagrams using the OpenRouter API. It tests 15 different models across words of varying lengths (3-10 letters) and evaluates their performance.

## Features

- Tests 15 different language models via OpenRouter API
- Uses NLTK's word corpus to generate test words
- Validates anagrams for correctness
- Enforces strict JSON response format
- Saves detailed results to Excel (XLSX) file
- Creates a visual heatmap showing model performance by word length
- Tracks response times and JSON compliance rates

## Requirements

- Python 3.7+
- OpenRouter API key
- Required packages (see `requirements.txt`)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/anagram-bench.git
cd anagram-bench
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

4. Add your OpenRouter API key to the `.env` file:
```
OPENROUTER_API_KEY=your_api_key_here
```

## Usage

Run the benchmark:
```bash
python anagram_benchmark.py
```

The benchmark will:
1. Generate 5 random words for each length (3-10 letters)
2. Test each of the 15 models on all words
3. Validate if the generated anagrams are correct
4. Save results to an Excel file with timestamp
5. Generate a heatmap visualization

## Output Files

The script generates two output files:

1. **Excel file** (`anagram_benchmark_results_YYYYMMDD_HHMMSS.xlsx`):
   - Summary sheet: Overall performance metrics for each model
   - All Examples sheet: Detailed results for every test case

2. **Heatmap** (`anagram_benchmark_heatmap_YYYYMMDD_HHMMSS.png`):
   - Visual representation of accuracy by model and word length
   - Color scale from red (poor performance) to blue (excellent performance)

## Models Tested

The benchmark tests the following 15 models:
- OpenAI: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- Anthropic: Claude-3.5-sonnet, Claude-3-haiku
- Google: Gemini-2.0-flash-exp, Gemini-pro-1.5
- Meta: Llama-3.3-70b, Llama-3.1-405b
- Mistral: Mistral-large
- Others: Grok-2, Qwen-2.5-72b, Phi-3-medium, Hermes-3, DeepSeek-chat

## Benchmark Metrics

For each model, the benchmark tracks:
- **Accuracy**: Percentage of valid anagrams generated
- **Response Time**: Average time to generate responses
- **JSON Compliance**: Percentage of responses in correct JSON format
- **Performance by Word Length**: Accuracy for each word length category

## Customization

You can customize the benchmark by modifying these constants in `anagram_benchmark.py`:

```python
MIN_WORD_LENGTH = 3      # Minimum word length to test
MAX_WORD_LENGTH = 10     # Maximum word length to test
WORDS_PER_LENGTH = 5     # Number of words to test per length
```

You can also modify the `MODELS` list to test different models available on OpenRouter.

## How It Works

1. **Word Generation**: Random words are selected from NLTK's word corpus
2. **Prompt Engineering**: Each model receives a carefully crafted prompt requesting an anagram in JSON format
3. **API Calls**: Uses OpenRouter's structured output feature to enforce JSON responses
4. **Validation**: Checks if the response:
   - Contains the same letters as the original word
   - Is different from the original word
   - Is in valid JSON format
5. **Analysis**: Results are aggregated and visualized

## Example Results

The heatmap shows model performance across different word lengths:
- **Blue cells**: High accuracy (close to 100%)
- **White cells**: Medium accuracy (around 50%)
- **Red cells**: Low accuracy (close to 0%)

## Troubleshooting

- **API Key Error**: Ensure your OpenRouter API key is correctly set in the `.env` file
- **Rate Limiting**: The script includes a 0.5-second delay between API calls
- **Network Issues**: Check your internet connection and proxy settings if applicable
- **NLTK Data**: The script will automatically download the required NLTK word corpus on first run

## License

This project is open source and available under the MIT License.