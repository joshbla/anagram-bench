"""
Script to analyze and visualize results from an existing Excel file.
Useful for re-creating visualizations or further analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from pathlib import Path


def analyze_results(excel_file):
    """Analyze results from an existing Excel file."""
    print(f"Analyzing results from: {excel_file}")
    
    # Read the summary sheet
    summary_df = pd.read_excel(excel_file, sheet_name='Summary')
    examples_df = pd.read_excel(excel_file, sheet_name='All Examples')
    
    print("\n" + "="*60)
    print("OVERALL PERFORMANCE SUMMARY")
    print("="*60)
    
    # Sort by accuracy
    summary_df = summary_df.sort_values('Accuracy (%)', ascending=False)
    
    print("\nTop 5 Models by Overall Accuracy:")
    for idx, row in summary_df.head().iterrows():
        print(f"  {row['Model']}: {row['Accuracy (%)']:.1f}% "
              f"({row['Total Correct']}/{row['Total Attempts']})")
    
    print("\nBottom 5 Models by Overall Accuracy:")
    for idx, row in summary_df.tail().iterrows():
        print(f"  {row['Model']}: {row['Accuracy (%)']:.1f}% "
              f"({row['Total Correct']}/{row['Total Attempts']})")
    
    # Analyze by response time
    print("\nFastest Models (by average response time):")
    fastest = summary_df.nsmallest(5, 'Avg Response Time (s)')
    for idx, row in fastest.iterrows():
        print(f"  {row['Model']}: {row['Avg Response Time (s)']:.2f}s")
    
    # JSON compliance
    print("\nJSON Compliance Rates:")
    for idx, row in summary_df.iterrows():
        print(f"  {row['Model']}: {row['JSON Compliance (%)']:.1f}%")
    
    # Analyze by word length
    print("\n" + "="*60)
    print("PERFORMANCE BY WORD LENGTH")
    print("="*60)
    
    # Find columns for word length accuracy
    length_cols = [col for col in summary_df.columns if col.endswith('-letter accuracy (%)')]
    
    if length_cols:
        # Calculate average accuracy by word length across all models
        avg_by_length = {}
        for col in length_cols:
            avg_by_length[col] = summary_df[col].mean()
        
        print("\nAverage Accuracy Across All Models by Word Length:")
        for col, avg in sorted(avg_by_length.items()):
            print(f"  {col}: {avg:.1f}%")
        
        # Find which word lengths are hardest/easiest
        easiest = max(avg_by_length.items(), key=lambda x: x[1])
        hardest = min(avg_by_length.items(), key=lambda x: x[1])
        
        print(f"\nEasiest word length: {easiest[0]} ({easiest[1]:.1f}% average accuracy)")
        print(f"Hardest word length: {hardest[0]} ({hardest[1]:.1f}% average accuracy)")
    
    # Analyze specific examples
    print("\n" + "="*60)
    print("EXAMPLE ANALYSIS")
    print("="*60)
    
    # Find most commonly correctly solved words
    correct_examples = examples_df[examples_df['Is Valid'] == True]
    word_success_rate = correct_examples.groupby('Original Word').size()
    total_attempts_per_word = examples_df.groupby('Original Word').size()
    success_rates = (word_success_rate / total_attempts_per_word * 100).sort_values(ascending=False)
    
    print("\nWords Most Successfully Converted to Anagrams:")
    for word, rate in success_rates.head(10).items():
        attempts = total_attempts_per_word[word]
        successes = word_success_rate.get(word, 0)
        print(f"  '{word}': {rate:.1f}% ({successes}/{attempts})")
    
    print("\nWords Least Successfully Converted to Anagrams:")
    for word, rate in success_rates.tail(10).items():
        attempts = total_attempts_per_word[word]
        successes = word_success_rate.get(word, 0)
        print(f"  '{word}': {rate:.1f}% ({successes}/{attempts})")
    
    # Show some interesting anagram examples
    print("\nInteresting Valid Anagram Examples:")
    interesting = correct_examples.sample(n=min(10, len(correct_examples)))
    for idx, row in interesting.iterrows():
        print(f"  '{row['Original Word']}' → '{row['Generated Anagram']}' (by {row['Model']})")
    
    # Create additional visualizations
    create_additional_plots(summary_df, examples_df, excel_file)


def create_additional_plots(summary_df, examples_df, source_file):
    """Create additional visualization plots."""
    output_dir = Path(source_file).parent
    
    # 1. Bar chart of overall accuracy
    plt.figure(figsize=(12, 8))
    summary_df_sorted = summary_df.sort_values('Accuracy (%)', ascending=True)
    models = [m.split('/')[-1] for m in summary_df_sorted['Model']]
    accuracies = summary_df_sorted['Accuracy (%)']
    
    colors = plt.cm.RdBu(accuracies / 100)
    plt.barh(models, accuracies, color=colors)
    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.title('Overall Anagram Generation Accuracy by Model', fontsize=14)
    plt.xlim(0, 100)
    
    # Add value labels
    for i, (model, acc) in enumerate(zip(models, accuracies)):
        plt.text(acc + 1, i, f'{acc:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Response time vs accuracy scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(summary_df['Avg Response Time (s)'], 
                summary_df['Accuracy (%)'],
                s=100, alpha=0.6)
    
    # Add labels for each point
    for idx, row in summary_df.iterrows():
        model_name = row['Model'].split('/')[-1]
        plt.annotate(model_name, 
                    (row['Avg Response Time (s)'], row['Accuracy (%)']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    plt.xlabel('Average Response Time (seconds)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Performance: Accuracy vs Response Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_response_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Word length distribution of successes
    correct_by_length = examples_df[examples_df['Is Valid'] == True].groupby('Word Length').size()
    total_by_length = examples_df.groupby('Word Length').size()
    
    plt.figure(figsize=(10, 6))
    x = sorted(total_by_length.index)
    width = 0.35
    
    plt.bar([i - width/2 for i in x], 
            [total_by_length.get(i, 0) for i in x],
            width, label='Total Attempts', alpha=0.7)
    plt.bar([i + width/2 for i in x], 
            [correct_by_length.get(i, 0) for i in x],
            width, label='Successful Anagrams', alpha=0.7)
    
    plt.xlabel('Word Length', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Anagram Generation Success by Word Length', fontsize=14)
    plt.legend()
    plt.xticks(x)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'success_by_word_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nAdditional plots saved to: {output_dir}")


def main():
    """Main function to run the analysis."""
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
    else:
        # Try to find the most recent results file
        from pathlib import Path
        results_files = list(Path('.').glob('anagram_benchmark_results_*.xlsx'))
        if results_files:
            excel_file = str(max(results_files, key=lambda f: f.stat().st_mtime))
            print(f"No file specified, using most recent: {excel_file}")
        else:
            print("Usage: python analyze_results.py <excel_file>")
            print("Or run without arguments to analyze the most recent results file.")
            return
    
    if not Path(excel_file).exists():
        print(f"Error: File '{excel_file}' not found.")
        return
    
    analyze_results(excel_file)


if __name__ == "__main__":
    main()