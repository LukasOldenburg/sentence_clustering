import argparse
import pandas as pd
from utils import process_sentences, set_deterministic_mode

def main():
    parser = argparse.ArgumentParser(description="Process and analyze Excel data.")
    
    parser.add_argument('--excel_file_path', type=str, required=True, 
                        help='Path to the Excel file.')
    parser.add_argument('--column_names', type=str, nargs='+', required=True,
                        help='List of column names to process.')
    parser.add_argument('--translate_back_plot', action='store_true',
                        help='Enable or disable translating back for plotting.')
    parser.add_argument('--approach', type=str, choices=['bert-base-nli-mean-tokens', 'all-mpnet-base-v2', 'tfidf'],
                        required=True, help='Approach for processing sentences.')
    parser.add_argument('--dim_reduction_method', type=str, choices=['tsne', 'pca', 'umap'],
                        required=True, help='Method for dimensionality reduction.')
    parser.add_argument('--num_sample_sentences', type=int, required=True,
                        help='Number of sample sentences to process.')
    parser.add_argument('--k_means_cluster', type=int, required=True,
                        help='Number of clusters for K-means.')
    parser.add_argument('--seed', type=int, default=10, 
                        help='Seed for deterministic mode.')

    args = parser.parse_args()

    set_deterministic_mode(seed=args.seed)

    data = pd.read_excel(args.excel_file_path)

    process_sentences(data=data,
                      approach=args.approach,
                      dim_red=args.dim_reduction_method,
                      num_sample_sentences=args.num_sample_sentences,
                      column_names=args.column_names,
                      k_means_cluster=args.k_means_cluster,
                      translate_back=args.translate_back_plot)



if __name__ == '__main__':
    main()
