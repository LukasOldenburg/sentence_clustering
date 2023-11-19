#!/bin/bash

# Script to run the Python script for processing and analyzing Excel data.

# Define default values for the arguments
EXCEL_FILE_PATH="./data/example_data_clustering.xlsx" # Replace with your default Excel file path
COLUMN_NAMES=("Question" "Answer") # Replace with your default column names
APPROACH="all-mpnet-base-v2" # Embedding model 'bert-base-nli-mean-tokens', 'all-mpnet-base-v2', 'tfidf'
DIM_REDUCTION_METHOD="tsne" # Replace with your default dimensionality reduction method 'pca', 'tsne', 'umap'
NUM_SAMPLE_SENTENCES=3 # Replace with your default number of sample sentences given in the plot 
K_MEANS_CLUSTER=10 # Replace with your default number of K-means clusters used for sentence clustering
SEED=10 # Replace with your default seed value

# Run the Python script with the arguments
python ./cluster_sentences.py \
    --excel_file_path "$EXCEL_FILE_PATH" \
    --column_names "${COLUMN_NAMES[@]}" \
    --translate_back_plot \
    --approach $APPROACH \
    --dim_reduction_method $DIM_REDUCTION_METHOD \
    --num_sample_sentences $NUM_SAMPLE_SENTENCES \
    --k_means_cluster $K_MEANS_CLUSTER \
    --seed $SEED
