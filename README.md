
# Sentence Processing Toolkit

This repository contains tools for natural language processing, specifically focusing on clustering sentences and translating text. 
It provides a streamlined approach for handling large volumes of text data in form of e.g. questions or answers. 
The text data is embedded and grouped into clusters which is exemplary shown below:

[Example clustering results](results/cluster_analysis_Question.pdf)

## Features

The toolkit is divided into two core functionalities:

### 1. Sentence Clustering with Embeddings


The sentence clustering functionality uses embeddings to group sentences based on their semantic similarity. This is particularly useful for analyzing and categorizing large sets of textual data, such as customer feedback or survey responses.

- **Script:** `run_sentence_clustering.sh`

This component leverages advanced natural language processing techniques to generate embeddings for sentences and then applies clustering algorithms to group these sentences into meaningful clusters and visualize them.

### 2. (Optional) Translation of Sentences

For projects that involve multilingual datasets, particularly those requiring translation from German to English, this functionality is a valuable asset.

- **Script:** `run_translate.sh`


## Getting Started

To get started with this toolkit:

1. **Clone the Repository:**
   ```
   git clone https://github.com/LukasOldenburg/sentence_clustering.git
   ```

2. **Install Dependencies:**
   Ensure you have Python installed, along with necessary libraries. A `requirements.yml` file is included for easy setup with conda.
   ```
   cd /path_to_repo
   conda env create --file requirements.yml
   ```

3. **Run the Scripts:**
   - For sentence clustering:
     ```
     bash run_sentence_clustering.sh
     ```
   - For sentence translation (optional):
     ```
     bash run_translate.sh
     ```


