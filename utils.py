import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
from sklearn.manifold import TSNE
from matplotlib.patches import FancyBboxPatch
import random
import textwrap
import umap
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import os

def process_sentences(data, approach, column_names, translate_back, dim_red='pca', num_sample_sentences=2, k_means_cluster=10):
    
    data.fillna("No data", inplace=True)
    for name in column_names:
        original_sentences = data[name.replace("_translated", "")] if translate_back else None
        sentences = data[name]
        embeddings = generate_embeddings(sentences, 
                                        approach)
        kmeans = KMeans(n_clusters=k_means_cluster, 
                        random_state=0)
        clusters = kmeans.fit_predict(embeddings)
        plot_clusters(embeddings, 
                    clusters, 
                    sentences, 
                    method=dim_red, 
                    original_sentences=original_sentences,
                    num_examples=num_sample_sentences,
                    column_name=name)
        data[f"{name}_cluster_idx"] = pd.Series(clusters)

        clustered_data = data.groupby(f"{name}_cluster_idx")
        if not os.path.exists(f"./results/{name}"):
            os.makedirs(f"./results/{name}")
        for cluster_idx, cluster in clustered_data:
            sents = cluster[name.replace("_cluster_idx", "")].tolist()
            file_name = f'./results/{name}/cluster_{cluster_idx}.txt'
            with open(file_name, 'w') as file:
                for sent in sents:
                    file.write(sent + '\n')
            print(f'Datei {file_name} wurde erstellt.')

    data.to_excel("./results/cluster_results.xlsx")



def generate_embeddings(sentences, approach):
    
    sentences = sentences.to_list()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {approach} to create sentence embeddings")
    print(f"Inference on device: {device}")

    if approach == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words='english')
        embeddings = vectorizer.fit_transform(sentences).toarray()
    elif approach == 'bert-base-nli-mean-tokens':
        model_name = "sentence-transformers/bert-base-nli-mean-tokens"
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        print(f"Embedding model parameters: {sum(p.numel() for p in model.parameters())}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer(sentences, padding=True, return_tensors="pt", max_length=512)
        tokens.to(device)
        with torch.no_grad():
            model_output = model(**tokens)
            embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()
    elif approach == 'all-mpnet-base-v2':
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        model.to(device)
        print(f"Embedding model parameters: {sum(p.numel() for p in model.parameters())}")
        embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)
        embeddings = embeddings.cpu().numpy()
    else:
        raise ValueError("Unsupported approach")
    return embeddings



def plot_clusters(embeddings, labels, sentences, column_name, original_sentences=None, method='pca', num_examples=2):
    # Dimensionality reduction method
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, learning_rate=200)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2)
    else:
        raise ValueError("Invalid method. Choose 'pca', 'tsne', or 'umap'.")

    reduced_embeddings = reducer.fit_transform(embeddings)

    font_size=16

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.rcParams.update({'font.size': font_size}) # Update the font size for all plot elements

    # Assuming reduced_embeddings is your data and labels are your cluster labels
    unique_labels = set(labels)
    markers = ['o', 's', 'v', '<', '>', 'p', '8','D']
    colors = [
    "#ff0000", "#008000", "#0000ff", "#ffff00", "#ffa500",
    "#800080", "#00ffff", "#ff00ff", "#00ff00", "#008080",
    "#ff00ff", "#800000", "#808000", "#000080", "#808080",
    "#ff8c00", "#dc143c", "#228b22", "#4169e1", "#ffd700",
    "#9400d3", "#f08080", "#40e0d0", "#a0522d", "#da70d6"
    ]
    random.shuffle(colors)
    # Plot each cluster with a randomly chosen marker and color
    for label_idx, label in enumerate(unique_labels):
        cluster_data = reduced_embeddings[labels == label]
        marker = random.choice(markers)
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                    label=label, 
                    marker=marker,
                    color=colors[label_idx] if len(unique_labels) <= 25 else random.choice(colors),
                    s=50)
    
    # Mapping from point in plot to question number
    show_points_in_plot = True
    point_to_question = {}
    box_width = 140
    for i in range(len(set(labels))):  # For each cluster
        cluster_indices = [j for j, x in enumerate(labels) if x == i]
        random_indices = random.sample(cluster_indices, min(num_examples, len(cluster_indices)))
        for idx, index in enumerate(random_indices):
            point_number = f'{i}.{idx}'
            if show_points_in_plot:
                plt.text(reduced_embeddings[index, 0], reduced_embeddings[index, 1], point_number, fontsize=8)
            text = original_sentences if original_sentences is not None else sentences
            wrapped_text = textwrap.fill(text.iloc[index], width=box_width) 
            point_to_question[point_number] = wrapped_text

    expl_var = f"\nExplained Variance: {reducer.explained_variance_ratio_.sum()}" if method=="pca" else ""
    plt.title(f"Clusters with {method.upper()}{expl_var}")
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")

    # Get unique cluster labels and their counts
    _, label_counts = np.unique(labels, return_counts=True)
    legend_labels = [f'Cluster {label} ({count} instances)' for label, count in zip(unique_labels, label_counts)]
    plt.legend(title='Cluster', loc='upper left', bbox_to_anchor=(1.05, 1), labels=legend_labels, ncol=2 if len(set(labels)) > 25 else 1)


    # Determine the last question for each cluster
    last_questions = {key.split('.')[0]: key for key in point_to_question.keys()}
    # Build the text string
    textstr = "Sample questions for each cluster:\n\n"
    for p, q in point_to_question.items():
        cluster = p.split('.')[0]
        textstr += f"{p}: {q}"
        if p == last_questions[cluster]:
            textstr += '\n' + '-' * box_width + '\n'
        else:
            textstr += '\n'
            
    # Define the properties of the new text box
    props = dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor="black")

    # Create the textbox at the bottom of the plot, with the width matching the x-axis length
    fig.text(0, -0.05, textstr, fontsize=font_size, verticalalignment='top', 
             horizontalalignment='left', bbox=props, wrap=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    if not os.path.exists(f"./results"):
            os.makedirs(f"./results")
    plt.savefig(f'./results/cluster_analysis_{column_name}.pdf', format='pdf', bbox_inches='tight')


def set_deterministic_mode(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # If using CUDA, make it deterministic
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def translate_sentences(excel_file_path, column_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_excel(excel_file_path)

    for column in column_names:
        #  Load the translation model and tokenizer
        model_name = 'Helsinki-NLP/opus-mt-de-en'
        model = MarianMTModel.from_pretrained(model_name)
        model.to(device)
        print(f"Translation model parameters: {sum(p.numel() for p in model.parameters())}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        german_sentences = data[column].tolist()

        # Translate sentences
        english_sentences = []
        for idx, sentence in enumerate(tqdm(german_sentences, desc="Translate sentences into english ...")):
            # Tokenize the sentence
            if sentence is None or sentence == '' or pd.isna(sentence):
                english_sentences.append(sentence)  
            else:
                tokenized_text = tokenizer(sentence, return_tensors="pt")
                tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}
                # Translate and decode the sentence
                translated = model.generate(**tokenized_text)
                translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                english_sentences.append(translated_text)
        data[f"{column}_translated"] = english_sentences

    return data

