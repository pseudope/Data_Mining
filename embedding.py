import os

import torch
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from preprocess import get_csv

def load_model():

    """
    We use SBERT for making embeddings.
    https://www.sbert.net/index.html#
    """

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def load_data(file_path):

    # Check if the file exists
    if not os.path.exists(file_path):
        # If the file does not exist, call the get_csv method
        print("Creating CSV from JSON file.")
        get_csv()
    else:
        print("The file already exists.")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    return df

def get_embedding(df, file_path='arxiv-metadata-oai-snapshot.csv',
                  embeddings_path='embeddings.npy'):
    
    model = load_model()
    df = load_data(file_path)

    # Check if the file exists
    if not os.path.exists(embeddings_path):
        # If the file does not exist, call the get_embedding method
        print("Creating CSV from JSON file.")
        corpus_embeddings = model.encode(df["abstract"], show_progress_bar=True)
        np.save("./embeddings.npy", corpus_embeddings, allow_pickle=True)
    else:
        print("The file already exists.")
        corpus_embeddings = np.load("./embeddings.npy", allow_pickle=True)

    print(corpus_embeddings.shape)
    
    return corpus_embeddings

def clustering(corpus_embeddings, num_clusters):

    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    return cluster_assignment


def main():
    pass