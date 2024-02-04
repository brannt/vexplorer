# Document Clustering Dash App

This Dash application performs K-means clustering on a vector index of embedded document chunks and visualizes the results using a Plotly 3D scatter plot. Users can interact with the visualization by clicking on clusters to see the top-3 chunks closest to the cluster's centroid.

## Features

- K-means clustering on document embeddings.
- 3D scatter plot visualization of clusters.
- Interactive cluster inspection in the UI.
- Support for reading FAISS index files.

## Installation

To set up the project environment, you need to install the required dependencies:

```sh
pip install -r requirements.txt
```

## Usage

To run the app, use the following command:

```sh
python clustering_app.py <index_dir> <num_clusters>
```

Replace `<index_dir>` with the path to the directory containing the FAISS index and document storage, and `<num_clusters>` with the desired number of clusters.

Enjoy exploring your document clusters!
