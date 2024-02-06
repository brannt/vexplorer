import json
import sys
from pathlib import Path

import dash
import faiss
import numpy as np
import plotly.graph_objs as go
from dash import Input, Output, State, dcc, html
from scipy.cluster.vq import kmeans, vq
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import MDS, TSNE

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


class ClusteredIndex:
    def __init__(self, faiss_index, doc_storage) -> None:
        self._faiss_index = faiss_index
        self._doc_storage = doc_storage
        self._centroids = None
        self._cluster_labels = None

    @classmethod
    def from_index_dir(cls, index_dir):
        index_filename = index_dir / "faiss.index"
        doc_storage = index_dir / "docstore.json"
        index = faiss.read_index(str(index_filename))
        with doc_storage.open("r") as f:
            chunks, indices = json.load(f)
        chunks = dict(chunks)
        return cls(index, {int(i): chunks[guid] for i, guid in indices.items()})

    @property
    def np_index(self):
        return np.array(self._faiss_index.reconstruct_n(0, self._faiss_index.ntotal))

    def get_index_3d(self, method="naive"):
        # Take first 3 dimensions
        return self.np_index[:3]

    def cluster(self, num_clusters):
        np_index = self.np_index
        centroids, _ = kmeans(np_index, num_clusters)
        cluster_labels, _ = vq(np_index, centroids)
        self._centroids = centroids
        self._cluster_labels = cluster_labels

    def get_point_cluster(self, point_index):
        cluster_id = self.cluster_labels[point_index]
        # Retrieve the centroid for the clicked cluster
        centroid = self.centroids[cluster_id]
        return centroid, cluster_id

    def get_closest_chunks(self, vector, k=3):
        # 2d vector
        vector = vector.reshape(1, -1)
        _, labels = self._faiss_index.search(vector, k)
        return [self._doc_storage[i] for i in labels[0]]

    @property
    def centroids(self):
        if self._centroids is None:
            raise ValueError("Must cluster before accessing centroids")
        return self._centroids

    @property
    def cluster_labels(self):
        if self._cluster_labels is None:
            raise ValueError("Must cluster before accessing cluster labels")
        return self._cluster_labels


def reduce_to_3d(np_index, method):
    if method == "tsne":
        model = TSNE(n_components=3, perplexity=20)
    elif method == "mds":
        model = MDS(n_components=3, n_init=1, max_iter=100)
    elif method == "pca":
        model = PCA(n_components=3)
    elif method == "nmf":
        model = NMF(n_components=3)
    else:
        raise ValueError(f"Unrecognized method: {method}")
    return model.fit_transform(np_index)


def make_app(index_dir, num_clusters):
    index = None

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(
        [
            html.H1("VExplorer"),
            html.Div(
                [
                    dcc.Input(
                        id="dirpath",
                        type="text",
                        placeholder="Select an index directory",
                        value=index_dir or "",
                        size="70",
                    ),
                    dcc.Input(
                        id="num-clusters",
                        type="number",
                        placeholder="Number of clusters",
                        value=num_clusters,
                        size="1",
                    ),
                    dcc.Dropdown(
                        id="reduce-method",
                        options=[
                            {"label": "TSNE", "value": "tsne"},
                            {"label": "MDS", "value": "mds"},
                            {"label": "PCA", "value": "pca"},
                            {"label": "NMF", "value": "nmf"},
                        ],
                        value="tsne",
                        clearable=False,
                        style={
                            "display": "inline-block",
                            "vertical-align": "top",
                            "width": "100px",
                        },
                    ),
                ]
            ),
            dcc.Graph(
                id="cluster-plot",
            ),
            html.Pre(id="click-data"),
        ]
    )

    @app.callback(
        Output("cluster-plot", "figure"),
        [
            Input("dirpath", "value"),
            Input("num-clusters", "value"),
            Input("reduce-method", "value"),
        ],
    )
    def update_graph(dirpath, num_clusters, reduce_method="tsne"):
        index_dir = Path(dirpath)
        nonlocal index
        index = ClusteredIndex.from_index_dir(index_dir)
        index.cluster(num_clusters)
        index_3d = reduce_to_3d(index.np_index, method=reduce_method)
        return {
            "data": [
                go.Scatter3d(
                    x=index_3d[:, 0],
                    y=index_3d[:, 1],
                    z=index_3d[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=index.cluster_labels,
                        colorscale="Viridis",
                        opacity=0.8,
                    ),
                )
            ],
            "layout": go.Layout(
                margin={"l": 0, "r": 0, "b": 0, "t": 0},
                scene={
                    "xaxis": {"title": "X"},
                    "yaxis": {"title": "Y"},
                    "zaxis": {"title": "Z"},
                },
            ),
        }

    # Callback for handling clicks on the cluster plot
    @app.callback(
        Output("click-data", "children"),
        [Input("cluster-plot", "clickData")],
    )
    def display_click_data(clickData):
        if clickData is None:
            return "Click on a cluster to see the top-3 closest chunks."

        # Retrieve the point index from the clicked data

        point_index = clickData["points"][0]["pointNumber"]
        # Find the cluster id for the clicked point
        centroid, cluster_id = index.get_point_cluster(point_index)
        closest_chunks = index.get_closest_chunks(centroid, 3)

        # Format the output
        output = "\n---\n".join(
            ["Top-3 closest chunks to the clicked cluster:"]
            + [d["pageContent"] for d in closest_chunks]
        )
        return output

    return app


if __name__ == "__main__":
    try:
        index_dir = sys.argv[1]
        num_clusters = int(sys.argv[2])
    except IndexError:
        index_dir, num_clusters = None, 3
    make_app(index_dir, num_clusters).run_server(debug=True)
