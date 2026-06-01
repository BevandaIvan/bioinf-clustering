###############################################################################
# CONSTANTS
###############################################################################

MEAN = "mean"
POSITION_MODE = "position-mode"
MODE_FOR_LARGE = "mode-for-large"
MEAN_NEAREST_SEQUENCE = "mean-nearest-sequence"

###############################################################################

from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Sequence
import encoders
import clustering
import Bio.Cluster
import hdbscan


# Returns only for large clusters (100+ sequences)
# Returns (centroids, clusterids) tuple, where e.g. clusterids[0] == "2" iff centroids[0] is the centroid of cluster 2
def clustercentroids_mode(
    encoded_sequences: np.ndarray, clusterid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    seq_freqs = defaultdict(lambda: defaultdict(int))
    for i, seq in enumerate(encoded_sequences):
        seq_freqs[clusterid[i]][
            tuple(seq)
        ] += 1  # E.g. In cluster 0, increment the counter for "ACCTGA..." once
    # Leave only the large clusters
    seq_freqs = {k: v for k, v in seq_freqs.items() if sum(v.values()) >= 100}

    centroids = []
    clusterids = []
    for k, v in seq_freqs.items():
        centroids.append(max(v, key=v.get))
        clusterids.append(k)

    centroids, clusterids = np.array(centroids), np.array(clusterids)
    return centroids, clusterids


# Returns base for each of 296 positions thats the most common in each cluster
def clustercentroids_position_mode(
    encoded_sequences: np.ndarray, clusterid: np.ndarray
) -> np.ndarray:
    unique_clusters = np.unique(clusterid)
    max_cid = int(np.max(clusterid))
    centroids = np.zeros((max_cid + 1, encoded_sequences.shape[1]))

    for cid in unique_clusters:
        cluster_seqs = encoded_sequences[clusterid == cid]
        freqs = np.sum(cluster_seqs, axis=0)
        freqs_reshaped = freqs.reshape(-1, 4)
        max_indices = np.argmax(freqs_reshaped, axis=1)
        consensus_onehot = np.zeros_like(freqs_reshaped)
        consensus_onehot[np.arange(freqs_reshaped.shape[0]), max_indices] = 1
        centroids[int(cid)] = consensus_onehot.flatten()
    return centroids


"""
Get mean of all the sequences in the cluster,
then return the sequence closest to that mean
"""


def clustercentroids_mean_nearest_sequence(
    encoded_sequences: np.ndarray, clusterid: np.ndarray
) -> np.ndarray:
    unique_clusters = np.unique(clusterid)
    max_cid = int(np.max(clusterid))
    centroids = np.zeros((max_cid + 1, encoded_sequences.shape[1]))

    for cid in unique_clusters:
        mask = clusterid == cid
        cluster_embeddings = encoded_sequences[mask]
        centroid = cluster_embeddings.mean(axis=0)
        sims = cosine_similarity(centroid.reshape(1, -1), cluster_embeddings)
        best_idx = sims.argmax()
        centroids[int(cid)] = cluster_embeddings[best_idx]

    return centroids


"""
Returns a dict like:

centroid: list[sequences in that centroid's cluster...]

NOTE: This should probably not depends on encoders, i.e.
it should take and spit out encoded sequences, not raw strings
"""


def get_centroids(
    sequences: Sequence[str],
    cluster_alg: str,
    centroids_alg: str,
    encoder: str,
    nclusters: int = 4,
) -> dict[str, list[str]]:
    if encoder == encoders.DNABERT:
        encoded_sequences = encoders.encode_bert(*sequences)
    elif encoder == encoders.ONEHOT:
        encoded_sequences = encoders.encode_onehot(*sequences)
    else:
        raise ValueError(f"Unknown encoder scheme: {encoder}")

    if cluster_alg == clustering.KMEANS:
        clusterid, _, __ = Bio.Cluster.kcluster(
            encoded_sequences, nclusters=nclusters, npass=4
        )
    elif cluster_alg == clustering.HDBSCAN:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        clusterid = clusterer.fit_predict(encoded_sequences)
        # HACK, but it's fine - singletons get removed later on anyway
        for i, id in enumerate(clusterid):
            if id == -1:
                clusterid[i] = np.max(clusterid) + 1
    else:
        raise ValueError(f"Unknown clustering algorithm: {cluster_alg}")

    if centroids_alg == MEAN_NEAREST_SEQUENCE:
        centroids = clustercentroids_mean_nearest_sequence(encoded_sequences, clusterid)
    elif centroids_alg == POSITION_MODE:
        centroids = clustercentroids_position_mode(encoded_sequences, clusterid)
    else:
        centroids, _ = Bio.Cluster.clustercentroids(
            encoded_sequences, clusterid=clusterid
        )
        if centroids_alg == MODE_FOR_LARGE:
            centroids_mode, clusterids = clustercentroids_mode(
                encoded_sequences, clusterid=clusterid
            )
            if not centroids_mode.shape == (0,) and not clusterids.shape == (0,):
                centroids[clusterids.astype(int)] = (
                    centroids_mode  # Fancy indexing (numpy thing)  # We just patch the mean centroids here with mode ones for large clusters
                )

    if encoder == encoders.DNABERT:
        centroids = encoders.decode_bert(centroids)
    else:
        centroids = encoders.decode_onehot(centroids)

    cluster_seqs = {i: [] for i in range(max(clusterid) + 1)}
    for seq, cid in zip(sequences, clusterid):
        cluster_seqs[cid].append(seq)

    return {centroid: cluster_seqs[i] for i, centroid in enumerate(centroids)}
