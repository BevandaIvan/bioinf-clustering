import centroids
import filemgmt
import clustering
from collections import defaultdict
from tqdm import tqdm

"""
So, this is a bit of a messy one.
It returns two things:
1. variants - a set of variants found accross ALL samples for
the given parameters
2. (if per_sample == True) a dict like:
    sample_filename_1: {
        variants: [variant strings...],
        centroid_string_1: [sequences in that centroid's cluster...],
        centroid_string_2: [sequences in that centroid's cluster...],
    },

NOTE: This should probably have a flag to suppress tqdm
"""


# Assumes .fastq
def get_gene_variants(
    dir: str,
    threshold: int,
    cluster_alg: str,
    centroids_alg: str,
    encoder: str,
    per_sample: bool,
    nclusters: int = 4,
) -> tuple[set[str], dict[str, dict[str, list[str]]]]:
    variants = set()
    sample_memory = {}

    paths = filemgmt.get_paths(dir)

    # For centroids of size [2, 99], we check in how many samples they appear
    # If >= threshold, we add it to the list of gene variants
    cent_occurences = defaultdict(int)
    for p in tqdm(paths):
        sequences = filemgmt.read_records(p)
        if sequences == []:
            continue
        # NOTE: This check should probably not be here...
        if len(sequences) < 2 or (
            len(sequences) < nclusters and cluster_alg == clustering.KMEANS
        ):
            continue
        cents = centroids.get_centroids(
            sequences,
            cluster_alg=cluster_alg,
            centroids_alg=centroids_alg,
            encoder=encoder,
            nclusters=nclusters,
        )

        if per_sample:
            sample_memory[str(p)] = cents

        cents = {c: l for c, l in cents.items() if len(l) > 1}  # Removing singletons
        definite_variants = {
            c for c, s in cents.items() if len(s) >= 100
        }  # >= 100 are surely actual gene variants

        variants |= definite_variants
        cents = {c: s for c, s in cents.items() if c not in definite_variants}
        for c in cents.keys():
            cent_occurences[c] += 1

    # Thresholding
    for c, cnt in cent_occurences.items():
        if cnt >= threshold:
            variants.add(c)

    if per_sample:
        for sample, cents in sample_memory.items():
            sample_variants = set(cents.keys())
            sample_memory[sample]["variants"] = sample_variants.intersection(variants)

    return variants, sample_memory
