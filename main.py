import argparse
from io import TextIOWrapper
import clustering
import centroids
import encoders
import variants
import filemgmt


def validate_combination(
    cluster_alg: str, centroids_alg: str, n_clusters: int, encoder: str
):
    if encoder == encoders.DNABERT:
        assert centroids_alg == centroids.MEAN_NEAREST_SEQUENCE
    if cluster_alg == clustering.KMEANS:
        assert isinstance(n_clusters, int) and n_clusters >= 1


def write_pretty(s: str, f: TextIOWrapper) -> None:
    padding = (80 - len(s) - 2) // 2
    f.write(f"{"="*padding} {s} {"="*padding}\n")


def write_detailed_report(
    gene_variants: set[str],
    analyzed_samples: dict[str, dict[str, list[str]]],
    dest: str,
) -> None:
    with open(dest, "w") as f:
        write_pretty("AGGREGATE VARIANTS", f)
        for variant in gene_variants:
            f.write(variant + "\n")
        for sample in analyzed_samples.keys():
            write_pretty(f"VARIANTS FOUND IN {sample}", f)
            for variant in analyzed_samples[sample]["variants"]:
                f.write(variant + "\n")
                f.write("-" * 80 + "\n")
            for i, cluster_centroid in enumerate(analyzed_samples[sample]):
                # HACK
                if cluster_centroid == "variants":
                    continue
                cluster_sequences = analyzed_samples[sample][cluster_centroid]
                write_pretty(f"{sample} CLUSTER {i}", f)
                f.write(f"CENTROID: {cluster_centroid}\n")
                f.write("-" * 80 + "\n")
                for seq in cluster_sequences:
                    f.write(seq + "\n")
                    f.write("-" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("datadir")
    parser.add_argument("--cluster-alg", type=str, default=clustering.KMEANS)
    parser.add_argument("--centroids-alg", type=str, default=centroids.MEAN)
    parser.add_argument("--threshold", type=int, default=4)
    parser.add_argument(
        "-d", "--detailed-report-dest", type=str, default="detailed.txt"
    )
    parser.add_argument("--encoder", type=str, default=encoders.ONEHOT)
    parser.add_argument("-n", "--n-clusters", type=int, default=4)
    parser.add_argument("dest", type=str, default="variants.fasta")

    args = parser.parse_args()
    validate_combination(
        args.cluster_alg, args.centroids_alg, args.n_clusters, args.encoder
    )

    # See comment above function signature in variants.py for explanation
    gene_variants, analyzed_samples = variants.get_gene_variants(
        args.datadir,
        threshold=args.threshold,
        cluster_alg=args.cluster_alg,
        centroids_alg=args.centroids_alg,
        encoder=args.encoder,
        per_sample=args.detailed_report_dest,
        nclusters=args.n_clusters,
    )

    filemgmt.write_records(args.dest, list(gene_variants))
    if args.detailed_report_dest:
        write_detailed_report(
            gene_variants, analyzed_samples, args.detailed_report_dest
        )
