Listed below are instructions for running this code locally on Linux machines. For other platforms, please modify the commands for creating and activating a virtual environment accordingly.

0. Clone the repository and change into the root directory:
```
git clone https://github.com/BevandaIvan/bioinf-clustering.git
cd bioinf-clustering
```

1. Create a Python virtual environment:
```
python -m venv .venv
```

2. Activate the environment:
```
source .venv/bin/activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Run the code and wait (the following uses all available options for illustrative purposes):
The first argument (hdbscan_mns_th5_variants.fasta) is the file where variants (found across all samples) will be written.
The argument to -d (hdbscan_mns_th5_detailed.txt) is the optional destination for a detailed report, listing the clusters & variants for each sample individually. Those do get quite long due to cluster output, though.
```
python main.py ./data/ hdbscan_mns_th5_variants.fasta -d hdbscan_mns_th5_detailed.txt --cluster-alg hdbscan --centroids-alg mean-nearest-sequence --threshold 5 --encoder onehot
```
hdbscan_mns_th5_variants.fasta


Given below are the possible values for each options. Please note that some may not be supported, but the program will notify you in that case.

encoder:
- onehot
- dnabert

cluster-alg:
- hdbscan
- kmeans

centroids-alg:
- mean
- mode-for-large
- position-mode
- mean-nearest-sequence

threshold: any integer greater than 0