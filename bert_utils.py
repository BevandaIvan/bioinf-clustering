import transformers
import torch
import numpy as np
from typing import Sequence

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
Taken from https://github.com/jerryji1993/DNABERT/tree/master

@article{ji2021dnabert,
    author = {Ji, Yanrong and Zhou, Zhihan and Liu, Han and Davuluri, Ramana V},
    title = "{DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome}",
    journal = {Bioinformatics},
    volume = {37},
    number = {15},
    pages = {2112-2120},
    year = {2021},
    month = {02},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btab083},
    url = {https://doi.org/10.1093/bioinformatics/btab083},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/37/15/2112/50578892/btab083.pdf},
}


@misc{zhou2023dnabert2,
      title={DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome}, 
      author={Zhihan Zhou and Yanrong Ji and Weijian Li and Pratik Dutta and Ramana Davuluri and Han Liu},
      year={2023},
      eprint={2306.15006},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}
}
"""


def seq2kmer(seq, k=6):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x : x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers


class BertEncoder:
    def __init__(self, model_name="zhihan1996/DNA_bert_6"):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = transformers.AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        ).to(DEVICE)

        # Could be implemented much better
        self.memory = {}

    def _encode_sequence(self, sequence: str, save: bool = False) -> np.ndarray:
        kmers = seq2kmer(sequence)
        inputs = self.tokenizer(kmers, add_special_tokens=True, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state
        embedding = hidden.mean(dim=1).cpu().numpy().squeeze(0)

        if save:
            self.memory[sequence] = embedding

        return embedding

    # Horribly unoptimized
    def encode_sequences(
        self, sequences: Sequence[str], save: bool = False
    ) -> np.ndarray:
        encs = []
        for sequence in sequences:
            enc = self._encode_sequence(sequence, save)
            encs.append(enc)

        return np.array(encs)

    def _decode_sequence(self, encoded_sequence: np.ndarray) -> str | None:
        for k, v in self.memory.items():
            if np.allclose(v, encoded_sequence, atol=1e-6):
                return k
        return None

    # Again, unoptimized
    def decode_sequences(
        self, encoded_sequences: Sequence[np.ndarray], assert_all_decodable: bool = True
    ) -> list[str]:
        decoded = []
        for seq in encoded_sequences:
            decoded.append(self._decode_sequence(seq))

        if assert_all_decodable:
            assert (
                None not in decoded
            ), "Sequences could not be decoded — were they encoded with save=True?"
        return decoded
