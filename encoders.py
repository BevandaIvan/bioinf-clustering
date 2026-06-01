###############################################################################
# CONSTANTS
###############################################################################

ONEHOT = "onehot"
DNABERT = "dnabert"

###############################################################################

import numpy as np
from bert_utils import BertEncoder
from typing import Sequence


def encode_onehot(*sequences: str) -> np.ndarray:
    letter_encodings = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
    }
    encoded_sequences = []
    for seq in sequences:
        encoded_sequence = []
        for letter in seq:
            encoded_sequence += letter_encodings[letter]
        encoded_sequences.append(encoded_sequence)
    return np.array(encoded_sequences)


def decode_onehot(sequences: np.ndarray) -> list[str]:
    letter_encodings = {
        "A": np.array([1, 0, 0, 0]),
        "C": np.array([0, 1, 0, 0]),
        "G": np.array([0, 0, 1, 0]),
        "T": np.array([0, 0, 0, 1]),
    }

    def find_closest(vector: np.ndarray) -> str:
        min_dist = float("inf")
        closest_letter = None
        for letter, encoding in letter_encodings.items():
            dist = 1 / 4 * np.sum((encoding - vector) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_letter = letter
        return closest_letter

    # Unoptimized!
    decoded_sequences = []
    for seq in sequences:
        decoded_sequence = ""
        assert len(seq) % 4 == 0
        num_letters = len(seq) // 4
        for i in range(num_letters):
            letter = find_closest(seq[i * 4 : i * 4 + 4])
            decoded_sequence += letter
        decoded_sequences.append(decoded_sequence)
    return decoded_sequences


def encode_bert(*sequences: str) -> np.ndarray:
    if getattr(encode_bert, "encoder", None) is None:
        encode_bert.encoder = BertEncoder()
    return encode_bert.encoder.encode_sequences(sequences, save=True)


def decode_bert(encoded_sequences: Sequence[np.ndarray]) -> list[str]:
    if getattr(encode_bert, "encoder", None) is None:
        raise Exception("decode_bert called before encoding")
    return encode_bert.encoder.decode_sequences(
        encoded_sequences, assert_all_decodable=True
    )
