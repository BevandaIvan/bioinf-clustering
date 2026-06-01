from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from typing import Callable
from pathlib import Path


def read_records(
    filename: str, record_filter: Callable[[str], bool] = lambda r: len(r.seq) == 296
) -> list[str]:
    records = []
    extension = filename.split(".")[-1]
    for record in SeqIO.parse(filename, extension):
        if record_filter(record):
            records.append(str(record.seq))
    return records


def write_records(filename: str, sequences: list[str]) -> None:
    if not filename.endswith(".fasta"):
        filename += ".fasta"
    records = [SeqRecord(Seq(s), id=f"variant_{i}") for i, s in enumerate(sequences)]
    SeqIO.write(records, filename, "fasta")


def get_paths(
    dir: str, file_filter: Callable[[str], bool] = lambda p: p.name.startswith("J")
) -> list[str]:
    paths = list(Path(dir).rglob("*.fastq"))
    return [str(p) for p in paths if file_filter(p)]
