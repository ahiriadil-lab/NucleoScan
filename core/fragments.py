"""Generate N-terminal (and optionally C-terminal) fragment FASTA files."""
import os

from config import (
    REFERENCE_FASTA, FRAGMENTS_DIR,
    MIN_FRAGMENT_LENGTH, MAX_FRAGMENT_LENGTH, PDB_ID,
    FRAGMENT_DIRECTION,
)


def read_fasta(fasta_path: str) -> str:
    """Return the sequence string from a single-record FASTA file."""
    sequence = []
    with open(fasta_path) as fh:
        for line in fh:
            if not line.startswith(">"):
                sequence.append(line.strip())
    return "".join(sequence)


def write_fragment_fasta(sequence: str, length: int, out_dir: str,
                         direction: str = "N") -> str:
    """
    Write a FASTA for a fragment of given length, return path.

    direction='N' : N-terminal fragment (residues 1..length)
    direction='C' : C-terminal fragment (residues N-length+1..N)

    File names:
        N-terminal: fragment_05.fasta
        C-terminal: fragment_C05.fasta
    """
    if direction == "C":
        fragment = sequence[-length:]
        fasta_path = os.path.join(out_dir, f"fragment_C{length:02d}.fasta")
        header = f">{PDB_ID}_fragC{length}"
    else:
        fragment = sequence[:length]
        fasta_path = os.path.join(out_dir, f"fragment_{length:02d}.fasta")
        header = f">{PDB_ID}_frag{length}"

    with open(fasta_path, "w") as fh:
        fh.write(f"{header}\n{fragment}\n")
    return fasta_path


def main(direction: str = None):
    """
    Generate fragment FASTAs for lengths MIN_FRAGMENT_LENGTH..MAX_FRAGMENT_LENGTH.

    direction : 'N'       — N-terminal only
                'C'       — C-terminal only
                'both'    — both N-terminal and C-terminal
                None      — use FRAGMENT_DIRECTION from config (default)
    """
    # Use config default if direction not specified
    if direction is None:
        direction = FRAGMENT_DIRECTION

    os.makedirs(FRAGMENTS_DIR, exist_ok=True)

    if not os.path.exists(REFERENCE_FASTA):
        raise FileNotFoundError(
            f"Reference FASTA not found: {REFERENCE_FASTA}"
        )

    sequence = read_fasta(REFERENCE_FASTA)
    full_length = len(sequence)
    max_len = min(MAX_FRAGMENT_LENGTH, full_length)

    dirs = []
    if direction in ("N", "both"):
        dirs.append("N")
    if direction in ("C", "both"):
        dirs.append("C")

    paths = []
    for d in dirs:
        for length in range(MIN_FRAGMENT_LENGTH, max_len + 1):
            path = write_fragment_fasta(sequence, length, FRAGMENTS_DIR, direction=d)
            paths.append(path)
            frag_seq = sequence[:length] if d == "N" else sequence[-length:]
            prefix = "C" if d == "C" else ""
            print(f"  fragment_{prefix}{length:02d}.fasta  ({length} aa, {d}-term): {frag_seq}")

    print(f"\n{len(paths)} fragment FASTAs written to {FRAGMENTS_DIR}/")
    return paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="N", choices=["N", "C", "both"],
                    help="Fragment direction: N-terminal, C-terminal, or both")
    args = ap.parse_args()
    main(direction=args.direction)
