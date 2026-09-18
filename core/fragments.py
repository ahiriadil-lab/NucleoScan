"""Generate N-terminal (and optionally C-terminal) fragment FASTA files."""
import os

from config import (
    REFERENCE_FASTA, FRAGMENTS_DIR,
    MIN_FRAGMENT_LENGTH, MAX_FRAGMENT_LENGTH, PDB_ID,
    SLIDING_WINDOW_SIZES, SLIDING_WINDOW_STEP,
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
        N-terminal: fragment_N05.fasta  (also fragment_05.fasta for legacy compat)
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


def write_sliding_fragment_fasta(sequence: str, start: int, window_size: int,
                                  out_dir: str) -> str:
    """
    Write a FASTA for an internal (sliding) fragment seq[start:start+window_size].

    File name: fragment_S{start:03d}_W{window_size:02d}.fasta
    """
    fragment = sequence[start:start + window_size]
    fasta_path = os.path.join(out_dir, f"fragment_S{start:03d}_W{window_size:02d}.fasta")
    header = f">{PDB_ID}_fragS{start}_W{window_size}"
    with open(fasta_path, "w") as fh:
        fh.write(f"{header}\n{fragment}\n")
    return fasta_path


def generate_sliding_fragments(sequence: str, out_dir: str,
                                window_sizes: list = None,
                                step: int = None) -> list:
    """
    Generate all sliding/internal fragment FASTAs.

    Produces seq[i:i+W] for each W in window_sizes and each valid start i.
    Start positions: 0, step, 2*step, ... (while i+W <= len(sequence)).

    Parameters
    ----------
    sequence     : full protein sequence string
    out_dir      : output directory for FASTA files
    window_sizes : list[int] — window widths (default: SLIDING_WINDOW_SIZES from config)
    step         : int — step size (default: SLIDING_WINDOW_STEP from config)

    Returns list of written FASTA paths.
    """
    if window_sizes is None:
        window_sizes = SLIDING_WINDOW_SIZES
    if step is None:
        step = SLIDING_WINDOW_STEP
    if step < 1:
        step = 1

    n = len(sequence)
    paths = []
    for ws in sorted(set(window_sizes)):
        if ws > n:
            continue
        for start in range(0, n - ws + 1, step):
            path = write_sliding_fragment_fasta(sequence, start, ws, out_dir)
            paths.append(path)
            frag_seq = sequence[start:start + ws]
            print(f"  fragment_S{start:03d}_W{ws:02d}.fasta  ({ws} aa, pos {start}-{start+ws-1}): {frag_seq}")
    return paths


def main(direction: str = None):
    """
    Generate fragment FASTAs for lengths MIN_FRAGMENT_LENGTH..MAX_FRAGMENT_LENGTH.

    direction : 'N'       — N-terminal only
                'C'       — C-terminal only
                'both'    — both N-terminal and C-terminal
                'sliding' — internal sliding window fragments
                None      — use FRAGMENT_DIRECTION from config (default)
    """
    # Use config default if direction not specified
    if direction is None:
        direction = FRAGMENT_DIRECTION

    os.makedirs(FRAGMENTS_DIR, exist_ok=True)

    if not os.path.exists(REFERENCE_FASTA):
        raise FileNotFoundError(
            f"Reference FASTA not found: {REFERENCE_FASTA}\n"
            "Run download_reference.py first."
        )

    sequence = read_fasta(REFERENCE_FASTA)
    full_length = len(sequence)
    max_len = min(MAX_FRAGMENT_LENGTH, full_length)

    # Sliding window mode
    if direction == "sliding":
        paths = generate_sliding_fragments(sequence, FRAGMENTS_DIR)
        print(f"\n{len(paths)} sliding fragment FASTAs written to {FRAGMENTS_DIR}/")
        return paths

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
    ap.add_argument("--direction", default="N", choices=["N", "C", "both", "sliding"],
                    help="Fragment direction: N-terminal, C-terminal, or both")
    args = ap.parse_args()
    main(direction=args.direction)
