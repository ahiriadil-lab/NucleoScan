"""Download 1L2Y from RCSB PDB and write a FASTA file."""
import os
import sys
from Bio import PDB, SeqIO
from Bio.PDB import PDBList
from Bio.SeqUtils import seq1

from config import PDB_ID, REFERENCE_PDB, REFERENCE_FASTA, DATA_DIR


def download_pdb(pdb_id: str, dest_dir: str) -> str:
    """Download PDB file directly from RCSB, return path to downloaded file."""
    import requests
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    final_path = os.path.join(dest_dir, f"{pdb_id}.pdb")
    with open(final_path, "w") as fh:
        fh.write(resp.text)
    return final_path


def clean_pdb(pdb_path: str, output_path: str, chain_id: str = "A") -> tuple:
    """
    Clean PDB file: keep only specified chain and standard amino acids.

    Parameters:
        pdb_path: Path to raw PDB file
        output_path: Path to write cleaned PDB
        chain_id: Chain letter to keep ("A", "I", "3", etc.), or None for first chain

    Returns:
        (output_path, actual_chain_letter)
    """
    from Bio.PDB import PDBParser, PDBIO, Select, is_aa

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    model = structure[0]
    chains_list = list(model.get_chains())

    # Resolve chain: if chain_id is None, use first chain
    target_chain = None
    if chain_id is None:
        target_chain = chains_list[0] if chains_list else None
    else:
        for chain in chains_list:
            if chain.get_id() == chain_id:
                target_chain = chain
                break
        if target_chain is None and chains_list:
            target_chain = chains_list[0]

    actual_chain_id = target_chain.get_id() if target_chain else "A"

    # Custom select class
    class ChainSelect(Select):
        def accept_chain(self, chain):
            return chain.get_id() == actual_chain_id

        def accept_residue(self, residue):
            # Keep standard amino acids only (no HETATM, no water)
            return is_aa(residue, standard=True)

    # Write cleaned structure
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path, ChainSelect())

    return (output_path, actual_chain_id)


def extract_fasta(pdb_path: str, pdb_id: str, fasta_path: str) -> str:
    """Extract sequence of first chain from PDB model 0 and write FASTA."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)

    model = structure[0]
    # Pick the first chain
    chain = next(iter(model.get_chains()))
    residues = [r for r in chain.get_residues() if PDB.is_aa(r, standard=True)]
    sequence = "".join(seq1(r.get_resname()) for r in residues)

    with open(fasta_path, "w") as fh:
        fh.write(f">{pdb_id}\n{sequence}\n")

    print(f"Sequence ({len(sequence)} aa): {sequence}")
    return sequence


def fetch_uniprot_sequence(accession: str, cache_dir: str = None) -> str:
    """
    Fetch the full protein sequence from the UniProt REST API.

    Returns the raw sequence string (single-letter amino acids).
    Caches the FASTA file locally to avoid repeated API calls.
    Raises RuntimeError on network failure or invalid accession.
    """
    import requests

    if cache_dir is None:
        cache_dir = os.path.join(DATA_DIR, "uniprot")
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{accession}.fasta")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path) as fh:
            raw = fh.read()
    else:
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        raw = resp.text
        with open(cache_path, "w") as fh:
            fh.write(raw)

    # Parse FASTA — skip header lines
    seq = "".join(
        line.strip() for line in raw.splitlines()
        if line.strip() and not line.startswith(">")
    )
    if not seq:
        raise RuntimeError(f"Empty sequence returned for UniProt {accession}")
    return seq


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(REFERENCE_PDB):
        print(f"{REFERENCE_PDB} already exists, skipping download.")
    else:
        print(f"Downloading {PDB_ID} …")
        download_pdb(PDB_ID, DATA_DIR)
        print(f"Saved to {REFERENCE_PDB}")

    print(f"Extracting FASTA …")
    seq = extract_fasta(REFERENCE_PDB, PDB_ID, REFERENCE_FASTA)
    print(f"FASTA written to {REFERENCE_FASTA}")
    return seq


if __name__ == "__main__":
    main()
