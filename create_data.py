import os
import pandas as pd
import numpy as np
from rdkit import Chem
import networkx as nx
from utils import TestbedDataset

# ==== helpers ====
ELEMENTS = ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B',
            'V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H','Li','Ge','Cu',
            'Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']

sequence_vocabulary = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
sequence_dict = {ch: i+1 for i, ch in enumerate(sequence_vocabulary)}
max_sequence_length = 1000

def encode_one_hot_unknown(value, allowable_set):
    if value not in allowable_set:
        value = allowable_set[-1]
    return [value == s for s in allowable_set]

def get_atom_features(atom):
    return np.array(
        encode_one_hot_unknown(atom.GetSymbol(), ELEMENTS) +
        encode_one_hot_unknown(atom.GetDegree(), list(range(11))) +
        encode_one_hot_unknown(atom.GetTotalNumHs(), list(range(11))) +
        encode_one_hot_unknown(atom.GetValence(getExplicit=False), list(range(11))) +
        [atom.GetIsAromatic()]
    )

def convert_smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    atom_count = mol.GetNumAtoms()
    features = [get_atom_features(atom) / sum(get_atom_features(atom)) for atom in mol.GetAtoms()]
    edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    graph = nx.Graph(edges).to_directed()
    edge_index = [[u, v] for u, v in graph.edges]
    return atom_count, features, edge_index

def encode_sequence(protein):
    encoded = np.zeros(max_sequence_length)
    for i, aa in enumerate(protein[:max_sequence_length]):
        encoded[i] = sequence_dict.get(aa, 0)
    return encoded

# ==== main ====
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    # 1. Load KIBA CSV
    df = pd.read_csv("kiba_all.csv")
    print(f"Loaded KIBA dataset with {len(df)} rows")

    # 2. Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/kiba_train.csv", index=False)
    test_df.to_csv("data/kiba_test.csv", index=False)

    # 3. Build graph dictionary
    compound_smiles = set(train_df["compound_iso_smiles"]).union(test_df["compound_iso_smiles"])
    smile_graphs = {sm: convert_smile_to_graph(sm) for sm in compound_smiles}

    # 4. Prepare tensors
    def prepare_split(split_df):
        xd = split_df["compound_iso_smiles"].tolist()
        xt = [encode_sequence(seq) for seq in split_df["target_sequence"].tolist()]
        y = split_df["affinity"].values
        return xd, xt, y

    train_xd, train_xt, train_y = prepare_split(train_df)
    test_xd, test_xt, test_y = prepare_split(test_df)

    # 5. Save to .pt using TestbedDataset
    os.makedirs("data/processed", exist_ok=True)
    print("Preparing data/processed/kiba_train.pt...")
    TestbedDataset(root="data", dataset="kiba_train", xd=train_xd, xt=train_xt, y=train_y, smile_graph=smile_graphs)
    print("Preparing data/processed/kiba_test.pt...")
    TestbedDataset(root="data", dataset="kiba_test", xd=test_xd, xt=test_xt, y=test_y, smile_graph=smile_graphs)
    print("✅ Saved kiba_train.pt and kiba_test.pt")
