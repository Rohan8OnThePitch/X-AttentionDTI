import torch
from rdkit import Chem
from rdkit.Chem import rdmolops


def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]


def atom_features(atom):
    allowed_atomic_nums = [6, 7, 8, 16, 9, 17, 35, 53, 15, 11, 19, 20, 12, 30]
    atom_num = atom.GetAtomicNum()
    atom_feat = one_hot(atom_num, allowed_atomic_nums + ["Misc"])

    atom_feat += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, "Misc"])
    atom_feat += one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2, "Misc"])

    hybridization = atom.GetHybridization()
    atom_feat += one_hot(hybridization, [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        "Misc"
    ])

    atom_feat += [int(atom.GetIsAromatic())]
    atom_feat += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, "Misc"])

    # 🔥 ADD THIS BACK (this was missing)
    try:
        chirality = atom.GetChiralTag()
        atom_feat += one_hot(chirality, [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            "Misc"
        ])
    except:
        atom_feat += [1, 0, 0, 0]

    atom_feat += [atom.GetTotalValence()]
    atom_feat += [atom.GetMass() * 0.01]
    atom_feat += [int(atom.IsInRing())]

    return atom_feat


def build_drug_tensors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    node_feats = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()],
        dtype=torch.float,
    )

    hyperedges = []
    hyperedge_types = []

    # Bond hyperedges
    for bond in mol.GetBonds():
        hyperedges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        hyperedge_types.append(0)

    # Ring hyperedges
    for ring in Chem.GetSymmSSSR(mol):
        hyperedges.append(list(ring))
        hyperedge_types.append(1)

    # Radius=2 environment hyperedges
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        env = rdmolops.FindAtomEnvironmentOfRadiusN(mol, radius=2, rootedAtAtom=idx)
        atoms = set()
        for bond_id in env:
            bond = mol.GetBondWithIdx(bond_id)
            atoms.add(bond.GetBeginAtomIdx())
            atoms.add(bond.GetEndAtomIdx())
        if len(atoms) > 1:
            hyperedges.append(list(atoms))
            hyperedge_types.append(2)

    node_indices = []
    hedge_indices = []

    for hedge_id, nodes in enumerate(hyperedges):
        for n in nodes:
            node_indices.append(n)
            hedge_indices.append(hedge_id)

    hyperedge_indices = torch.tensor(
        [node_indices, hedge_indices],
        dtype=torch.long
    )

    hyperedge_types = torch.tensor(hyperedge_types, dtype=torch.long)

    # Single sample → batch size = 1
    batch_indices = torch.zeros(node_feats.size(0), dtype=torch.long)

    return node_feats, hyperedge_indices, hyperedge_types, batch_indices