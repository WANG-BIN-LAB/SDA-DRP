from rdkit import Chem
from rdkit.Chem import AllChem
import os

input_file = "/data2/cxc2/data/DTI/DrugBank/smiles_cid.txt"
output_dir = "/data2/cxc2/data/DTI/DrugBank/Drug_PDB"
os.makedirs(output_dir, exist_ok=True)

with open(input_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue

        smiles, name = parts[0], parts[1]
        out_file = os.path.join(output_dir, f"{name}.pdb")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Failed SMILES: {name}")
            continue
        
        mol = Chem.AddHs(mol)

        if AllChem.EmbedMolecule(mol) != 0:
            print(f"Embedding failed: {name}")
            continue

        AllChem.UFFOptimizeMolecule(mol)

        Chem.MolToPDBFile(mol, out_file)
        print("Generated:", out_file)
