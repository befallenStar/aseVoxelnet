# -*- encoding: utf-8 -*-
from ase import Atoms
from ase.data.pubchem import pubchem_conformer_search

from loader.pointclouds_to_voxelgrid import *
from loader.visualize_npy import visualization

colors = {1: (1, 1, 1), 6: (.5, .5, .5), 7: (0, 0, 1),
          8: (1, 1, 0), 16: (1, 0, 0)}
atoms_vector = [1, 6, 7, 8, 16]


def load_atoms(cid=None, smiles=None) -> Atoms:
    inputs = [cid, smiles]
    input_check = [a is not None for a in inputs]
    if input_check.count(True) == 1:
        index = input_check.index(True)
        if index == 0:
            compounds = pubchem_conformer_search(cid=cid)
        else:
            compounds = pubchem_conformer_search(smiles=smiles)
    else:
        return Atoms()
    return compounds[0].atoms


def load(atoms: Atoms):
    print(atoms.symbols)
    pointcloud = []
    for step, number in enumerate(atoms.numbers):
        point = atoms.positions[step].tolist()
        # point.extend(colors[number])
        point.append(number)
        pointcloud.append(point)
    return pointcloud


def load_voxel(cid=None, smiles=None):
    if cid and smiles:
        return None
    if not cid and not smiles:
        return None
    atoms = load_atoms(cid=cid)
    pointcloud = load(atoms)
    data_loader = DataLoader(pointcloud)
    full_mat = data_loader(mag_coeff=100, sigma=1)
    return full_mat


def main():
    cid = 31525001
    atoms = load_atoms(cid=cid)
    # smiles = "CNO"
    # atoms=load_atoms(smiles=smiles)
    pointcloud = load(atoms)
    data_loader = DataLoader(pointcloud)
    # full_mat=data_loader(mag_coeff=100,sigma=1)
    # visualization(full_mat)
    for mag_coeff, sigma in zip([50], [2]):
        full_mat = data_loader(mag_coeff=mag_coeff, sigma=sigma)
        visualization(full_mat)


if __name__ == '__main__':
    main()
