# -*- encoding: utf-8 -*-
from ase import Atoms
from ase.data.pubchem import pubchem_conformer_search

from loader.pointclouds_to_voxelgrid import *
from loader.visualize_npy import visualization

colors = {1: (1, 1, 1), 6: (.5, .5, .5), 7: (0, 0, 1),
          8: (1, 1, 0), 15: (1, 0, 1), 16: (1, 0, 0), 17: (0, 1, 0)}
atoms_vector = [1, 6, 7, 8, 15, 16, 17]


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


def load_voxel(atoms, mag_coeff=20, sigma=1, threshold=0.7):
    pointcloud = load(atoms)
    data_loader = DataLoader(pointcloud)
    full_mat = data_loader(mag_coeff=mag_coeff, sigma=sigma)
    D, H, W, C = full_mat.shape
    D_new = ((D >> 3) + 1) << 3
    H_new = ((H >> 3) + 1) << 3
    W_new = ((W >> 3) + 1) << 3
    voxel = np.zeros([D_new, H_new, W_new, C])
    voxel[(D_new - D) >> 1:((D_new - D) >> 1) + D,
    (H_new - H) >> 1:((H_new - H) >> 1) + H,
    (W_new - W) >> 1:((W_new - W) >> 1) + W, :] = full_mat
    # psm: [(D_new>>2) - 1, 2, H_new>>1, W_new>>1]
    # rm: [(D_new>>2) - 1, 14, H_new>>1, W_new>>1]
    pos_equal_true = voxel >= threshold
    targets_true = voxel == 1
    pos_equal_one = np.zeros([(D_new >> 2) - 1, H_new >> 1, W_new >> 1, 2])
    neg_equal_one = np.zeros([(D_new >> 2) - 1, H_new >> 1, W_new >> 1, 2])
    targets = np.zeros([(D_new >> 2) - 1, H_new >> 1, W_new >> 1, 14])
    for i in range(0, D_new - 4, 4):
        for j in range(0, H_new, 2):
            for k in range(0, W_new, 2):
                for c in range(2):
                    if pos_equal_true[i:i + 4, j:j + 2, k:k + 2,
                       c:c + C // 2].sum():
                        pos_equal_one[i // 4, j // 2, k // 2, c] = 1
                for f in range(0, 14, 7):
                    if targets_true[i:i + 4, j:j + 2, k:k + 2,
                       f:f + 7].sum():
                        targets[i // 4, j // 2, k // 2, f] = 1
    neg_equal_one[pos_equal_one == 0] = 1
    # print(pos_equal_one)
    # print(neg_equal_one)
    # print(targets)
    return voxel, pos_equal_one, neg_equal_one, targets


def main():
    cid = 31525001
    atoms = load_atoms(cid=cid)
    # smiles = "CNO"
    # atoms=load_atoms(smiles=smiles)
    pointcloud = load(atoms)
    data_loader = DataLoader(pointcloud)
    # full_mat=data_loader(mag_coeff=100,sigma=1)
    # visualization(full_mat)
    for mag_coeff, sigma in zip([20], [2]):
        full_mat = data_loader(mag_coeff=mag_coeff, sigma=sigma)
        visualization(full_mat)


if __name__ == '__main__':
    main()
