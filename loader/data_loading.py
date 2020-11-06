# -*- encoding: utf-8 -*-
import os

from ase.db import connect

from pointcloud2voxel import load_atoms


def load_db(path='..\\data', ase_db='ase.db'):
    ase_path = os.path.join(path, ase_db)
    conn = connect(ase_path)
    atomses = []
    for row in conn.select():
        atoms = row.toatoms()
        atomses.append(atoms)
    return atomses


def merge(datapath='..\\data',destination='..\\ase_data', ase_db='ase.db'):
    atomses = []
    for file in os.listdir(datapath):
        # print(file)
        filepath = os.path.join(datapath, file)
        conn = connect(filepath)
        for row in conn.select():
            atoms = row.toatoms()
            atomses.append(atoms)

    for atoms in atomses:
        ase_path = os.path.join(destination, ase_db)
        conn = connect(ase_path)
        conn.write(atoms)
    print("{} atoms have been merged".format(len(atomses)))


def main():
    path = '..\\data'
    # cids = [3, 7, 11, 12, 13, 19, 21, 22, 29, 33, 34, 35, 44, 45, 49]
    for cid in range(1000):
        try:
            atoms = load_atoms(cid=cid + 1)
            # print('cid: ' + str(cid))
            name = atoms.symbols
            # print('atoms: ' + str(name))
            filepath = os.path.join(path, str(name) + '.db')
            print('filepath: ' + filepath)
            if not os.path.exists(filepath):
                conn = connect(filepath)
                conn.write(atoms)
            print(str(cid + 1) + " " + str(name) + ' done')
        except ValueError as e:
            print(cid + 1)
            print(str(e))


if __name__ == '__main__':
    # main()
    merge(ase_db='ase-1000.db')
    # atomses = load_db()
    # print(atomses)
