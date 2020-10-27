# -*- encoding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable

from loss import VoxelLoss
from pointcloud2voxel import load_atoms
from pointcloud2voxel import load_voxel

np.set_printoptions(suppress=True)


def main():
    net_path = 'model/aseVoxelNet.pkl'
    net = torch.load(net_path)

    atoms = load_atoms(cid=101)
    voxel, pos_equal_one, neg_equal_one, targets = load_voxel(atoms,
                                                              mag_coeff=30,
                                                              sigma=1,
                                                              threshold=0.7)
    voxel_features = Variable(torch.FloatTensor(voxel))
    pos_equal_one = Variable(torch.FloatTensor(pos_equal_one))
    neg_equal_one = Variable(torch.FloatTensor(neg_equal_one))
    targets = Variable(torch.FloatTensor(targets))

    psm, rm = net(voxel_features)
    # D,_,H,W=psm.shape
    # psm=psm.detach().reshape([D,H,W,-1])
    # print(psm.detach().numpy())
    # print('-'*50)
    # rm=rm.detach().reshape([D,H,W,-1])
    # print(rm.detach().numpy())

    criterion = VoxelLoss(alpha=1.5, beta=1)
    conf_loss, loc_loss = criterion(rm, psm, pos_equal_one,
                                    neg_equal_one, targets)
    loss = conf_loss + loc_loss
    print(conf_loss)
    print(loc_loss)
    print(loss)


if __name__ == '__main__':
    main()
