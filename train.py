# -*- encoding: utf-8 -*-
import time

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable

from loader.data_loading import load_db
from loss import VoxelLoss
from pointcloud2voxel import load_voxel
from voxelnet import VoxelNet


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


def main():
    try:
        net = VoxelNet()

        net.train()

        # initialization
        print('Initializing weights...')
        net.apply(weights_init)
        # cid = 33
        # atoms = load_atoms(cid=cid)
        atomses = load_db(path='ase_data', ase_db='ase-100.db')
        # define optimizer
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        # define loss function
        criterion = VoxelLoss(alpha=1.5, beta=1)
        losses, conf_losses, loc_losses = [], [], []

        t0 = time.time()
        for atoms in atomses:
            voxel, pos_equal_one, neg_equal_one, targets = load_voxel(atoms,
                                                                      mag_coeff=30,
                                                                      sigma=1,
                                                                      threshold=0.7)
            # print("voxel: " + str(voxel.shape))
            # wrapper to variable
            voxel_features = Variable(torch.FloatTensor(voxel))
            pos_equal_one = Variable(torch.FloatTensor(pos_equal_one))
            neg_equal_one = Variable(torch.FloatTensor(neg_equal_one))
            targets = Variable(torch.FloatTensor(targets))
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            psm, rm = net(voxel_features)

            # calculate loss
            conf_loss, loc_loss = criterion(rm, psm, pos_equal_one,
                                            neg_equal_one, targets)
            loss = conf_loss + loc_loss

            # backward
            loss.backward()
            optimizer.step()
            print('Loss: %.4f || Conf Loss: %.4f || Loc Loss: %.4f' % (
                loss.data, conf_loss.data, loc_loss.data))
            losses.append(loss.data)
            conf_losses.append(conf_loss.data)
            loc_losses.append(loc_loss.data)

            # # print(voxel==1)
            # psm, rm = net.forward(torch.from_numpy(voxel).float())
            # # probability score map and regression map
            # print('probability score map: ' + str(psm.shape))
            # print('regression map: ' + str(rm.shape))

        # print(losses)
        # print(conf_losses)
        # print(loc_losses)
        t1 = time.time()
        print('Timer: %.4f sec.' % (t1 - t0))
        x = [i for i in range(len(atomses))]
        plt.plot(x, losses, color='r', linestyle='-')
        plt.plot(x, conf_losses, color='g', linestyle='--')
        plt.plot(x, loc_losses, color='b', linestyle='-.')
        plt.show()
        torch.save(net,'model/aseVoxelNet-1.pkl')
    except ValueError as e:
        print(str(e))


if __name__ == '__main__':
    main()
