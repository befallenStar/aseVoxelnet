# -*- encoding: utf-8 -*-
import time
from itertools import permutations

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable

from loader.data_loading import load_db
from pointcloud2voxel import load_voxel
from voxelnet import SVFE


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


def main():
    try:
        feature_length = 128
        # load atoms of ase format from ase.db
        atomses = load_db(path='ase_data', ase_db='ase.db')

        # create a VoxelNet object
        # feature_length should be confirmed first, equals the length of the result
        net = SVFE(feature_length)
        net.train()
        # initialization
        print('Initializing weights...')
        net.apply(weights_init)

        # use the basic loss function

        # criterion=nn.L1Loss()
        # criterion=nn.SmoothL1Loss()
        criterion = nn.MSELoss()

        # define optimizer
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        # define loss function
        losses = []

        # record the time cost
        t0 = time.time()
        for atoms in atomses:
            for rotate in permutations([0,90,180,270],3):
                # iterate the atoms
                voxel = load_voxel(atoms, mag_coeff=30, sigma=1,rotate=rotate)
                # print("voxel: " + str(voxel.shape))
                # wrapper to variable
                voxel_features = Variable(torch.FloatTensor(voxel))

                # use initial charges as a kind of feature to feed in loss functions
                charges = list(atoms.get_initial_charges())
                charges.extend([0 for _ in range(feature_length - len(charges))])
                charges = torch.Tensor(charges)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                vwfs = net(voxel_features)

                # calculate loss
                loss = criterion(vwfs, charges)

                # backward
                loss.backward()
                optimizer.step()
                print('Loss: %.4f' % (loss.data))
                losses.append(loss.data)

                # predicted feature
                # print(vwfs)

        print(losses)
        t1 = time.time()
        print('Timer: %.4f sec.' % (t1 - t0))

        # plot the loss curve
        x = [i for i in range(len(atomses))]
        plt.plot(x, losses, color='r', linestyle='-')
        plt.show()

        # save the model
        # torch.save(net,'model/aseVoxelNet-1.pkl')
    except ValueError as e:
        print(str(e))


if __name__ == '__main__':
    main()
