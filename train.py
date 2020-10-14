# -*- encoding: utf-8 -*-
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.utils.data as data
import time
from voxelnet import VoxelNet
from loss import VoxelLoss
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import torch.backends.cudnn
from pointcloud2voxel import load_voxel

import cv2


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


def train():
    net = VoxelNet()
    # net.cuda()

    net.train()

    # initialization
    print('Initializing weights...')
    net.apply(weights_init)
    return net
    # print(net)
    # define optimizer
    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    # define loss function
    # criterion = VoxelLoss(alpha=1.5, beta=1)


def main():
    net = train()
    cid = 31525001
    voxel = load_voxel(cid=cid)
    print(voxel.shape)
    psm, rm = net.forward(torch.from_numpy(voxel).float())
    print(psm, rm)


if __name__ == '__main__':
    main()

    # training process
    # batch_iterator = None
    # epoch_size = len(dataset) // cfg.N
    # print('Epoch size', epoch_size)
    # for iteration in range(10000):
    #         if (not batch_iterator) or (iteration % epoch_size == 0):
    #             # create batch iterator
    #             batch_iterator = iter(data_loader)
    #
    #         voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets, images, calibs, ids = next(batch_iterator)
    #
    #         # wrapper to variable
    #         voxel_features = Variable(torch.cuda.FloatTensor(voxel_features))
    #         pos_equal_one = Variable(torch.cuda.FloatTensor(pos_equal_one))
    #         neg_equal_one = Variable(torch.cuda.FloatTensor(neg_equal_one))
    #         targets = Variable(torch.cuda.FloatTensor(targets))
    #
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #
    #         # forward
    #         t0 = time.time()
    #         psm,rm = net(voxel_features, voxel_coords)
    #
    #         # calculate loss
    #         conf_loss, reg_loss = criterion(rm, psm, pos_equal_one, neg_equal_one, targets)
    #         loss = conf_loss + reg_loss
    #
    #         # backward
    #         loss.backward()
    #         optimizer.step()
    #
    #         t1 = time.time()
    #
    #
    #         print('Timer: %.4f sec.' % (t1 - t0))
    #         print('iter ' + repr(iteration) + ' || Loss: %.4f || Conf Loss: %.4f || Loc Loss: %.4f' % \
    #               (loss.data[0], conf_loss.data[0], reg_loss.data[0]))
    #
    #         # visualization
    #         #draw_boxes(rm, psm, ids, images, calibs, 'pred')
    #         draw_boxes(targets.data, pos_equal_one.data, images, calibs, ids,'true')
