import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    # CNN
    def __init__(self):
        '''
        ######################################
        define your nerual network layers here
        ######################################
        '''

    def forward(self, x):
        '''
        ###############################################################
        do the forward propagation here
        x: model input with shape:(batch_size, frame_num, feature_size)
        frame_num is how many frame one wav have
        feature_size is the dimension of the feature
        ###############################################################
        '''

class FcNet(nn.Module):
    # DNN
    def __init__(self):
        '''
        ######################################
        define your nerual network layers here
        ######################################
        '''

    def forward(self, x):
        '''
        ###############################################################
        do the forward propagation here
        x: model input with shape:(batch_size, frame_num, feature_size)
        frame_num is how many frame one wav have
        feature_size is the dimension of the feature
        ###############################################################
        '''

