#!usr/bin/env python
#-*- coding: utf8 -*-

#   BibleNet
#
#   Author: GETALP TEAM
#   Last Modified: 27/03/2019
#
#   Université Grenoble Alpes

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def compute_padding(kernel_size, padding_type):
    """
    Computes padding value
    :param kernel_size: kernel's width
    :param padding_type: type of padding wanted by the user
    :return:
    """
    if padding_type == 'same':
        pad_value = math.floor(kernel_size / 2)
        return pad_value, kernel_size % 2 == 0
    else:
        raise NotImplementedError


class Attention(nn.Module):
    def __init__(self, input_dimension, attn_dimension, activation=torch.tanh):
        super(Attention, self).__init__()

        self.linear1 = nn.Linear(input_dimension, attn_dimension)
        self.linear2 = nn.Linear(attn_dimension, 1)
        self.activation = activation

    def forward(self, inp):
        x = self.linear1(inp)
        x = self.activation(x)
        x = self.linear2(x)
        alphas = F.softmax(x, dim=1)

        weighted = torch.sum(alphas * inp, dim=1)
        return weighted, alphas


class Identity(nn.Module):
    """
    Identity layer (returns its input)
    """

    def __init__(self):
        """
        No arguments
        """
        super(Identity, self).__init__()

    def forward(self, x):
        """
        :param x: input tensor of shape whatever you want ;)
        :return: input tensor of shape whatever was fed in
        """
        return x


class Conv1D_BatchNorm(nn.Module):
    """
    1D Convolution with BatchNorm
    """

    def __init__(self, input_dim, output_dim, kernel_size, stride=1, padding='same', activation=F.relu):
        """
        :param input_dim: (int) input dimension
        :param kernel_size: (int) kernel size/width
        :param activation: (callable) activation function (from functional module (F) such as F.relu)
        :param padding: (str) padding which be used (as for now, only same is supported)
        """
        super(Conv1D_BatchNorm, self).__init__()

        padding, self.remove_timestep = compute_padding(kernel_size, padding)
        self.convolution = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding,
                                     bias=False)
        self.activation = activation
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
        :param x: input tensor of shape [batch, channels, sequence]
        :return: convolved and normed tensor of shape [batch, channel, sequence].
                 input sequence should have the same length as output sequence
                 because stride=1 with "same" padding
        """
        input_size = x.size(-1)
        x = self.convolution(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        if self.remove_timestep:
            # padding is not exactly same, sometimes there is one more timestep
            return x[:, :, :input_size]
        assert input_size == x.size(-1)
        return x


class Conv1D_Bank(nn.Module):
    """
    1D convolution bank
    """

    def __init__(self, input_dim=128, output_dim=128, conv_K=8, activation=F.relu, padding='same'):
        """
        :param input_dim: (int) number of input dimension
        :param conv_K: (int) number of convolution to perform
        :param activation: (callable) activation function (from functional such as F.relu)
        :param padding: (str) padding which be used (as for now, only same is supported)
        """
        super(Conv1D_Bank, self).__init__()

        self.activation = activation
        self.padding = padding

        self.convolutions = nn.ModuleList()
        for nth_convolution in range(conv_K):
            convolution = Conv1D_BatchNorm(input_dim, input_dim, kernel_size=nth_convolution+1, stride=1,
                                           padding=padding, activation=activation)
            self.convolutions.append(convolution)

    def forward(self, x):
        """
        :param x: input tensor of shape [batch, channel, sequence]
        :return: tensor convolved K times and concatenated on filter axis. Shape [batch, K*channel, sequence]
        """

        out_convolution = []
        x = x.permute(0,2,1)
        for convolve in self.convolutions:
            convolved = convolve(x)
            out_convolution.append(convolved)
        x = torch.cat(out_convolution, dim=1)
        x = x.permute(0,2,1)
        return x


class BibleNet(nn.Module):
    def __init__(self, input_dim=40, output_dim=256, recurrent_units=256, recurrent_layers=2, attn_dimension=256, conv_K=16, bidirectional=True):
        super(BibleNet, self).__init__()

        attn_input_dimension = recurrent_units * 2 if bidirectional else recurrent_units
        self.bank = Conv1D_Bank(input_dim, input_dim, conv_K=16)
        self.project = nn.Linear(input_dim*conv_K, output_dim)
        self.recurrence = nn.LSTM(input_size=output_dim, hidden_size=recurrent_units, num_layers=recurrent_layers,
                                  bidirectional=bidirectional, batch_first=True)
        self.attend = Attention(input_dimension=attn_input_dimension, attn_dimension=attn_dimension)

    def forward(self, x):
        x = self.bank(x)
        x = self.project(x)
        x, (h, c) = self.recurrence(x)
        weighted, alphas = self.attend(x)
        return weighted, alphas
