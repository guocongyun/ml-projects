#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn

class HighWay(nn.Module):
    """ HighWay Layer, i.e. a layer of  highway network 
    that takes the output of convolutional network as input
    """
    def __init__(self, embedded_word_size):
        """ Init HighWay Instance.
        @param embedded_word_size: int
        """
        super(HighWay, self).__init__()
        self.embedded_word_size = embedded_word_size
        self.proj_projection = nn.Linear(in_features=embedded_word_size,out_features=embedded_word_size)
        self.gate_projection = nn.Linear(in_features=embedded_word_size,out_features=embedded_word_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_conv_out):
        """ Run a forward step that map a batch of x_conv_out to x_high_way
        @param x_conv_out: tensor of (batch_size, embedded_word_size)
        @return x_highway: tensor of (batch_size, embedded_word_size)
        """
        assert x_conv_out.size()[1] == self.embedded_word_size, print(f'{x_conv_out.size()} conv_out size')

        x_proj = self.relu(self.proj_projection(x_conv_out)) # IMPORTANT torch.size() == (2,3)
        assert x_proj.size() == x_conv_out.size(), print(f'{x_proj.size()} x_proj size') # size should be number of embedded words x batch

        x_gate = self.sigmoid(self.gate_projection(x_conv_out))
        assert x_gate.size() == x_conv_out.size(), print(f'{x_gate.size()} x_gate size')

        x_highway = x_gate * x_proj + (1-x_gate) * x_conv_out
        assert x_highway.size() == x_conv_out.size(), print(f'{x_highway.size()} x_highway size')

        return x_highway

### END YOUR CODE 

if __name__ == '__main__':
    conv_out = torch.tensor([[0,0,0],[0,0,0]], dtype=torch.float)
    highway = HighWay(embedded_word_size = 3)
    x_highway = highway.forward(conv_out)
    print(x_highway)
    # highway.forward()@
