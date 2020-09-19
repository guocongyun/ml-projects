#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn

class CNN(nn.Module):
    """ CNN Layer, i.e. a layer of  cnn network 
    that takes the output of convolutional network as input
    """
    def __init__(self, input_channel_count, output_channel_count, kernel_size=5):
        """ Init HighWay Instance.
        @param input_channel_count: int 
        @param output_channel_count: int
        @param kernel_size: int
        """
        super(CNN, self).__init__()
        self.input_channel_count = input_channel_count
        self.output_channel_count = output_channel_count
        self.conv = nn.Conv1d(in_channels=input_channel_count, 
                              out_channels=output_channel_count, 
                              kernel_size=kernel_size)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.relu = nn.ReLU()
 
    def forward(self, x_reshaped):
        """ Run a forward step that map a batch of x_reshaped to x_conv_out
        @param x_reshaped: tensor of (max_sentence_length * batch_size, e_char, max_word_length)
        @return x_conv_out: tensor of (max_sentence_length * batch_size, e_word)
        """
        assert x_reshaped.size()[1] == self.input_channel_count, print(f'{x_reshaped.size()} x_reshaped size')

        x_conv = self.conv(x_reshaped)
        assert x_conv.size()[1] == self.output_channel_count, print(f'{x_conv.size()} x_conv size')

        # x_conv_out = x_conv.max(dim=1) == np.max(dim1)

        x_conv_out = self.max_pool(x_conv).squeeze(dim=2)
        assert x_conv_out.size()[1] == self.output_channel_count, print(f'{x_conv_out.size()} x_conv_out size')

        return x_conv_out

### END YOUR CODE 

if __name__ == '__main__':
    x_reshaped = torch.tensor([[[0,0,0],[0,0,0]],[[0,0,0],[0,0,0]]], dtype=torch.float)
    cnn = CNN(2, 2, 1)
    x_highway = cnn.forward(x_reshaped)
    print(x_highway)
    # highway.forward()@


### END YOUR CODE

