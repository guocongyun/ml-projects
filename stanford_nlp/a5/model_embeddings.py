#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
from highway import Highway
from cnn import CNN
# IMPORTANT import highway imports the module instead of the class

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

# from cnn import CNN
# from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        pad_token_idx = vocab.char2id['<pad>']
        self.embed_size = embed_size
        self.e_char = 50  # e_char = 50 (see 1. (j) bullet point 5)
        self.e_word = embed_size  # e_word

        self.char_embed = nn.Embedding(num_embeddings=len(vocab.char2id),
                                       embedding_dim=self.e_char,
                                       padding_idx=pad_token_idx)
        self.vocab = vocab
        self.highway = Highway(embed_size)
        self.cnn = CNN(input_channel_count = self.e_char, 
                       output_channel_count = self.e_word)
        self.dropout = nn.Dropout(p=0.3)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        # TODO:
        # IMPORTANT e_char, e_word == input, output channels
        # IMPORTANT max_word_len, max_sentence_len = input, output layers

        # 0) size input shape
        (sentence_len, batch_size, max_word_len) = input.size()
        # 1) look up embedding to get x_emb
        x_emb = self.char_embed(input.long())
        assert x_emb.size() == (sentence_len, batch_size, max_word_len, self.e_char), print(x_emb.size())

        # 2) reshape the x_emb to x_reshape
        x_reshape = x_emb.view(sentence_len * batch_size, max_word_len, self.e_char).permute(0,2,1)
        assert x_reshape.size() == (sentence_len * batch_size, self.e_char, max_word_len), print(x_reshape.size()); 

        # 3) run through the convolution network to get x_conv_out
        x_conv_out = self.cnn.forward(x_reshape)
        assert x_conv_out.size() == (sentence_len * batch_size, self.e_word), print(x_conv_out.size())

        # 4) run through the highway layer to get the x_highway
        x_highway = self.highway.forward(x_conv_out)
        assert x_highway.size() == (sentence_len * batch_size, self.e_word), print(x_highway.size())

        # 5) reshape x_highway into (sentence_len, batch_size, self.e_char)
        x_highway = x_highway.view(sentence_len, batch_size, self.e_word)
        assert x_highway.size() == (sentence_len, batch_size, self.e_word), print(x_highway.size())

        # 5) drop out x_highway to get x_word_emb
        x_word_emb = self.dropout(x_highway)
        assert x_word_emb.size() == (sentence_len, batch_size, self.e_word), print(x_word_emb.size())

        return x_word_emb



        ### END YOUR CODE

