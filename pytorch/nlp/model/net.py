"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax. Be careful to ensure your dimensions are correct after each step.

    You are encouraged to have a look at the network in pytorch/vision/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available to you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params, is_phrase):
        """
        We define an recurrent network that predicts the NER tags for each token in the sentence. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains vocab_size, embedding_dim, lstm_hidden_dim
        """
        super(Net, self).__init__()

        # the embedding takes as input the vocab_size and the embedding_dim

        # the LSTM takes as input the size of its input (embedding_dim), its hidden size
        # for more details on how to use it, check out the documentation
        if is_phrase:
            self.lstm = nn.LSTM(params.word_embedding_dim, params.word_hidden_dim, batch_first=True)
        else:
            self.lstm = nn.LSTM(params.vid_embedding_dim, params.vid_hidden_dim, batch_first=True)

        # the fully connected layer transforms the output to give the final output layer        
    def forward(self, s, anchor_is_phrase = False):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) is a tensor containing triplet of embeddings (batch size x mess (3 sequences of embeddings))
            s: (Variable) tensor with batch_size x max_sequence_len x embedding_dim

        Returns:
            out: (Variable) dimension batch_size*seq_len x num_tags with the log probabilities of tokens for each token
                 of each sentence.

        Note: the dimensions after each step are provided
        """
        #                                -> batch_size x seq_len
        # dim: batch_size x seq_len x embedding_dim
        # run the LSTM along the sentences of length seq_len
        s, _ = self.lstm(s)
        s.data.contiguous()
        return s
