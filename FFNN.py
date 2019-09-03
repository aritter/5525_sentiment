# FFNN.py
#
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5525_fall19.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

import sys

import numpy as np
from Eval import Eval

import torch
import torch.nn as nn
import torch.optim as optim

from imdb import IMDBdata

class FFNN(nn.Module):
    def __init__(self, X, Y, VOCAB_SIZE, DIM_EMB=10, NUM_CLASSES=2):
        super(FFNN, self).__init__()
        (self.VOCAB_SIZE, self.DIM_EMB, self.NUM_CLASSES) = (VOCAB_SIZE, DIM_EMB, NUM_CLASSES)
        #TODO: Initialize parameters.

    def forward(self, X, train=False):
        #TODO: Implement forward computation.
        return torch.randn(self.NUM_CLASSES)


def Eval_FFNN(X, Y, mlp):
    num_correct = 0
    for i in range(len(X)):
        logProbs = mlp.forward(X[i], train=False)
        pred = torch.argmax(logProbs)
        if pred == Y[i]:
            num_correct += 1
    print("Accuracy: %s" % (float(num_correct) / float(len(X))))

def Train_FFNN(X, Y, vocab_size, n_iter):
    print("Start Training!")
    mlp = FFNN(X, Y, vocab_size)
    #TODO: initialize optimizer.

    for epoch in range(n_iter):
        total_loss = 0.0
        for i in range(len(X)):
            pass
            #TODO: compute gradients, do parameter update, compute loss.
        print(f"loss on epoch {epoch} = {total_loss}")
    return mlp

if __name__ == "__main__":
    train = IMDBdata("%s/train" % sys.argv[1])
    train.vocab.Lock()
    test  = IMDBdata("%s/dev" % sys.argv[1], vocab=train.vocab)
    
    mlp = Train_FFNN(train.XwordList, (train.Y + 1.0) / 2.0, train.vocab.GetVocabSize(), int(sys.argv[2]))
    Eval_FFNN(test.XwordList, (test.Y + 1.0) / 2.0, mlp)
