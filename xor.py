# xor.py
#
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5525_spring19.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

import torch
import torch.nn as nn
from torch import optim
import random
import numpy as np

#Define the computation graph; one layer hidden network
class FFNN(nn.Module):
    def __init__(self, dim_i, dim_h, dim_o):
        super(FFNN, self).__init__()
        self.V = nn.Linear(dim_i, dim_h)
        self.g = nn.Tanh()
        self.W = nn.Linear(dim_h, dim_o)
        self.softmax = nn.Softmax(dim=0)
       

        #Initalize weights (Glorot and Bengio 2010)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x):
        return self.softmax(self.W(self.g(self.V(x))))

if __name__ == "__main__":
    train_X = np.array([[0,0], [0,1], [1,0], [1,1]])
    train_Y = np.array([0,1,1,0])

    num_classes  = 2
    num_hidden   = 10
    num_features = 2

    ffnn = FFNN(num_features, num_hidden, num_classes)
    optimizer = optim.Adam(ffnn.parameters(), lr=0.1)

    for epoch in range(100):
        total_loss = 0.0
        #Randomly shuffle examples in each epoch
        shuffled_i = list(range(0,len(train_Y)))
        random.shuffle(shuffled_i)
        for i in shuffled_i:
            x        = torch.from_numpy(train_X[i]).float()
            y_onehot = torch.zeros(num_classes)
            y_onehot[train_Y[i]] = 1

            ffnn.zero_grad()
            probs = ffnn.forward(x)
            loss = torch.neg(torch.log(probs)).dot(y_onehot)
            total_loss += loss
            
            loss.backward()
            optimizer.step()
            
        print("loss on epoch %i: %f" % (epoch, total_loss))

    #Evaluate on the training set:
    num_errors = 0
    for i in range(len(train_Y)):
        x = torch.from_numpy(train_X[i]).float()
        y = train_Y[i]
        probs = ffnn.forward(x)
        prediction = torch.argmax(probs)
        if y != prediction:
            num_errors += 1
    print("number of errors: %d" % num_errors)
        
    
