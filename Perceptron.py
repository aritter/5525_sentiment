# Perceptron.py
# --------------
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5525_spring19.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

import sys

import numpy as np
from Eval import Eval

from imdb import IMDBdata

class Perceptron:
    def __init__(self, X, Y, N_ITERATIONS):
        self.N_ITERATIONS = N_ITERATIONS
        #TODO: Initalize parameters
        self.Train(X,Y)

    def ComputeAverageParameters(self):
        #TODO: Compute average parameters (do this part last)
        return

    def Train(self, X, Y):
        #TODO: Estimate perceptron parameters
        return

    def Predict(self, X):
        #TODO: Implement perceptron classification
        return 1

    def Eval(self, X_test, Y_test):
        Y_pred = self.Predict(X_test)
        ev = Eval(Y_pred, Y_test)
        return ev.Accuracy()

if __name__ == "__main__":
    train = IMDBdata("%s/train" % sys.argv[1])
    test  = IMDBdata("%s/dev" % sys.argv[1], vocab=train.vocab)
    
    ptron = Perceptron(train.X, train.Y, int(sys.argv[2]))
    ptron.ComputeAverageParameters()
    print(ptron.Eval(test.X, test.Y))

    #TODO: Print out the 20 most positive and 20 most negative words
