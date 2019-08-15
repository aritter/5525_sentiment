# imdb.py
# --------------
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5525_spring19.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

import os
import sys

import torch

#Sparse matrix implementation
from scipy.sparse import csr_matrix
from Vocab import Vocab
import numpy as np
from collections import Counter

np.random.seed(1)

class IMDBdata:
    def __init__(self, directory, vocab=None):
        """ Reads in data into sparse matrix format """
        pFiles = os.listdir("%s/pos" % directory)
        nFiles = os.listdir("%s/neg" % directory)

        if not vocab:
            self.vocab = Vocab()
        else:
            self.vocab = vocab

        #For csr_matrix (see http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)
        X_values = []
        X_row_indices = []
        X_col_indices = []
        Y = []

        XwordList = []

        #Read positive files
        for i in range(len(pFiles)):
            f = pFiles[i]
            for line in open("%s/pos/%s" % (directory, f)):
                wordList   = [self.vocab.GetID(w.lower()) for w in line.split(" ") if self.vocab.GetID(w.lower()) >= 0]
                XwordList.append(wordList)
                wordCounts = Counter(wordList)
                for (wordId, count) in wordCounts.items():
                    if wordId >= 0:
                        X_row_indices.append(i)
                        X_col_indices.append(wordId)
                        X_values.append(count)
            Y.append(+1.0)

        #Read negative files
        for i in range(len(nFiles)):
            f = nFiles[i]
            for line in open("%s/neg/%s" % (directory, f)):
                wordList   = [self.vocab.GetID(w.lower()) for w in line.split(" ") if self.vocab.GetID(w.lower()) >= 0]
                XwordList.append(wordList)
                wordCounts = Counter(wordList)
                for (wordId, count) in wordCounts.items():
                    if wordId >= 0:
                        X_row_indices.append(len(pFiles)+i)
                        X_col_indices.append(wordId)
                        X_values.append(count)
            Y.append(-1.0)
            
        self.vocab.Lock()

        #Create a sparse matrix in csr format
        self.X = csr_matrix((X_values, (X_row_indices, X_col_indices)), shape=(max(X_row_indices)+1, self.vocab.GetVocabSize()))
        self.Y = np.asarray(Y)

        #Randomly shuffle
        index = np.arange(self.X.shape[0])
        np.random.shuffle(index)
        self.X = self.X[index,:]
        self.XwordList = [torch.LongTensor(XwordList[i]) for i in index]  #Two different sparse formats, csr and lists of IDs (XwordList).
        self.Y = self.Y[index]

if __name__ == "__main__":
    data = IMDBdata("../../data/aclImdb/train/")
    print(data.X)
