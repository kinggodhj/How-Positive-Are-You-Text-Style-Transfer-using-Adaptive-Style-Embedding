import argparse
import numpy as np
import pdb

from getAmazon import getAccuracy, getBleu

parser=argparse.ArgumentParser()
parser.add_argument('--path1', type=str, default='./save/generation/5_output.txt')
args=parser.parse_args()

def accTest(path):
    test=list()
    pred=list()
    positive=list()
    negative=list()
    labels=list()
#    f = open('args.file_path', 'r')
#    f = open('./yelp/sentiment.test.0', 'r')
#    f=open('./output.txt', 'r')
    f=open(path, 'r')
    lines = f.readlines()
    for line in lines:
        try:
            test.append(line.rstrip().split("'")[1])
        except Exception as e:
            test.append(line.rstrip())
    f.close()
    
    zero_labels=np.zeros(len(test)//2, dtype=int)
    one_labels=np.ones(len(test)//2, dtype=int)
    labels=np.concatenate((one_labels, zero_labels), 0)
#    pdb.set_trace()
    acc=getAccuracy(test, labels[0:len(test)], positive, negative, pred)

    print(acc)

if __name__=="__main__":
    accTest(args.path1)
