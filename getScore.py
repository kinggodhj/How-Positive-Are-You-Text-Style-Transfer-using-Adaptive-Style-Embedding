import argparse
import fasttext
from functools import reduce
import math
import pdb
import operator

from sklearn.metrics import accuracy_score

model=fasttext.load_model('./train.sentiment.bin')

def geometric_mean(precisions):
    return (reduce(operator.mul, precisions))**(1.0/len(precisions))

def clip_count(cand_d, ref_ds):
    '''
    arguments:
        cand_d:{'I': 1, 'had': 1, 'a':1, 'dinner':1}
        ref_ds:[{'He':1, 'had': 1, 'a':1, 'dinner':1},
               {'He':1, 'had':1, 'a':1, 'lunch':1}[
    returns:

    '''
    count=0
    for key, value in cand_d.items():
        key_max=0
#         for ref in ref_ds:
#            if key in ref:
        if key in ref_ds:
            key_max=max(key_max, ref_ds[key])
#                key_max=max(key_max, ref[key])
        clipped_count=min(value, key_max)
        count+=clipped_count
    return count

def ngram_precision(candidate, references, n):
    def _count_ngram(sentence, n):
        ngram_d={}
        words=sentence.strip().split()
        leng=len(words)
        limits=leng-n+1
        
        for i in range(limits):
            ngram=' '.join(words[i:i+n]).lower()
            if ngram in ngram_d.keys():
                ngram_d[ngram]+=1
            else:
                ngram_d[ngram]=1
        return ngram_d
    
    clipped_count=0
    count=0
    for si in range(len(candidate)):
#        ref_counts=[]
#        for reference in references:
#            ngram_d=_count_ngram(reference, n)
#            ref_counts.append(ngram_d)
        ref_counts=_count_ngram(references[si], n)                
        cand_dict=_count_ngram(candidate[si], n)
        n_grams=0
        for key, values in cand_dict.items():
            n_grams+=values

        clipped_count+=clip_count(cand_dict, ref_counts)
        count+=n_grams
    if clipped_count==0:
        pr=0
    else:
        pr=float(clipped_count) / count
    return pr

def brevity_penalty(c, r):
    if c>r:
        bp=1
    else:
        bp=math.exp(1-(float(r)/c))
    return bp

def best_length_match(ref_lens, cand_len):
    least_diff=abs(cand_len-ref_lens[0])
    best=ref_lens[0]
    for ref_len in ref_lens:
        if abs(cand_len-ref_len) < least_diff:
            least_diff=abs(cand_l-ref_len)
            best=ref_len
    return best

def calculate_bp(candidate, references):
    r, c=0, 0
#    bp=list()    
    for si in range(len(candidate)):
#        ref_lengths=list()
        len_ref=len(references[si].strip().split())        
#        for reference in references:
#            ref_length=len(reference[si].strip().split())
#            ref_lengths.append(ref_length)
        len_candidate=len(candidate[si].strip().split())
#        r+=best_length_match(ref_length, len_candidate)
#         r+=best_length_match(ref_lengths, len_candidate)
        r+=len_ref
        c+=len_candidate
    bp=brevity_penalty(c, r)
    return bp
        
def getBleu(candidate, references):
    precisions=list()
    for i in range(4):
        pr=ngram_precision(candidate, references, i+1)
        precisions.append(pr)
    bp=calculate_bp(candidate, references)
    bleu=geometric_mean(precisions)*bp
    return bleu, bp, precisions

def getAccuracy(generated_sentence, labels, positive=None, negative=None, pred=None):
    if positive is None:
        positive=list()
        negative=list()
        pred=list()

#    #one sentence
#    if type(generated_sentence)==str:
#        result=model.predict(generated_sentence, k=1, threshold=0.5)
#        if '0' in result[0][0]:
#            pred.append(0)
#        elif '1' in result[0][0]:
#            pred.append(1)
#        return accuracy_score(true, pred)
#    
    for i in range(len(generated_sentence)):
        result=model.predict(generated_sentence[i], k=1, threshold=0.5)
        if '0' in result[0][0]:
            negative.append(generated_sentence[i])
            pred.append(0)
        elif '1' in result[0][0]:
            positive.append(generated_sentence[i])
            pred.append(1)
        else:
            print('classification error')
            pdb.set_trace()
    return accuracy_score(labels, pred)
