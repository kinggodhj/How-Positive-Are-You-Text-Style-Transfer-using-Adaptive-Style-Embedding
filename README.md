# How Positive Are You: Text Style Transfer using Adaptive Style Embedding 
How Positive Are You: Text Style Transfer using Adaptive Style Embedding(COLING 2020)

<https://www.aclweb.org/anthology/2020.coling-main.191.pdf>

## Dependencies 
python 3.7.3

torch 1.4.0

numpy 1.18.1

--------------------------------

### 1. Preparing Data
In this paper, "Yelp" and "Amazon" dataset are used.

Run the processing file in each dataset folder.

`python processed_data.py`
    
    
### 2. Training    
Run below code for training from scartch

`python main.py`


### 3. Evaluation
Accuracy, BLEU score, Perplexity are used to evaluate.

For calcualte perpexity, download ["SRILM"](http://www.speech.sri.com/projects/srilm/download)

After that modify the path in the "eval.sh" file.

`HUMAN_FILE=~/path of human_yelp file`

`SOURCE_FILE=~/path of sentiment.test file`

`PPL_PATH=~/path of srilm `

`ACC_PATH=~/path of acc.py file`


For yelp, run the below code. 

`./eval.sh`
