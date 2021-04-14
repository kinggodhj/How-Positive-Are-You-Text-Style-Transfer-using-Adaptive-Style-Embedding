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

nltk is used for tokenization.

Run the processing file in each dataset folder.

`python processed_data.py`

### 2. Model description    
Whole model architecture consists of two module

1) Auto-encoder based on Transformer

Transformer is trained by reconstruction loss (Cross-entropy)

Especially, decoder's input is (encoder's latent vector + style embedding). 

Possible combination: (positive sentence-positive embedding, negative sentence-negative embedding)

Positive style embedding is added to input positive sentence.

Also, negative style embedding is added to input negative sentence.


2) Style embedding module

Style embedding module is trained by classification loss

Style embedding vector is trained by similarities with encoder's latent vector

Positive sentences are compressed to latent vector that is containing positive information.

So style embedding can be trained to represent each style information using similarities. 


settings)

Transformer # layers: 2-layers 

Transforemr embedding, latent, model size: 256

Style embedding size: 256

### 3. Training    
Run below code for training from scartch

`python main.py`


### 4. Evaluation
Accuracy, BLEU score, Perplexity are used to evaluate.

For calcualte perpexity, download ["SRILM"](http://www.speech.sri.com/projects/srilm/download)

After that modify the path in the "eval.sh" file.

`HUMAN_FILE=~/path of human_yelp file`

`SOURCE_FILE=~/path of sentiment.test file`

`PPL_PATH=~/path of srilm `

`ACC_PATH=~/path of acc.py file`


For yelp, run the below code. 

`./eval.sh`


### 5. Style transfer

