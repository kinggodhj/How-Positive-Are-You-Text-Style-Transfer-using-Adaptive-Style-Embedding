# How Positive Are You: Text Style Transfer using Adaptive Style Embedding 
How Positive Are You: Text Style Transfer using Adaptive Style Embedding(COLING 2020)

<https://www.aclweb.org/anthology/2020.coling-main.191.pdf>


--------------------------------


Model description    
Whole model architecture consists of two module

1) Auto-encoder based on Transformer

      Transformer is trained by reconstruction loss (Cross-entropy)

      Especially, decoder's input is (encoder's latent vector + style embedding). 

      Possible combination: (positive sentence-positive embedding, negative sentence-negative embedding)


2) Style embedding module

      Style embedding module is trained by classification loss

      Style embedding vector is trained by similarities with encoder's latent vector

--------------------------------

## Dependencies 
python 3.7.3

torch 1.4.0

numpy 1.18.1

nltk

--------------------------------

### 1. Preparing Data
In this paper, "Yelp" and "Amazon" dataset are used.

nltk is used for tokenization.

Run the processing file in each dataset folder.

`python processed_data.py`

--------------------------------

### 2. Training    
Run below code for training from scartch

`python main.py`

#### Default settings)
```
Transformer # layers: 2-layers 

Transforemr embedding, latent, model size: 256

Style embedding size: 256
```

--------------------------------

### 3. Style transfer

--------------------------------

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

--------------------------------

### Other settings

To improve the PPL scores, I've tried a lot of options.

Below two options are meaningful

1) Using the pre-trained word embeddings from GPT-2

   Fluency is reflected by PPL score.
   
   The low PPL score means low fluency performance. 
   
   Therefore, to inhance the fluency, use the pre-trained word embeddings.
   
   i've used ![image](https://user-images.githubusercontent.com/37800546/114685418-cac64980-9d4c-11eb-8894-47c26b929980.png) [huggingface](https://huggingface.co/transformers/)
   
   GPT2Model setting is added in <U>model.py</U> file
   ```
   from transformers import GPT2Model
   
   gpt = GPT2Model.from_pretrained('gpt2')
   pre = gpt.get_input_embeddings()
   share_embedding = Embeddings(d_model, d_vocab)
    
   share_embedding.weight = nn.Parameter(pre.weight)
   ```
   If pre-trained word embeddings is used, training is done after 1 or 2 epochs (less than 5 epochs).
   
   Also, PPL score is emproved.
   
2) Pre-Layer normalization & Layer normalization to style embedding module 
   
   [Pre-layer normalization](https://openreview.net/forum?id=B1x8anVFPr) boosts the speed of training
   
   Encoder & Decoder class is modified in <U>model.py</U> file
   ```
   def forward(self, x, mask):
      for layer in self.layers:
         x=self.norm(x)
         x=layer(x, memory, src_mask, tgt_mask)
      return self.norm(x) 
   ```
   
   Style embedding using layer normalization 
   
   StyleEmbeddings class is modified in <U>model.py</U> file
   ```
   def __init__(self, n_style, d_style):
       super(StyleEmbeddings, self).__init__()
       self.lut=nn.Embedding(n_style, d_style)
       self.norm=LayerNorm(d_style)
  
   def forward(self, x):
       return self.norm(self.lut(x))
   ```
