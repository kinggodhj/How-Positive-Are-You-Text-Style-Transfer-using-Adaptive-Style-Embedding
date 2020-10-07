import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import math, copy, time
import numpy as np
import pdb
import torch.nn.utils.rnn as rnn_utils

from beam import Beam
from data import get_cuda, to_var, calc_bleu


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention' """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    """Implement the PE function."""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, float(d_model), 2) *
                             -(math.log(10000.0) / float(d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class StyleEmbeddings(nn.Module):
    def __init__(self, n_style, d_style):
        super(StyleEmbeddings, self).__init__()
        self.lut = nn.Embedding(n_style, d_style)

    def forward(self, x):
        return self.lut(x)


################ Encoder ################
class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


################ Decoder ################
class Decoder(nn.Module):
    """Generic N layer decoder with masking."""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


################ Generator ################
class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, style_embed, generator, position_layer, model_size, latent_size, gpu):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.style_embed=style_embed
        #self.stlye_layer=nn.Linear()
        self.generator = generator
        self.position_layer = position_layer
        self.model_size = model_size
        self.latent_size = latent_size
        self.conv=nn.Conv1d(2, 1, 3, padding=1) #input channel, out channel, kernel size
        self.pool=nn.MaxPool1d(1)
        self.softmax=nn.Softmax(-1)
        self.gpu=gpu
        self.sigmoid = nn.Sigmoid()

        # self.memory2latent = nn.Linear(self.model_size, self.latent_size)
        # self.latent2memory = nn.Linear(self.latent_size, self.model_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences.
        """
        latent = self.encode(src, src_mask)  # (batch_size, max_src_seq, d_model)
        latent = self.sigmoid(latent)
        # memory = self.position_layer(memory)

        latent = torch.sum(latent, dim=1)  # (batch_size, d_model)
#        logit=self.decode(latent.unsqueeze(1), tgt, style, tgt_mask)
        logit=self.decode(latent.unsqueeze(1), tgt, tgt_mask)
#        logit = self.decode(latent.unsqueeze(1), tgt, tgt_mask)  # (batch_size, max_tgt_seq, d_model)
        prob = self.generator(logit)  # (batch_size, max_seq, vocab_size)
        return latent, prob
    
    def getGpu(self):
        return self.gpu

    def infer_encode(self, src, src_mask, style):
        style_mod=1-style #style_transfer #128,1
        #style_whole=torch.cat((style, style_mod), 1)
        
        emb=self.src_embed(src)   #(batch, seq, dim)
        score=self.getWeight(get_cuda(emb.clone().detach(), self.gpu), style_mod)    #bath, seq
        input=score*emb
        return self.encoder(input, src_mask)

    def getMemory(self, src, src_mask):
        latent = self.encode(src, src_mask)
        latent=self.sigmoid(latent)
        # memory = self.position_layer(memory)

        return latent
    
    def getLatent(self, src, src_mask):
        """
        Take in and process masked src and target sequences.
        """
        latent = self.encode(src, src_mask)  # (batch_size, max_src_seq, d_model)
        latent = self.sigmoid(latent)
        # memory = self.position_layer(memory)
        latent = torch.sum(latent, dim=1)  # (batch_size, d_model)
        return latent 

    def getOutput(self, latent, tgt, tgt_mask):
#        logit=self.decode(latent.unsqueeze(1), tgt, style, tgt_mask)
        logit=self.decode(latent.unsqueeze(1), tgt, tgt_mask)
        prob=self.generator(logit)
        return prob

    def getSim(self, latent, style=None):
        #latent_norm=torch.norm(latent, dim=-1) #batch, dim
        latent_clone=get_cuda(latent.clone(), self.gpu)
        if style is not None:
            style=style.unsqueeze(2)
            style=self.style_embed(style.long())
            pdb.set_trace()
            style=style.reshape(style.size(0), style.size(1), style.size(-1))
        else:
            style=get_cuda(torch.tensor([[0], [1]]).long(), self.gpu)
            style=torch.cat(latent.size(0)*[style]) #128, 2, 1
            style=style.reshape(latent_clone.size(0), -1, 1)
            style=self.style_embed(style) #(batch. style_num, 1, dim)
            style=style.reshape(style.size(0), style.size(1), -1)
        
        dot=torch.bmm(style, latent_clone.unsqueeze(2)) #batch, style_num, 1
        dot=dot.reshape(dot.size(0), dot.size(1))
        return style, dot

    def encode(self, src, src_mask):
#        pdb.set_trace()
        return self.encoder(self.src_embed(src), src_mask)
     
#    def decode(self, memory, tgt, style, tgt_mask):
    def decode(self, memory, tgt, tgt_mask):
        # memory: (batch_size, 1, d_model)=latent
        src_mask = get_cuda(torch.ones(memory.size(0), 1, 1).long(), self.gpu)

        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, )

#    def greedy_decode(self, latent, style, max_len, start_id):
    def greedy_decode(self, latent, max_len, start_id):
        '''
        latent: (batch_size, max_src_seq, d_model)
        src_mask: (batch_size, 1, max_src_len)
        '''
        batch_size = latent.size(0)
        # memory = self.latent2memory(latent)
        ys = get_cuda(torch.ones(batch_size, 1).fill_(start_id).long(), self.gpu)  # (batch_size, 1)
        for i in range(max_len - 1):
            out = self.decode(latent.unsqueeze(1), to_var(ys, self.gpu), to_var(subsequent_mask(ys.size(1)).long(), self.gpu))
            prob = self.generator(out[:, -1])
            # print("prob", prob.size())  # (batch_size, vocab_size)
            _, next_word = torch.max(prob, dim=1)
            # print("next_word", next_word.size())  # (batch_size)

            # print("next_word.unsqueeze(1)", next_word.unsqueeze(1).size())

            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
            # print("ys", ys.size())
        return ys[:, 1:]

    def beam_decode(self, latent, beam_size, max_len, start_id):
        '''
        latent: (batch_size, max_src_seq, d_model)
        src_mask: (batch_size, 1, max_src_len)
        '''
        memory_beam = latent.detach().repeat(beam_size, 1, 1)
        beam = Beam(beam_size=beam_size, min_length=0, n_top=beam_size, ranker=None) 
        batch_size = latent.size(0)
        candidate=get_cuda(torch.zeros(beam_size, batch_size, max_len), self.gpu)
        global_scores=get_cuda(torch.zeros(beam_size), self.gpu)
        
        tmp_cand=get_cuda(torch.zeros(beam_size*beam_size), self.gpu)
        tmp_scores=get_cuda(torch.zeros(beam_size*beam_size), self.gpu)
        
        ys = get_cuda(torch.ones(batch_size, 1).fill_(start_id).long(), self.gpu)  # (batch_size, 1)
        candidate[:,:,0]=ys.clone()
        #first 
        out = self.decode(latent.unsqueeze(1), to_var(ys, self.gpu), to_var(subsequent_mask(ys.size(1)).long(), self.gpu))
        prob = self.generator(out[:, -1])
        scores, ids=prob.topk(k=beam_size, dim=1) #shape:1,baem_size
        global_scores=scores.view(-1)
        candidate[:,:,1]=ids.transpose(0,1)
        for i in range(1,max_len-1):
            for j in range(beam_size):
#                candidate[j,:,:i+1] = torch.cat([candidate[j,:,:i], ids[j]], dim=-1)
                tmp=candidate[j,:,:i+1].view(1,-1)
                #tmp_cand:3
                tp, tc=self.recursive_beam(beam_size, latent.unsqueeze(1), to_var(tmp.long(), self.gpu), to_var(subsequent_mask(tmp.size(1)).long(), self.gpu))
                tmp_cand[beam_size*j:beam_size*(j+1)]=tc.view(-1)
                tmp_scores[beam_size*j:beam_size*(j+1)]=tp.view(-1) + global_scores[j]
            beam_head_scores, beam_head_ids=tmp_scores.topk(k=beam_size, dim=0)
            global_scores=beam_head_scores
            can_list=[]
            for bb in range(beam_size):
                can_list.append(torch.cat([candidate[int(beam_head_ids[bb].item()/beam_size),:,:i+1].long(), tmp_cand[beam_head_ids[bb]].long().unsqueeze(0).unsqueeze(0)], dim=1))
#            c2=torch.cat([candidate[int(beam_head_ids[1].item()/beam_size),:,:i+1].long(), tmp_cand[beam_head_ids[1]].long().unsqueeze(0).unsqueeze(0)], dim=1)
#            c3=torch.cat([candidate[int(beam_head_ids[2].item()/3),:,:i+1].long(), tmp_cand[beam_head_ids[2]].long().unsqueeze(0).unsqueeze(0)], dim=1)
            for bb in range(beam_size):
                candidate[bb, :, :i+2]=can_list[bb]
#            candidate[0,:,:i+2]=c1
#            candidate[1,:,:i+2]=c2
#            candidate[2,:,:i+2]=c3
        top_s, top_i = global_scores.sort()
        candidate=candidate.view(beam_size, -1)
        candidate=candidate[:,1:]
        sorted_candidate=candidate.clone()
        for bb in range(beam_size):
            sorted_candidate[bb]=candidate[top_i[bb]]
        return sorted_candidate.long().view(beam_size, -1)

    def recursive_beam(self, beam_size, latent, input, mask):
        cand=torch.zeros(beam_size)
        cand_p=torch.zeros(beam_size)
        out=self.decode(latent, input, mask)
        prob=self.generator(out[:, -1])
        scores, ids=prob.topk(k=beam_size, dim=1)
        cand=ids.transpose(0,1)
        cand_p=scores.transpose(0,1)
        return cand_p, cand

#
#        candidate=torch.zeros(beam_size, batch_size, max_len)
#        for b in range(beam_size):
#            ys = get_cuda(torch.ones(batch_size, 1).fill_(start_id).long(), self.gpu)  # (batch_size, 1)
#            for i in range(max_len - 1):
#                out = self.decode(latent.unsqueeze(1), to_var(ys, self.gpu), to_var(subsequent_mask(ys.size(1)).long(), self.gpu))
#                prob = self.generator(out[:, -1])
#                # print("prob", prob.size())  # (batch_size, vocab_size)
#                
#                prob.topk(k=b, dim=1)
#                #_, next_word = torch.max(prob, dim=1)
#                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
#            candidate[b]=ys[:, 1:]
#            
#            pdb.set_trace()
#        for i in range(max_len - 1):
#            new_inputs=beam.get_current_state().unsqueeze(1) #beam_size, seq_len=1
##            new_inputs=beam.get_new_input()
#            out=self.decode(memory_beam, to_var(new_inputs, self.gpu), to_var(subsequent_mask(new_inputs.size(1)).long(), self.gpu))
#            prob=self.generator(out)
#            attention_b=self.decoder.layers[-1].src_attn.attn 
#            attention_b=attention_b.squeeze()
#            attention_b=attention_b.unsqueeze(0)
#
#            beam.advance(prob.squeeze(1), attention_b)
#            beam_current_origin=beam.get_current_origin()
#            
#            if beam.done():
#                break
#
#        scores, ks = beam.sort_finished(1)
#        hypothesises, attentions=[],[]
#        for i, (times, k) in enumerate(ks[:1]):
##            hypothesis, attention=beam.get_hypothesis(times, k)
#            hypothesis=beam.get_hypothesis(times, k)
#            hypothesises.append(hypothesis)
##            attentions.append(attention)
#
##        self.attentions=attentions
#        self.hypothesises=[[token.item() for token in h] for h in hypothesises]
#            
#        hs = [h for h in self.hypothesises]
#        #pdb.set_trace()
#        return list(reversed(hs))
            

def make_model(d_vocab, N, d_model, latent_size, gpu, d_ff=1024, h=4, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    d_style=128
    position = PositionalEncoding(d_model, dropout)
    share_embedding = Embeddings(d_model, d_vocab)
    style_embedding=StyleEmbeddings(2, d_model)
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(share_embedding, c(position)),
        nn.Sequential(share_embedding, c(position)),
        style_embedding,
        Generator(d_model, d_vocab),
        c(position),
        d_model,
        latent_size,
        gpu
        )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class Projection(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.style_proj=nn.Sequential(
                    nn.Linear(latent_size, latent_size//2),
                    nn.LeakyReLU(0.2)
                    )
        self.rand_proj=nn.Sequential(
                    nn.Linear(latent_size, latent_size//2),
                    nn.LeakyReLU(0.2)
                    )
        self.proj=nn.Sequential(
                    nn.Linear(latent_size, latent_size),
                    nn.LeakyReLU(0.2)
                    )
        
    def forward(self, style_input, rand_input): 
        latent=self.proj(torch.cat((style_input, rand_input), -1))
        return latent

    def style(self, input):
        return self.style_proj(input)

    def random(self, input):
        return self.rand_proj(input)

class Classifier(nn.Module):
    def __init__(self, output_size, gpu):
        super().__init__()
        self.gpu=gpu
        self.fc1 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def getGpu(self):
        return self.gpu

    def forward(self, input):
        out = self.fc1(input)
        out = self.sigmoid(out)

        # out = F.log_softmax(out, dim=1)
        return out  # batch_size * label_size

    def getFc(self, input):
        return self.fc1(input)

    def getSig(self, input):
        return self.sigmoid(input)

class Projection(nn.Module):
    def __init__(self, latent_size, output_size, gpu):
        super().__init__()
        self.gpu=gpu
        self.fc1 = nn.Linear(latent_size, 100)
        self.relu1 = nn.LeakyReLU(0.2, )
        self.fc2 = nn.Linear(100, 50)
        self.relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(50, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def getGpu(self):
        return self.gpu

    def forward(self, input):
        out = self.fc1(input)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)

        # out = F.log_softmax(out, dim=1)
        return out  # batch_size * label_size
