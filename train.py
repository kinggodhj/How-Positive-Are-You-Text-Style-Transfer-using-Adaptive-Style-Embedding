import numpy as np
import pdb
import time
import torch
from sklearn.metrics import accuracy_score

from data import get_cuda, to_var
from model import LabelSmoothing

def train_pick(ae_model, dis_model, ae_optimizer, dis_optimizer, train_data_loader, epoch, args):
    print("Transformer Training process....")
    ae_model.train()

    print('-' * 94)
    epoch_start=time.time()

    loss_ae=list()
    loss_dis=list()
    acc=list() 
    
    ae_criterion=get_cuda(LabelSmoothing(size=args.vocab_size, padding_idx=args.id_pad, smoothing=0.1), args.gpu)
    dis_criterion=torch.nn.BCELoss(size_average=True)
    for it in range(train_data_loader.num_batch):
        ####################
        #####load data######
        ####################
        batch_sentences, tensor_labels, \
            tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_y, \
            tensor_tgt_mask, tensor_ntokens = train_data_loader.next_batch()
    
           
        latent=ae_model.getLatent(tensor_src, tensor_src_mask) #(128, 256)
        style, similarity=ae_model.getSim(latent) #style (128, 2, 256), sim(128, 2)
        dis_out=dis_model.forward(similarity)
        one=get_cuda(torch.tensor(1), args.gpu)
        zero=get_cuda(torch.tensor(0), args.gpu)
        style_pred=torch.where(dis_out>0.5, one, zero)
        style_pred=style_pred.reshape(style_pred.size(0))
        style_emb=get_cuda(style.clone()[torch.arange(style.size(0)), tensor_labels.squeeze().long()], args.gpu) #(128, 256)

        add_latent=latent+style_emb #batch, dim
        out=ae_model.getOutput(add_latent, tensor_tgt, tensor_tgt_mask)
        loss_rec=ae_criterion(out.contiguous().view(-1, out.size(-1)), tensor_tgt_y.contiguous().view(-1)) / tensor_ntokens.data

        loss_style=dis_criterion(dis_out, tensor_labels)
        
        pred=style_pred.to('cpu').detach().tolist()
        true=tensor_labels.squeeze().to('cpu').tolist()

        dis_acc=accuracy_score(pred, true)
        acc.append(dis_acc)
        loss=loss_rec+loss_style
        
        ae_optimizer.optimizer.zero_grad()
        dis_optimizer.zero_grad()
        loss.backward()
        ae_optimizer.step()
        dis_optimizer.step()

        loss_ae.append(loss_rec.item())
        loss_dis.append(loss_style.item())
        
        if it % 200 == 0:
            print('| epoch {:3d} | {:5d}/{:5d} batches |\n| rec loss {:5.4f} | dis loss {:5.4f} |\n'
            .format(epoch, it, train_data_loader.num_batch, loss_rec.item(), loss_style.item()))

    print('| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start)))
    
    return np.mean(loss_ae), np.mean(loss_dis), np.mean(acc), ae_model, dis_model, ae_optimizer, dis_optimizer

def val_pick(ae_model, dis_model, eval_data_loader, epoch, args):
    print("Transformer Validation process....")
    ae_model.eval()

    print('-' * 94)
    epoch_start=time.time()

    loss_ae=list()
    loss_dis=list()
   
    acc=list()
    
    ae_criterion=get_cuda(LabelSmoothing(size=args.vocab_size, padding_idx=args.id_pad, smoothing=0.1), args.gpu)
    dis_criterion=torch.nn.BCELoss(size_average=True)
    for it in range(eval_data_loader.num_batch):
        ####################
        #####load data######
        ####################
        batch_sentences, tensor_labels, \
            tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_y, \
            tensor_tgt_mask, tensor_ntokens = eval_data_loader.next_batch()
        

        latent=ae_model.getLatent(tensor_src, tensor_src_mask) #(128, 256)
        style, similarity=ae_model.getSim(latent) #style (128, 2, 256), sim(128, 2)
        dis_out=dis_model.forward(similarity)
        one=get_cuda(torch.tensor(1), args.gpu)
        zero=get_cuda(torch.tensor(0), args.gpu)
        style_pred=torch.where(dis_out>0.5, one, zero)
        style_pred=style_pred.reshape(style_pred.size(0))
        style_emb=get_cuda(style.clone()[torch.arange(style.size(0)), tensor_labels.squeeze().long()], args.gpu) #(128, 256)

        add_latent=latent+style_emb #batch, dim
        out=ae_model.getOutput(add_latent, tensor_tgt, tensor_tgt_mask)
        loss_rec=ae_criterion(out.contiguous().view(-1, out.size(-1)), tensor_tgt_y.contiguous().view(-1)) / tensor_ntokens.data

        loss_style=dis_criterion(dis_out, tensor_labels)
        
        pred=style_pred.to('cpu').detach().tolist()
        true=tensor_labels.squeeze().to('cpu').tolist()

        dis_acc=accuracy_score(pred, true)
        acc.append(dis_acc)
        
        loss_ae.append(loss_rec.item())
        loss_dis.append(loss_style.item())
        
        if it % 200 == 0:
            print('| epoch {:3d} | {:5d}/{:5d} batches |\n| rec loss {:5.4f} | dis loss {:5.4f} |\n'
            .format(epoch, it, eval_data_loader.num_batch, loss_rec.item(), loss_style.item()))

    print('| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start)))
    
    return np.mean(loss_ae), np.mean(loss_dis), np.mean(acc)
