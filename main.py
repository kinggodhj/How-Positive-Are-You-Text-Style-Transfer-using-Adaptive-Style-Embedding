# coding: utf-8
import argparse
import time
import math
import numpy as np
import os
import pdb
import torch
from tensorboardX import SummaryWriter

# Import your model files.
from data import get_cuda, id2text_sentence, to_var, non_pair_data_loader
from model import make_model, Classifier, NoamOpt
from setup import preparation, read_file, write_file
from train import train_pick, val_pick

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

######################################################################################
#  Environmental parameters
######################################################################################
parser = argparse.ArgumentParser(description="Here is your model discription.")
parser.add_argument('--id_pad', type=int, default=0, help='')
parser.add_argument('--id_unk', type=int, default=1, help='')
parser.add_argument('--id_bos', type=int, default=2, help='')
parser.add_argument('--id_eos', type=int, default=3, help='')

######################################################################################
#  File parameters
######################################################################################
parser.add_argument('--word_to_id_file', type=str, default='', help='')
parser.add_argument('--data_path', type=str, default='./data/yelp/processed_files/', help='')

######################################################################################
#  Model parameters
######################################################################################
parser.add_argument('--word_dict_max_num', type=int, default=5, help='')
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--max_sequence_length', type=int, default=20)
parser.add_argument('--num_layers_AE', type=int, default=2)
parser.add_argument('--transformer_model_size', type=int, default=256)
parser.add_argument('--transformer_ff_size', type=int, default=1024)
parser.add_argument('--epoch', type=int, default=200)

parser.add_argument('--latent_size', type=int, default=256)
parser.add_argument('--word_dropout', type=float, default=1.0)
parser.add_argument('--embedding_dropout', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--label_size', type=int, default=1)

parser.add_argument('--gpu', type=int, default=2)

parser.add_argument('--name', type=str, default='pick')
parser.add_argument('--load_model', type=bool)
parser.add_argument('--load_iter', type=int)
args = parser.parse_args()

device=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

summary=SummaryWriter('./runs/{}'.format(args.name))
######################################################################################
#  End of hyper parameters
######################################################################################

if __name__ == '__main__':
    preparation(args)

    ae_model = get_cuda(make_model(d_vocab=args.vocab_size,
                                   N=args.num_layers_AE,
                                   d_model=args.transformer_model_size,
                                   latent_size=args.latent_size,
                                   gpu=args.gpu, 
                                   d_ff=args.transformer_ff_size), args.gpu)

    dis_model=get_cuda(Classifier(1, args),args.gpu)
    ae_optimizer = NoamOpt(ae_model.src_embed[0].d_model, 1, 2000,
                           torch.optim.Adam(ae_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    dis_optimizer=torch.optim.Adam(dis_model.parameters(), lr=0.0001)
    if args.load_model:
        # Load models' params from checkpoint
        ae_model.load_state_dict(torch.load(args.current_save_path + '/{}_ae_model_params.pkl'.format(args.load_iter)))
        dis_model.load_state_dict(torch.load(args.current_save_path + '/{}_dis_model_params.pkl'.format(args.load_iter)))
       
        start=args.load_iter+1 
    else:
        start=0        
    train_data_loader=non_pair_data_loader(
        batch_size=args.batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size,
        gpu=args.gpu
    )

    train_data_loader.create_batches(args.train_file_list, args.train_label_list, if_shuffle=True)

    eval_data_loader=non_pair_data_loader(
        batch_size=args.batch_size, id_bos=args.id_bos,
        id_eos=args.id_eos, id_unk=args.id_unk,
        max_sequence_length=args.max_sequence_length, vocab_size=args.vocab_size,
        gpu=args.gpu
    )

    eval_file_list=[
        args.data_path+'sentiment.dev.0',
        args.data_path+'sentiment.dev.1',
    ]

    eval_label_list=[
        [0],
        [1],
    ]
    eval_data_loader.create_batches(eval_file_list, eval_label_list, if_shuffle=False)

    for epoch in range(start, args.epoch):
        loss_ae, loss_dis, acc, ae_model, dis_model, ae_optimizer, dis_optimizer=train_pick(ae_model, dis_model, ae_optimizer, dis_optimizer, train_data_loader, epoch, args)
            
        summary.add_scalar('{}/train/Transformer'.format(args.name), loss_ae, epoch)
        summary.add_scalar('{}/train/Discriminator'.format(args.name), loss_dis, epoch)
        summary.add_scalar('{}/train/Acc'.format(args.name), acc, epoch)
            
        torch.save(ae_model.state_dict(), args.current_save_path + '/{}_ae_model_params.pkl'.format(epoch))
        torch.save(dis_model.state_dict(), args.current_save_path + '/{}_dis_model_params.pkl'.format(epoch))
            
        loss_v_ae, loss_v_dis, acc=val_pick(ae_model, dis_model, eval_data_loader, epoch, args)

        summary.add_scalar('{}/val/Transformer'.format(args.name), loss_v_ae, epoch)
        summary.add_scalar('{}/val/Discriminator'.format(args.name), loss_v_dis, epoch)
        summary.add_scalar('{}/val/Acc'.format(args.name), acc, epoch)
    
    print("Done!")

