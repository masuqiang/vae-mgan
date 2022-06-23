
from vaemodel import Model
import numpy as np
import pickle
import torch
import os
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

def main(model, lamda,CMFM_loss, beta):
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--num_shots', type=int)
    parser.add_argument('--generalized', type=str2bool)
    parser.add_argument('--model', default=model)
    parser.add_argument('--lamda', default=lamda)
    parser.add_argument('--CMFM_loss', default=CMFM_loss)
    parser.add_argument('--beta', default=beta)
    args = parser.parse_args()
    hyperparameters = {
        'num_shots': 0,
        'device': 'cuda',
        'model_specifics': {'cross_reconstruction': True,
                            'name': 'CADA',
                            'distance': 'wasserstein',
                            },

        'lr_gen_model': 0.00015,
        'generalized': True,
        'batch_size': 50,
        'xyu_samples_per_class': {
                                  'RS': (200, 0, 400, 0),
                                  },
        'loss': 'l1',
        'auxiliary_data_source': 'attributes',
        'lr_cls': 0.001,
        'dataset': 'RS',
        'hidden_size_rule': {'resnet_features': (512, 512),
                             'attributes': (256, 256),
                             'sentences': (1450, 665)},
        'latent_size': 32,
        'x_dim': 512,
        'num_classes': 70,
        'num_novel_classes': 10,
        'epochs': 50
    }
    cls_train_steps = [{'dataset': 'RS', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 50},   ]
    hyperparameters['dataset'] = args.dataset
    hyperparameters['num_shots'] = args.num_shots
    hyperparameters['generalized'] = args.generalized
    hyperparameters['cls_train_steps'] = [x['cls_train_steps'] for x in cls_train_steps
                                          if all([hyperparameters['dataset'] == x['dataset'],
                                                  hyperparameters['num_shots'] == x['num_shots'],
                                                  hyperparameters['generalized'] == x['generalized']])][0]
    hyperparameters['model'] = args.model
    hyperparameters['lamda'] = args.lamda
    hyperparameters['CMFM_loss'] = args.CMFM_loss
    hyperparameters['beta'] = args.beta

    if hyperparameters['generalized']:
        if hyperparameters[ 'num_shots'] == 0:
            hyperparameters['samples_per_class'] = {'RS': (200, 0, 400, 0)}
        else:
            hyperparameters['samples_per_class'] = {'RS': (200, 0, 200, 200)}
    else:
        if hyperparameters['num_shots'] == 0:
            hyperparameters['samples_per_class'] = {'RS': (0, 0, 200, 0)}
        else:
            hyperparameters['samples_per_class'] = {'RS': (0, 0, 200, 200)}
    model = Model(hyperparameters)
    model.to(hyperparameters['device'])
    saved_state = torch.load('./saved_models/CADA_trained.pth.tar')
    model.load_state_dict(saved_state['state_dict'])
    for d in model.all_data_sources_without_duplicates:
        model.encoder[d].load_state_dict(saved_state['encoder'][d])
        model.decoder[d].load_state_dict(saved_state['decoder'][d])

    losses = model.train_vae()
    s,u,h, history = model.train_classifier()
    best_A=0.0
    acc = []
    if hyperparameters['generalized'] == True:
        acc = [hi[2] for hi in history]
    elif hyperparameters['generalized'] == False:
        acc = [hi[1] for hi in history]
    for i in acc:
        if best_A<i:
            best_A=i
    return best_A



if __name__ == '__main__':
