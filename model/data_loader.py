import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import os
from pathlib import Path
import pickle
import copy
import pandas as pd
import random
import csv


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label


class DATA_LOADER(object):
    def __init__(self, dataset, aux_datasource, device='cuda'):
        print("The current working directory is")
        print(os.getcwd())
        folder = str(Path(os.getcwd()))
        print(folder)
        if folder[-5:] == 'model':
            project_directory = Path(os.getcwd()).parent
        else:
            project_directory = folder

        print('Project Directory--------------------------:')
        print(project_directory)
        data_path = str(project_directory) + '/data'
        print('Data Path')
        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.device = device
        self.dataset = dataset
        self.auxiliary_data_source = aux_datasource
        self.all_data_sources = ['resnet_features'] + [self.auxiliary_data_source]
        if self.dataset == 'RS':
            self.datadir = self.data_path + '/RS/'
        self.read_matdataset()
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def next_batch(self, batch_size):
        #####################################################################
        # gets batch from train_feature = 7057 samples from 150 train classes
        #####################################################################
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.data['train_seen']['resnet_features'][idx]
        batch_label = self.data['train_seen']['labels'][idx]
        batch_att = self.aux_data[batch_label]
        return batch_label, [batch_feature, batch_att]

    def text_read(self, filename):
        try:
            file = open(filename, 'r')
        except IOError:
            error = []
            return error
        content = file.readlines()
        for i in range(len(content)):
            content[i] = content[i][:len(content[i]) - 1]
        file.close()
        return content

    def read_matdataset(self):
        vis_features_path = self.datadir + "..."
        label_path = self.datadir + "/label.txt"
        feature = []
        with open(vis_features_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                feature.append(line)
        feature = np.array(feature)
        label = text_read(label_path)
        label = np.array(label).squeeze().astype(int) - 1
        trainval_loc = self.text_read(self.datadir + "/...")
        trainval_loc = np.array(trainval_loc).squeeze().astype(int)
        test_unseen_loc = self.text_read(self.datadir + "/...")
        test_unseen_loc = np.array(test_unseen_loc).squeeze().astype(int)
        id_test_seen = self.text_read('/...')
        test_seen_loc = np.array(id_test_seen).squeeze().astype(int) - 1
        '''
        path = self.datadir + "/features/S/..."
        sem_feats = np.load(path)
        self.aux_data = torch.from_numpy(sem_feats).float().to(self.device)
        '''
        w2v = []
        path = self.datadir + "/features/S/..."
        data = sio.loadmat(path)
        for i in data:
            w2v.append(data[i])
        new_w2v = w2v[3:]
        sem_feats = np.array(new_w2v).squeeze()  #
        self.aux_data = torch.from_numpy(sem_feats).float().to(self.device)
        scaler = preprocessing.MinMaxScaler()
        train_feature = scaler.fit_transform(feature[trainval_loc])
        test_seen_feature = scaler.transform(feature[test_seen_loc])
        test_unseen_feature = scaler.transform(feature[test_unseen_loc])
        train_feature = torch.from_numpy(train_feature).float().to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature).float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature).float().to(self.device)
        train_label = torch.from_numpy(label[trainval_loc]).long().to(self.device)
        test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long().to(self.device)
        test_seen_label = torch.from_numpy(label[test_seen_loc]).long().to(self.device)
        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device)
        self.novelclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)
        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.novelclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.train_mapped_label = map_label(train_label, self.seenclasses)
        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels'] = train_label
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[train_label]
        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None
        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label
        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[test_unseen_label]
        self.data['test_unseen']['labels'] = test_unseen_label
        self.novelclass_aux_data = self.aux_data[self.novelclasses]
        self.seenclass_aux_data = self.aux_data[self.seenclasses]

    def transfer_features(self, n, num_queries='num_features'):
        for i, s in enumerate(self.novelclasses):
            features_of_that_class = self.data['test_unseen']['resnet_features'][
                                     self.data['test_unseen']['labels'] == s, :]
            if 'attributes' == self.auxiliary_data_source:
                attributes_of_that_class = self.data['test_unseen']['attributes'][
                                           self.data['test_unseen']['labels'] == s, :]
                use_att = True
            else:
                use_att = False
            if 'sentences' == self.auxiliary_data_source:
                sentences_of_that_class = self.data['test_unseen']['sentences'][self.data['test_unseen']['labels'] == s,
                                          :]
                use_stc = True
            else:
                use_stc = False
            if 'word2vec' == self.auxiliary_data_source:
                word2vec_of_that_class = self.data['test_unseen']['word2vec'][self.data['test_unseen']['labels'] == s,
                                         :]
                use_w2v = True
            else:
                use_w2v = False
            if 'glove' == self.auxiliary_data_source:
                glove_of_that_class = self.data['test_unseen']['glove'][self.data['test_unseen']['labels'] == s, :]
                use_glo = True
            else:
                use_glo = False
            if 'wordnet' == self.auxiliary_data_source:
                wordnet_of_that_class = self.data['test_unseen']['wordnet'][self.data['test_unseen']['labels'] == s, :]
                use_hie = True
            else:
                use_hie = False
            num_features = features_of_that_class.size(0)
            indices = torch.randperm(num_features)
            if num_queries != 'num_features':
                indices = indices[:n + num_queries]
            if i == 0:

                new_train_unseen = features_of_that_class[indices[:n], :]

                if use_att:
                    new_train_unseen_att = attributes_of_that_class[indices[:n], :]
                if use_stc:
                    new_train_unseen_stc = sentences_of_that_class[indices[:n], :]
                if use_w2v:
                    new_train_unseen_w2v = word2vec_of_that_class[indices[:n], :]
                if use_glo:
                    new_train_unseen_glo = glove_of_that_class[indices[:n], :]
                if use_hie:
                    new_train_unseen_hie = wordnet_of_that_class[indices[:n], :]
                new_train_unseen_label = s.repeat(n)
                new_test_unseen = features_of_that_class[indices[n:], :]
                new_test_unseen_label = s.repeat(len(indices[n:]))
            else:
                new_train_unseen = torch.cat((new_train_unseen, features_of_that_class[indices[:n], :]), dim=0)
                new_train_unseen_label = torch.cat((new_train_unseen_label, s.repeat(n)), dim=0)

                new_test_unseen = torch.cat((new_test_unseen, features_of_that_class[indices[n:], :]), dim=0)
                new_test_unseen_label = torch.cat((new_test_unseen_label, s.repeat(len(indices[n:]))), dim=0)

                if use_att:
                    new_train_unseen_att = torch.cat((new_train_unseen_att, attributes_of_that_class[indices[:n], :]),
                                                     dim=0)
                if use_stc:
                    new_train_unseen_stc = torch.cat((new_train_unseen_stc, sentences_of_that_class[indices[:n], :]),
                                                     dim=0)
                if use_w2v:
                    new_train_unseen_w2v = torch.cat((new_train_unseen_w2v, word2vec_of_that_class[indices[:n], :]),
                                                     dim=0)
                if use_glo:
                    new_train_unseen_glo = torch.cat((new_train_unseen_glo, glove_of_that_class[indices[:n], :]), dim=0)
                if use_hie:
                    new_train_unseen_hie = torch.cat((new_train_unseen_hie, wordnet_of_that_class[indices[:n], :]),
                                                     dim=0)
        self.data['test_unseen']['resnet_features'] = copy.deepcopy(new_test_unseen)
        self.data['test_unseen']['labels'] = copy.deepcopy(new_test_unseen_label)
        self.data['train_unseen']['resnet_features'] = copy.deepcopy(new_train_unseen)
        self.data['train_unseen']['labels'] = copy.deepcopy(new_train_unseen_label)
        self.ntrain_unseen = self.data['train_unseen']['resnet_features'].size(0)
        if use_att:
            self.data['train_unseen']['attributes'] = copy.deepcopy(new_train_unseen_att)
        if use_w2v:
            self.data['train_unseen']['word2vec'] = copy.deepcopy(new_train_unseen_w2v)
        if use_stc:
            self.data['train_unseen']['sentences'] = copy.deepcopy(new_train_unseen_stc)
        if use_glo:
            self.data['train_unseen']['glove'] = copy.deepcopy(new_train_unseen_glo)
        if use_hie:
            self.data['train_unseen']['wordnet'] = copy.deepcopy(new_train_unseen_hie)
        self.data['train_seen_unseen_mixed'] = {}
        self.data['train_seen_unseen_mixed']['resnet_features'] = torch.cat(
            (self.data['train_seen']['resnet_features'], self.data['train_unseen']['resnet_features']), dim=0)
        self.data['train_seen_unseen_mixed']['labels'] = torch.cat(
            (self.data['train_seen']['labels'], self.data['train_unseen']['labels']), dim=0)
        self.ntrain_mixed = self.data['train_seen_unseen_mixed']['resnet_features'].size(0)
        if use_att:
            self.data['train_seen_unseen_mixed']['attributes'] = torch.cat(
                (self.data['train_seen']['attributes'], self.data['train_unseen']['attributes']), dim=0)
        if use_w2v:
            self.data['train_seen_unseen_mixed']['word2vec'] = torch.cat(
                (self.data['train_seen']['word2vec'], self.data['train_unseen']['word2vec']), dim=0)
        if use_stc:
            self.data['train_seen_unseen_mixed']['sentences'] = torch.cat(
                (self.data['train_seen']['sentences'], self.data['train_unseen']['sentences']), dim=0)
        if use_glo:
            self.data['train_seen_unseen_mixed']['glove'] = torch.cat(
                (self.data['train_seen']['glove'], self.data['train_unseen']['glove']), dim=0)
        if use_hie:
            self.data['train_seen_unseen_mixed']['wordnet'] = torch.cat(
                (self.data['train_seen']['wordnet'], self.data['train_unseen']['wordnet']), dim=0)


def text_read(filename):
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content
