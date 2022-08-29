import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data
from data_loader import DATA_LOADER as dataloader
import final_classifier as classifier
import models
from torch.autograd import Variable
import tsne_plot

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction = nn.NLLLoss()
    def forward(self, x):
        o = self.logic(self.fc(x))
        return o
class Model(nn.Module):
    def __init__(self, hyperparameters):
        super(Model, self).__init__()
        self.device = hyperparameters['device']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        self.all_data_sources = ['resnet_features', self.auxiliary_data_source]
        self.DATASET = hyperparameters['dataset']
        self.num_shots = hyperparameters['num_shots']
        self.latent_size = hyperparameters['latent_size']
        self.batch_size = hyperparameters['batch_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.warmup = hyperparameters['model_specifics']['warmup']
        self.generalized = hyperparameters['generalized']
        self.classifier_batch_size = 32
        self.img_seen_samples = hyperparameters['samples_per_class'][self.DATASET][0]
        self.att_seen_samples = hyperparameters['samples_per_class'][self.DATASET][1]
        self.att_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][2]
        self.img_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][3]
        self.reco_loss_function = hyperparameters['loss']
        self.nepoch = hyperparameters['epochs']
        self.lr_cls = hyperparameters['lr_cls']
        self.cross_reconstruction = hyperparameters['model_specifics']['cross_reconstruction']
        self.cls_train_epochs = hyperparameters['cls_train_steps']
        self.dataset = dataloader(self.DATASET, copy.deepcopy(self.auxiliary_data_source), device=self.device)
        self.x_dim = hyperparameters['x_dim']
        self.num_classes = hyperparameters['num_classes']
        self.num_novel_classes = hyperparameters['num_novel_classes']
        self.model = hyperparameters['model']
        self.lamda = hyperparameters['lamda']
        self.beta = hyperparameters['beta']
        self.CD_loss = hyperparameters['CD_loss']
        feature_dimensions = [ self.x_dim, self.dataset.aux_data.size(1)]
        self.encoder = {}
        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.encoder[datatype] = models.encoder_template(dim, self.latent_size, self.hidden_size_rule[datatype],
                                                             self.device)
            print(str(datatype) + ' ' + str(dim))
        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size, dim, self.hidden_size_rule[datatype],
                                                             self.device)
        parameters_to_optimize = list(self.parameters())
        for datatype in self.all_data_sources:
            parameters_to_optimize += list(self.encoder[datatype].parameters())
            parameters_to_optimize += list(self.decoder[datatype].parameters())
        self.optimizer = optim.Adam(parameters_to_optimize, lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999),
                                    eps=1e-08, weight_decay=0, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [30, 60], gamma=0.1, last_epoch=-1)
        if self.reco_loss_function == 'l2':
            self.reconstruction_criterion = nn.MSELoss(size_average=False)
        elif self.reco_loss_function == 'l1':
            self.reconstruction_criterion = nn.L1Loss(size_average= False)  # True, False
        self.mse_loss = nn.MSELoss()
        self.ones = Variable(torch.ones([self.batch_size, 1]), requires_grad=False).float().cuda()
        self.zeros = Variable(torch.zeros([self.batch_size, 1]), requires_grad=False).float().cuda()
        if self.model == "D1+D2":
            self.discriminator_x = models.discriminator_xs(x_dim=self.x_dim, layers='1200')
            self.discriminator_s = models.discriminator_xs(x_dim=self.dataset.aux_data.size(1), layers='256')
            self.dis_x_opt = optim.Adam(self.discriminator_x.parameters(), lr=hyperparameters['lr_gen_model'],
                                        betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
            self.dis_s_opt = optim.Adam(self.discriminator_s.parameters(), lr=hyperparameters['lr_gen_model'],
                                        betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        elif self.model == "D3":
            self.discriminator = models.discriminator(x_dim=self.x_dim, s_dim=self.dataset.aux_data.size(1),
                                                      layers='1200 600')
            self.dis_opt = optim.Adam(self.discriminator.parameters(), lr=hyperparameters['lr_gen_model'],
                                      betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        elif self.model == "D1+D2+D3":
            self.discriminator_x = models.discriminator_xs(x_dim=self.x_dim, layers='1200')
            self.discriminator_s = models.discriminator_xs(x_dim=self.dataset.aux_data.size(1), layers='256')
            self.dis_x_opt = optim.Adam(self.discriminator_x.parameters(), lr=hyperparameters['lr_gen_model'],
                                        betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
            self.dis_s_opt = optim.Adam(self.discriminator_s.parameters(), lr=hyperparameters['lr_gen_model'],
                                        betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
            self.discriminator = models.discriminator(x_dim=self.x_dim, s_dim=self.dataset.aux_data.size(1),
                                                      layers='1200 600')
            self.dis_opt = optim.Adam(self.discriminator.parameters(), lr=hyperparameters['lr_gen_model'],
                                      betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        #########################################################################################
    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
            eps = eps.expand(sigma.size())
            return mu + sigma * eps
        else:
            return mu
    def forward(self):
        pass
    def map_label(self, label, classes):
        mapped_label = torch.LongTensor(label.size()).to(self.device)
        for i in range(classes.size(0)):
            mapped_label[label == classes[i]] = i
        return mapped_label
    def train_dis_step(self, img, att):
        mu_img, logvar_img = self.encoder['resnet_features'](img)
        z_from_img = self.reparameterize(mu_img, logvar_img)
        mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)
        z_from_att = self.reparameterize(mu_att, logvar_att)
        img_from_img = self.decoder['resnet_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)
        img_from_att = self.decoder['resnet_features'](z_from_att)
        att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)
        img_from_att = img_from_att.detach()
        att_from_img = att_from_img.detach()
        img_from_img = img_from_img.detach()
        att_from_att = att_from_att.detach()
        if self.model == "D1+D2":
            true_scores_x = self.discriminator_x.forward(img)
            fake_scores_x_1 = self.discriminator_x.forward(img_from_img)
            fake_scores_x_2 = self.discriminator_x.forward(img_from_att)
            true_scores_s = self.discriminator_s.forward(att)
            fake_scores_s_1 = self.discriminator_s.forward(att_from_att)
            fake_scores_s_2 = self.discriminator_s.forward(att_from_img)
            ADV_loss_x_1 = self.mse_loss(true_scores_x, self.ones)
            ADV_loss_x_2 = self.mse_loss(fake_scores_x_1, self.zeros)
            ADV_loss_x_3 = self.mse_loss(fake_scores_x_2, self.zeros)
            ADV_loss_s_1 = self.mse_loss(true_scores_s, self.ones)
            ADV_loss_s_2 = self.mse_loss(fake_scores_s_1, self.zeros)
            ADV_loss_s_3 = self.mse_loss(fake_scores_s_2, self.zeros)
            ADV_loss_x = ADV_loss_x_1 + ADV_loss_x_2 + ADV_loss_x_3
            # ADV_loss_x = ADV_loss_x_1+ ADV_loss_x_2
            # ADV_loss_x = ADV_loss_x_1 +ADV_loss_x_3
            ADV_loss_s = ADV_loss_s_1 + ADV_loss_s_2 + ADV_loss_s_3
            # ADV_loss_s = ADV_loss_s_1+ADV_loss_s_2
            # ADV_loss_s = ADV_loss_s_1+ADV_loss_s_3
            ADV_loss = ADV_loss_x + ADV_loss_s
            self.discriminator_x.zero_grad()
            self.discriminator_s.zero_grad()
            ADV_loss.backward()
            self.dis_x_opt.step()
            self.dis_s_opt.step()
        elif self.model == "D3":
            true_scores = self.discriminator.forward(img, att)
            fake_scores_1 = self.discriminator.forward(img, att_from_att)
            fake_scores_2 = self.discriminator.forward(img, att_from_img)
            fake_scores_3 = self.discriminator.forward(img_from_img, att)
            fake_scores_4 = self.discriminator.forward(img_from_att, att)
            ADV_loss_0 = self.mse_loss(true_scores, self.ones)
            ADV_loss_1 = self.mse_loss(fake_scores_1, self.zeros)
            ADV_loss_2 = self.mse_loss(fake_scores_2, self.zeros)
            ADV_loss_3 = self.mse_loss(fake_scores_3, self.zeros)
            ADV_loss_4 = self.mse_loss(fake_scores_4, self.zeros)
            ADV_loss = ADV_loss_0 + ADV_loss_1 + ADV_loss_2 + ADV_loss_3 + ADV_loss_4
            self.discriminator.zero_grad()
            ADV_loss.backward()
            self.dis_opt.step()
        elif self.model == "D1+D2+D3":
            true_scores_x = self.discriminator_x.forward(img)
            fake_scores_x_1 = self.discriminator_x.forward(img_from_img)
            fake_scores_x_2 = self.discriminator_x.forward(img_from_att)
            true_scores_s = self.discriminator_s.forward(att)
            fake_scores_s_1 = self.discriminator_s.forward(att_from_att)
            fake_scores_s_2 = self.discriminator_s.forward(att_from_img)
            ADV_loss_x_1 = self.mse_loss(true_scores_x, self.ones)
            ADV_loss_x_2 = self.mse_loss(fake_scores_x_1, self.zeros)
            ADV_loss_x_3 = self.mse_loss(fake_scores_x_2, self.zeros)
            ADV_loss_s_1 = self.mse_loss(true_scores_s, self.ones)
            ADV_loss_s_2 = self.mse_loss(fake_scores_s_1, self.zeros)
            ADV_loss_s_3 = self.mse_loss(fake_scores_s_2, self.zeros)
            ADV_loss_x = ADV_loss_x_1 + ADV_loss_x_2 + ADV_loss_x_3
            # ADV_loss_x = ADV_loss_x_1+ ADV_loss_x_2
            # ADV_loss_x = ADV_loss_x_1 +ADV_loss_x_3
            ADV_loss_s = ADV_loss_s_1 + ADV_loss_s_2 + ADV_loss_s_3
            # ADV_loss_s = ADV_loss_s_1+ADV_loss_s_2
            # ADV_loss_s = ADV_loss_s_1+ADV_loss_s_3
            dis1_loss_xs = ADV_loss_x + ADV_loss_s
            true_scores = self.discriminator.forward(img, att)
            fake_scores_1 = self.discriminator.forward(img, att_from_att)
            fake_scores_2 = self.discriminator.forward(img, att_from_img)
            fake_scores_3 = self.discriminator.forward(img_from_img, att)
            fake_scores_4 = self.discriminator.forward(img_from_att, att)
            ADV_loss_0 = self.mse_loss(true_scores, self.ones)
            ADV_loss_1 = self.mse_loss(fake_scores_1, self.zeros)
            ADV_loss_2 = self.mse_loss(fake_scores_2, self.zeros)
            ADV_loss_3 = self.mse_loss(fake_scores_3, self.zeros)
            ADV_loss_4 = self.mse_loss(fake_scores_4, self.zeros)
            dis2_loss = ADV_loss_0 + ADV_loss_1 + ADV_loss_2 + ADV_loss_3 + ADV_loss_4
            ADV_loss = dis1_loss_xs + dis2_loss
            self.discriminator_x.zero_grad()
            self.discriminator_s.zero_grad()
            self.discriminator.zero_grad()
            ADV_loss.backward()
            self.dis_x_opt.step()
            self.dis_s_opt.step()
            self.dis_opt.step()
        ADV_loss = self.lamda * ADV_loss
        return ADV_loss.item()
    def train_vae_step(self, img, att, label):
        mu_img, logvar_img = self.encoder['resnet_features'](img)
        z_from_img = self.reparameterize(mu_img, logvar_img)
        mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)
        z_from_att = self.reparameterize(mu_att, logvar_att)
        if self.CMFM_loss:
            if self.CMFM_loss == "cos_loss":
                CMFM_loss = models.cos_loss(z_from_img, z_from_att, label)
            if self.CMFM_loss == "mse_loss":
                CMFM_loss = models.mse_loss(z_from_img, z_from_att, label)
            if self. CMFM_loss == "dot_loss":
                CMFM_loss = models.dot_loss(z_from_img, z_from_att, label)
        img_from_img = self.decoder['resnet_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)
        reconstruction_loss = self.reconstruction_criterion(img_from_img, img) \
                              + self.reconstruction_criterion(att_from_att, att)
        img_from_att = self.decoder['resnet_features'](z_from_att)
        att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)
        cross_reconstruction_loss = self.reconstruction_criterion(img_from_att, img) \
                                    + self.reconstruction_criterion(att_from_img, att)
        KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
              + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))
        distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + \
                              torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))
        distance = distance.sum()
        mcdd_loss=models.mcdd_loss(z_from_att)
        if self.model == "D1+D2":
            dis_scores_x_1 = self.discriminator_x.forward(img_from_img)
            dis_scores_x_2 = self.discriminator_x.forward(img_from_att)
            ADV_loss_x_1 = self.mse_loss(dis_scores_x_1, self.ones)
            ADV_loss_x_2 = self.mse_loss(dis_scores_x_2, self.ones)
            ADV_loss_x = ADV_loss_x_1 + ADV_loss_x_2
            dis_scores_s_1 = self.discriminator_s.forward(att_from_att)
            dis_scores_s_2 = self.discriminator_s.forward(att_from_img)
            ADV_loss_s_1 = self.mse_loss(dis_scores_s_1, self.ones)
            ADV_loss_s_2 = self.mse_loss(dis_scores_s_2, self.ones)
            ADV_loss_s = ADV_loss_s_1 + ADV_loss_s_2
            ADV_loss = ADV_loss_x + ADV_loss_s
        elif self.model == "D3":
            dis_scores_1 = self.discriminator.forward(img, att_from_img)
            dis_scores_2 = self.discriminator.forward(img, att_from_att)
            dis_scores_3 = self.discriminator.forward(img_from_att, att)
            dis_scores_4 = self.discriminator.forward(img_from_img, att)
            ADV_loss_1 = self.mse_loss(dis_scores_1, self.ones)
            ADV_loss_2 = self.mse_loss(dis_scores_2, self.ones)
            ADV_loss_3 = self.mse_loss(dis_scores_3, self.ones)
            ADV_loss_4 = self.mse_loss(dis_scores_4, self.ones)
            ADV_loss = ADV_loss_1 + ADV_loss_2 + ADV_loss_3 + ADV_loss_4
        elif self.model == "D1+D2+D3":
            dis_scores_x_1 = self.discriminator_x.forward(img_from_img)
            dis_scores_x_2 = self.discriminator_x.forward(img_from_att)
            ADV_loss_x_1 = self.mse_loss(dis_scores_x_1, self.ones)
            ADV_loss_x_2 = self.mse_loss(dis_scores_x_2, self.ones)
            ADV_loss_x = ADV_loss_x_1 + ADV_loss_x_2
            dis_scores_s_1 = self.discriminator_s.forward(att_from_att)
            dis_scores_s_2 = self.discriminator_s.forward(att_from_img)
            ADV_loss_s_1 = self.mse_loss(dis_scores_s_1, self.ones)
            ADV_loss_s_2 = self.mse_loss(dis_scores_s_2, self.ones)
            ADV_loss_s = ADV_loss_s_1 + ADV_loss_s_2
            dis1_loss_xs = ADV_loss_x + ADV_loss_s
            dis_scores_1 = self.discriminator.forward(img, att_from_img)
            dis_scores_2 = self.discriminator.forward(img, att_from_att)
            dis_scores_3 = self.discriminator.forward(img_from_att, att)
            dis_scores_4 = self.discriminator.forward(img_from_img, att)
            ADV_loss_1 = self.mse_loss(dis_scores_1, self.ones)
            ADV_loss_2 = self.mse_loss(dis_scores_2, self.ones)
            ADV_loss_3 = self.mse_loss(dis_scores_3, self.ones)
            ADV_loss_4 = self.mse_loss(dis_scores_4, self.ones)
            dis2_loss = ADV_loss_1 + ADV_loss_2 + ADV_loss_3 + ADV_loss_4
            ADV_loss = dis1_loss_xs + dis2_loss
        else:
            ADV_loss = None
        vae_factor = 1
        cross_reconstruction_factor =1
        distance_factor =0.01
       
        self.optimizer.zero_grad()
        loss = reconstruction_loss - vae_factor * KLD
        if cross_reconstruction_loss > 0:
            loss += cross_reconstruction_factor * cross_reconstruction_loss
        if distance > 0:
            loss += distance_factor * distance
        if ADV_loss:
            if ADV_loss > 0:
                loss += self.lamda * ADV_loss
        if self.beta:
            if CMFM_loss > 0:
                loss += self.beta * CMFM_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def train_vae(self): 
        losses = []
        self.dataset.novelclasses = self.dataset.novelclasses.long().cuda()
        self.dataset.seenclasses = self.dataset.seenclasses.long().cuda()
        self.train()
        self.reparameterize_with_noise = True
        for epoch in range(0, self.nepoch):
            self.current_epoch = epoch
            for key, value in self.encoder.items():
                self.encoder[key].train()
            for key, value in self.decoder.items():
                self.decoder[key].train()
            i = -1
            for iters in range(0, self.dataset.ntrain, self.batch_size):
                i += 1
                if len((self.model).strip()) > 0:
                    label, data_from_modalities = self.dataset.next_batch(self.batch_size)
                    label = label.long().to(self.device)
                    for j in range(len(data_from_modalities)):
                        data_from_modalities[j] = data_from_modalities[j].to(self.device)
                        data_from_modalities[j].requires_grad = False
                    loss_D = self.train_dis_step(data_from_modalities[0], data_from_modalities[1])
                    label, data_from_modalities = self.dataset.next_batch(self.batch_size)
                    label = label.long().to(self.device)
                    for j in range(len(data_from_modalities)):
                        data_from_modalities[j] = data_from_modalities[j].to(self.device)  #
                        data_from_modalities[j].requires_grad = False
                    loss_V = self.train_vae_step(data_from_modalities[0], data_from_modalities[1],
                                                 label)
                    loss = loss_D + loss_V
                else:
                    label, data_from_modalities = self.dataset.next_batch(self.batch_size)
                    label = label.long().to(self.device)
                    for j in range(len(data_from_modalities)):
                        data_from_modalities[j] = data_from_modalities[j].to(self.device)  # 把一个batch的数据送往cuda
                        data_from_modalities[j].requires_grad = False
                    loss = self.train_vae_step(data_from_modalities[0], data_from_modalities[1],
                                               label)
                if i % 300 == 0:
                    print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t' +
                          ' | loss ' + str(loss)[:5])
                if i % 300 == 0 and i > 0:
                    losses.append(loss)
   
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()
        return losses
    def train_classifier(self, show_plots=False):
        if self.num_shots > 0:
            self.dataset.transfer_features(self.num_shots, num_queries='num_features')
        history = [] 
        cls_seenclasses = self.dataset.seenclasses
        cls_novelclasses = self.dataset.novelclasses
        train_seen_feat = self.dataset.data['train_seen']['resnet_features']
        train_seen_label = self.dataset.data['train_seen']['labels']
        novelclass_aux_data = self.dataset.novelclass_aux_data
        seenclass_aux_data = self.dataset.seenclass_aux_data
        novel1_corresponding_labels = self.dataset.novelclasses.long().to(self.device)
        seen_corresponding_labels = self.dataset.seenclasses.long().to(self.device)
        novel_test_feat = self.dataset.data['test_unseen'][
            'resnet_features']
        seen_test_feat = self.dataset.data['test_seen'][
            'resnet_features']
        test_seen_label = self.dataset.data['test_seen']['labels']
        test_novel_label = self.dataset.data['test_unseen']['labels']
        train_unseen_feat = self.dataset.data['train_unseen']['resnet_features']
        train_unseen_label = self.dataset.data['train_unseen']['labels']

        if self.generalized == False:
            novel_corresponding_labels = self.map_label(novel1_corresponding_labels, novel1_corresponding_labels)
            if self.num_shots > 0:              
                train_unseen_label = self.map_label(train_unseen_label, cls_novelclasses)          
            test_novel_label = self.map_label(test_novel_label, cls_novelclasses)      
            cls_novelclasses = self.map_label(cls_novelclasses, cls_novelclasses)
        if self.generalized:       
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes)
        else:
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_novel_classes) 
        clf.apply(models.weights_init)
        with torch.no_grad():
            self.reparameterize_with_noise = False
            mu1, var1 = self.encoder['resnet_features'](novel_test_feat)
            test_novel_X = self.reparameterize(mu1, var1).to(self.device).data
            test_novel_Y = test_novel_label.to(self.device)
            mu2, var2 = self.encoder['resnet_features'](seen_test_feat)
            test_seen_X = self.reparameterize(mu2, var2).to(self.device).data
            test_seen_Y = test_seen_label.to(self.device)
            self.reparameterize_with_noise = True
            def sample_train_data_on_sample_per_class_basis(features, label, sample_per_class):
                sample_per_class = int(sample_per_class)
                if sample_per_class != 0 and len(label) != 0:
                    classes = label.unique()
                    for i, s in enumerate(classes):
                        features_of_that_class = features[label == s, :]
                        multiplier = torch.ceil(torch.cuda.FloatTensor(
                            [max(1, sample_per_class / features_of_that_class.size(0))])).long().item()
                        features_of_that_class = features_of_that_class.repeat(multiplier, 1)
                        if i == 0:
                            features_to_return = features_of_that_class[:sample_per_class, :]
                            labels_to_return = s.repeat(sample_per_class)
                        else:
                            features_to_return = torch.cat(
                                (features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                            labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)),
                                                         dim=0)
                    return features_to_return, labels_to_return
                else:
                    return torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])
            img_seen_feat, img_seen_label = sample_train_data_on_sample_per_class_basis(
                train_seen_feat, train_seen_label, self.img_seen_samples)
            img_unseen_feat, img_unseen_label = sample_train_data_on_sample_per_class_basis(
                train_unseen_feat, train_unseen_label, self.img_unseen_samples)
            att_unseen_feat, att_unseen_label = sample_train_data_on_sample_per_class_basis(
                novelclass_aux_data,
                novel_corresponding_labels, self.att_unseen_samples)
            def convert_datapoints_to_z(features, encoder):
                if features.size(0) != 0:
                    mu_, logvar_ = encoder(features)
                    z = self.reparameterize(mu_, logvar_)
                    return z
                else:
                    return torch.cuda.FloatTensor([])
            z_unse_att = convert_datapoints_to_z(novelclass_aux_data, self.encoder[self.auxiliary_data_source])
            att_seen_feat, att_seen_label = sample_train_data_on_sample_per_class_basis(
                seenclass_aux_data,
                seen_corresponding_labels, self.att_seen_samples)
            z_seen_img = convert_datapoints_to_z(img_seen_feat, self.encoder['resnet_features'])
            z_unseen_img = convert_datapoints_to_z(img_unseen_feat, self.encoder['resnet_features'])
            z_seen_att = convert_datapoints_to_z(att_seen_feat, self.encoder[self.auxiliary_data_source])
            z_unseen_att = convert_datapoints_to_z(att_unseen_feat, self.encoder[self.auxiliary_data_source])
            train_Z = [z_seen_img, z_unseen_img, z_seen_att, z_unseen_att]
            train_L = [img_seen_label, img_unseen_label, att_seen_label, att_unseen_label]
            train_X = [train_Z[i] for i in range(len(train_Z)) if train_Z[i].size(0) != 0]
            train_Y = [train_L[i] for i in range(len(train_L)) if train_Z[i].size(0) != 0]
            train_X = torch.cat(train_X, dim=0)
            train_Y = torch.cat(train_Y, dim=0)

        cls = classifier.CLASSIFIER(clf, train_X, train_Y, test_seen_X, test_seen_Y, test_novel_X,
                                    test_novel_Y,
                                    cls_seenclasses, cls_novelclasses,
                                    self.num_classes, self.device, self.lr_cls, 0.5, 1,
                                    self.classifier_batch_size,
                                    self.generalized)
        test_novel_X=test_novel_X.cpu()
        test_novel_Y=test_novel_Y.cpu()
        z_unseen_att =z_unse_att.cpu()
        novel_corresponding_labels = novel_corresponding_labels.cpu()
        tsne_plot.main(z_unseen_att, novel_corresponding_labels, test_novel_X, test_novel_Y)
        for k in range(self.cls_train_epochs):
            if k > 0:
                if self.generalized:
                    cls.acc_seen, cls.acc_novel, cls.H = cls.fit()
                else:
                    cls.acc = cls.fit_zsl()
            if self.generalized:
                print('[%.1f]     novel=%.4f, seen=%.4f, h=%.4f , loss=%.4f' % (
                    k, cls.acc_novel, cls.acc_seen, cls.H, cls.average_loss))
                history.append([torch.as_tensor(cls.acc_seen).item(), torch.as_tensor(cls.acc_novel).item(),
                                torch.as_tensor(cls.H).item()])
            else:
                print('[%.1f]  acc=%.4f ' % (k, cls.acc))
                history.append([0, torch.as_tensor(cls.acc).item(), 0])
        if self.generalized:
            return torch.as_tensor(cls.acc_seen).item(), torch.as_tensor(cls.acc_novel).item(), torch.as_tensor(
                cls.H).item(), history
        else:
            return 0, torch.as_tensor(cls.acc).item(), 0, history
