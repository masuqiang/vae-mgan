import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.5)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class encoder_template(nn.Module):

    def __init__(self, input_dim, latent_size, hidden_size_rule, device):
        super(encoder_template, self).__init__()

        if len(hidden_size_rule) == 2:
            self.layer_sizes = [input_dim, hidden_size_rule[0], latent_size]
        elif len(hidden_size_rule) == 3:
            self.layer_sizes = [input_dim, hidden_size_rule[0], hidden_size_rule[1], latent_size]

        modules = []
        for i in range(len(self.layer_sizes) - 2):
            modules.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            modules.append(nn.ReLU())
        self.feature_encoder = nn.Sequential(*modules)
        self._mu = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
        self._logvar = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
        self.apply(weights_init)
        self.to(device)

    def forward(self, x):
        h = self.feature_encoder(x)
        mu = self._mu(h)
        logvar = self._logvar(h)
        return mu, logvar


class decoder_template(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size_rule, device):
        super(decoder_template, self).__init__()
        self.layer_sizes = [input_dim, hidden_size_rule[-1], output_dim]
        self.feature_decoder = nn.Sequential(nn.Linear(input_dim, self.layer_sizes[1]), nn.ReLU(),
                                             nn.Linear(self.layer_sizes[1], output_dim))
        self.apply(weights_init)
        self.to(device)

    def forward(self, x):
        return self.feature_decoder(x)


class discriminator(nn.Module):
    def __init__(self, x_dim=2048, s_dim=300, layers='1200'):
        super(discriminator, self).__init__()
        self.x_dim = x_dim
        self.s_dim = s_dim
        total_dim = x_dim + s_dim
        layers = layers.split()
        fcn_layers = []
        for i in range(len(layers)):
            pre_hidden = int(layers[i - 1])
            num_hidden = int(layers[i])
            if i == 0:
                fcn_layers.append(nn.Linear(total_dim, num_hidden))
                fcn_layers.append(nn.ReLU())
            else:
                fcn_layers.append(nn.Linear(pre_hidden, num_hidden))
                fcn_layers.append(nn.ReLU())
            if i == len(layers) - 1:
                fcn_layers.append(nn.Linear(num_hidden, 1))
                fcn_layers.append(nn.Sigmoid())
        self.FCN = nn.Sequential(*fcn_layers)
        self.mse_loss = nn.MSELoss()

    def forward(self, X, S):
        XS = torch.cat([X, S], 1)
        return self.FCN(XS)

    def dis_loss(self, X, Xp, S, Sp):
        true_scores = self.forward(X, S)
        fake_scores = self.forward(Xp, S)
        ctrl_socres = self.forward(X, Sp)
        return self.mse_loss(true_scores, 1) + self.mse_loss(fake_scores, 0) + self.mse_loss(ctrl_socres, 0)


class discriminator_xs(nn.Module):
    def __init__(self, x_dim=2048, layers='1200'):
        super(discriminator_xs, self).__init__()
        self.x_dim = x_dim
        layers = layers.split()
        fcn_layers = []
        for i in range(len(layers)):
            pre_hidden = int(layers[i - 1])
            num_hidden = int(layers[i])
            if i == 0:
                fcn_layers.append(nn.Linear(self.x_dim, num_hidden))
                fcn_layers.append(nn.ReLU())
            else:
                fcn_layers.append(nn.Linear(pre_hidden, num_hidden))
                fcn_layers.append(nn.ReLU())
            if i == len(layers) - 1:
                fcn_layers.append(nn.Linear(num_hidden, 1))
                fcn_layers.append(nn.Sigmoid())
        self.FCN = nn.Sequential(*fcn_layers)
        self.mse_loss = nn.MSELoss()

    def forward(self, X):
        return self.FCN(X)



def dot_loss(v, s, l):
    n, d = v.shape
    s = s.T
    v = torch.mm(v, s)
    l = l.repeat(n, 1)
    l1 = l
    l = l.transpose(0, 1)
    l = l - l1
    l[l != 0] = -1
    l[l == 0] = 1
    v = v * l
    x1 = v[v > 0].mean()
    x2 = v[v < 0].mean()
    return x1 + x2

def mse_loss(zv, zs, l):
    n, d = zv.shape
    zv = zv.repeat(n, 1, 1)
    zv = zv.transpose(0, 1)
    zs = zs.repeat(n, 1, 1)
    zv = zv - zs
    zv = torch.pow(zv, 2)
    zv = zv.sum(dim=2)
    zv = zv.sqrt()
    l = l.repeat(n, 1)
    l1 = l
    l = l.transpose(0, 1)
    l = l - l1
    l[l != 0] = -1
    l[l == 0] = 1
    zv = zv * l
    x1 = zv[zv > 0].mean()
    x2 = zv[zv < 0].mean()
    return x1 + x2


def cos_loss(zv, zs, l):
    n, d = zv.shape
    zv = zv.repeat(n, 1, 1)
    zv = zv.transpose(0, 1)
    zs = zs.repeat(n, 1, 1)
    zv = F.cosine_similarity(zv, zs, dim=2)
    w = torch.ones(n, n).cuda()
    zv = w - zv
    l = l.repeat(n, 1)
    l1 = l
    l = l.transpose(0, 1)
    l = l - l1
    l[l != 0] = -1
    l[l == 0] = 1
    zv = zv * l
    x1 = zv[zv > 0].mean()
    x2 = zv[zv < 0].mean()
    return x1 + x2


def mcdd_loss(U):
    n, d = U.shape
    P = torch.eye(n).cuda()
    W = torch.ones(n, n).cuda()
    H = n * P - W
    H = H / n
    UT = U.transpose(0, 1)
    UT = torch.mm(UT, H)
    UT = torch.mm(UT, U)
    I = torch.eye(d).cuda()
    UT = UT - I
    UT = torch.norm(UT)
    UT = torch.pow(UT, 2)
    return UT
