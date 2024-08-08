import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import schedulefree

import torch
from torch import optim
from torch.nn import BatchNorm1d, ReLU, Linear, Module, Sequential, functional, Parameter
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from data_transformer import DataTransformer
from data import RandomWalkDataset


class Encoder(Module):

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):

        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):

        return self.seq(input_), self.sigma


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]

class TVAE:

    def __init__(self, embedding_dim=128, compress_dims=(256, 256), decompress_dims=(256, 256),
                 l2scale=1e-5, loss_factor=2, batch_size=500, lr=0.0001,
                 log_frequency=False, verbose=False, epochs=300, model_path="./model_tvae.pt", device="cuda"): # log_frequency was true

        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.lr = lr
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self._verbose = verbose
        self._epochs = epochs

        self._device = torch.device(device)

        
        self._log_frequency = log_frequency
        self._transformer = None
        self._data_sampler = None
        self._generator = None
        
        self.model_path = model_path


    def fit(self, train_data, discrete_columns=()):


        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        dataset = torch.from_numpy(train_data.astype(np.float32))   
        dataloader = DataLoader(RandomWalkDataset(dataset), batch_size=self.batch_size)
        dataiterator = iter(dataloader)

        data_dim = self._transformer.output_dimensions

        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        
        self.optimizer = schedulefree.AdamWScheduleFree(list(encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale, lr = self.lr)
        self.optimizer.train()
        

        step_iterator = tqdm(range(self._epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Deco. ({dec:.2f})'
            step_iterator.set_description(description.format(dec=0))


        for step in step_iterator:
            
            real_batch = next(dataiterator)
            self.optimizer.zero_grad()
            real = real_batch.to(self._device)
            mu, std, logvar = encoder(real)
            eps = torch.randn_like(std)
            emb = eps * std + mu
            rec, sigmas = self.decoder(emb)
            loss_1, loss_2 = _loss_function(
                rec, real, sigmas, mu, logvar,
                self._transformer.output_info_list, self.loss_factor)
            loss = loss_1 + loss_2
            loss.backward()
            self.optimizer.step()
            self.decoder.sigma.data.clamp_(0.01, 1.0)

            if self._verbose and step % 10 == 0:
                loss = loss.detach().cpu().item()
                step_iterator.set_description( description.format(dec = loss) )
                
        torch.save(self.decoder.state_dict(), self.model_path)

    def sample(self, train_data, samples, discrete_columns=() ):

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)
        data_dim = self._transformer.output_dimensions
        
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
                    
        self.decoder.load_state_dict(torch.load(self.model_path))
        
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]

        return self._transformer.inverse_transform(data)


