
import numpy as np
import pandas as pd
import random
import schedulefree
from tqdm import tqdm


import torch
from torch import optim, nn
from torch.nn import BatchNorm1d, LeakyReLU, Linear, Module
from torch.utils.data import DataLoader, IterableDataset

from torchsummary import summary

from data import RandomWalkDataset
# from base import BaseSynthesizer, random_state


class Generator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()        
        self.layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)  

    def forward(self, x):
        x = self.layer1(x)
        # x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.layer2(x)
        # x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.layer3(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.layer1(x))
        x = self.leaky_relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x


class GAN:

    def __init__(self, embedding_dim=128, hidden_dim=256, 
                 generator_lr=1e-4, generator_decay=0, discriminator_lr=1e-4,
                 discriminator_decay=0, batch_size=500, discriminator_steps=20,
                 log_frequency=True, verbose=False, epochs=300, pac=10, model_path="./model_gan.pt", device="cuda"):

        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None
        self.model_path = model_path

    def fit(self, train_data):


        train_data = train_data.to_numpy()
        
        self._feats_dim = train_data.shape[1]
            
        dataset = torch.from_numpy(train_data.astype(np.float32))   
        dataloader = DataLoader(RandomWalkDataset(dataset), batch_size=self._batch_size)
        dataiterator = iter(dataloader)
        
        criterion = nn.BCELoss()

        self._generator = Generator(self._embedding_dim, self._hidden_dim, self._feats_dim).to(self._device)
        
        print(self._generator)

        self._discriminator = Discriminator(self._feats_dim, self._hidden_dim).to(self._device)
        
        print(self._discriminator)

        optimizer_g = schedulefree.AdamWScheduleFree(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.1, 0.9), # was 0.5
            weight_decay=self._generator_decay
        )

        optimizer_d = schedulefree.AdamWScheduleFree(
            self._discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.1, 0.9), weight_decay=self._discriminator_decay # was 0.5
        )
        
        optimizer_g.train()
        optimizer_d.train()
  
        
        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        step_iterator = tqdm(range(self._epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            step_iterator.set_description(description.format(gen=0, dis=0))

        for step in step_iterator:


            labels_real = torch.ones((self._batch_size, 1)).to(self._device)
            labels_fake = torch.zeros((self._batch_size, 1)).to(self._device)

            real_batch = next(dataiterator)

            real_batch = real_batch.to(self._device)

            optimizer_d.zero_grad()
            output_real = self._discriminator(real_batch)

            noise = torch.randn((self._batch_size, self._embedding_dim)).to(self._device)
            generated_data = self._generator(noise)
            output_fake = self._discriminator(generated_data.detach())


            output_d = torch.cat((output_real, output_fake))
            labels_d = torch.cat((labels_real, labels_fake))

            loss_d = criterion(output_d, labels_d)
            loss_d.backward()        

            optimizer_d.step()

            optimizer_g.zero_grad()
            noise = torch.randn((self._batch_size, self._embedding_dim)).to(self._device)
            generated_data = self._generator(noise)
            output_g = self._discriminator(generated_data) # pas de detach pour retroprop dans G

            loss_g = criterion(output_g, labels_real)
            loss_g.backward()
            optimizer_g.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()


            if self._verbose:
                step_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )
        
        torch.save(self._generator.state_dict(), self.model_path)

    def sample(self, train_data, num_samples):
        
        self._feats_dim = train_data.shape[1]
        
        self.num_samples = num_samples
        
        self._generator = Generator(self._embedding_dim, self._hidden_dim, self._feats_dim).to(self._device)
        
        self._generator.load_state_dict(torch.load(self.model_path))
        
        self._generator.eval()
        
        return self._generator(torch.randn((self.num_samples, self._embedding_dim)).to(self._device))

