import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
import random
from tqdm import tqdm
from tqdm.auto import tqdm
tqdm._instances.clear()

import schedulefree

from data_transformer import DataTransformer

from positional_embeddings import PositionalEmbedding

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, num_features = 8, hidden_size = 128, hidden_layers = 3, emb_size = 128,
                 time_emb = "sinusoidal", input_emb = "sinusoidal", scale = 10.0):
        super().__init__()
        
        self.scale = scale
        
        self.time_mlp = PositionalEmbedding(emb_size, time_emb, scale)
        self.input_mlps = nn.ModuleList([PositionalEmbedding(emb_size, input_emb, scale) for _ in range(num_features)])

        concat_size = len(self.time_mlp.layer) + sum(len(mlp.layer) for mlp in self.input_mlps)      

        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, num_features))
        self.joint_mlp = nn.Sequential(*layers)


    def forward(self, x, t):
        x_embs = [self.input_mlps[i](x[:, i]) for i in range(x.shape[1])]
        t_emb = self.time_mlp(t)
        x = torch.cat(x_embs + [t_emb], dim=-1)
        x = self.joint_mlp(x)
        return x


class NoiseScheduler():
    def __init__(self,
                 device,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02, # was 0.02
                 beta_schedule="linear"):
        
        self.device = device

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t].to(self.device)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].to(self.device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t].to(self.device)
        s2 = self.posterior_mean_coef2[t].to(self.device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps].to(self.device)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].to(self.device)

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps
    
class RandomWalkDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.size =  len(dataset)
    def generator(self):
        random.seed(0)
        while True:
            yield self.dataset[random.randint(0,self.size-1)]
    def __iter__(self):
        return self.generator()
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
class TINY():

    def __init__(self, epochs=5000, batch_size = 1024, num_timesteps = 2000, lr = 0.0001, weight_decay = 0,
                 verbose = True , hidden_size = 1024, hidden_layers = 3, embedding_size = 64, 
                 time_embedding = "sinusoidal" , input_embedding = "sinusoidal" , scale = 2.0, 
                 beta_schedule="linear", model_path="./model_tiny.pt", device="cuda"):

        self.steps=epochs
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose 
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.embedding_size = embedding_size
        self.time_embedding = time_embedding
        self.input_embedding = input_embedding
        self.scale = scale
        self.beta_schedule = beta_schedule
        self.device = torch.device(device)
        
        self.model_path = model_path
        
        
    def fit(self, train_data):
        
        train_data = train_data.values
        dataset = torch.from_numpy(train_data.astype(np.float32))   
        dataloader = DataLoader(RandomWalkDataset(dataset), batch_size=self.batch_size)
        dataiterator = iter(dataloader)
        
        self.mlp = MLP(num_features=train_data.shape[1],
                    hidden_size=self.hidden_size,
                    hidden_layers=self.hidden_layers,
                    emb_size=self.embedding_size,
                    time_emb=self.time_embedding,
                    input_emb=self.input_embedding,
                    scale = self.scale).to(self.device)

        self.mlp.train()
        
        noise_scheduler = NoiseScheduler(device = self.device, num_timesteps = self.num_timesteps,
                                         beta_schedule = self.beta_schedule)


        # optimizer = torch.optim.AdamW(self.mlp.parameters(), lr = self.lr)
        self.optimizer = schedulefree.AdamWScheduleFree(self.mlp.parameters(), lr=self.lr)
        self.optimizer.train()
        
        step_size = 4000
        gamma = 0.5
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
        #                                     T_0 = 4000,# Number of iterations for the first restart
        #                                     T_mult = 1, # A factor increases TiTi after a restart
        #                                     eta_min = 1e-6) # Minimum learning rate
                    
        step_iterator = tqdm(range(self.steps), disable=(not self.verbose))
        
        if self.verbose:
            description = 'mloss ({mloss:.2f})'
            step_iterator.set_description(description.format(mloss=0))
            
        for step in step_iterator:

            batch = next(dataiterator)
            batch = batch.to(self.device)
            noise = torch.randn(batch.shape).to(self.device)
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (batch.shape[0],)).long()
            
            self.optimizer.zero_grad()

            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            noise_pred = self.mlp(noisy, timesteps.to(self.device))
            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)

            nn.utils.clip_grad_norm_(self.mlp.parameters(), 1.0)
            self.optimizer.step()
            

            # scheduler.step()
            


            if self.verbose:
                mloss = loss.detach().cpu().item()
                step_iterator.set_description(
                    description.format(mloss=mloss)
                )
        
        torch.save(self.mlp.state_dict(), self.model_path)
        

    def sample(self, train_data, num_samples):
        

        num_feat = train_data.shape[1]
        self.num_samples = num_samples
        
        noise_scheduler = NoiseScheduler(device = self.device, num_timesteps = self.num_timesteps,
                                         beta_schedule = self.beta_schedule)
        
        self.mlp = MLP(num_features=num_feat,
                    hidden_size=self.hidden_size,
                    hidden_layers=self.hidden_layers,
                    emb_size=self.embedding_size,
                    time_emb=self.time_embedding,
                    input_emb=self.input_embedding,
                    scale = self.scale).to(self.device)
  
        self.mlp.load_state_dict(torch.load(self.model_path))
        
        self.mlp.eval()
        
        

        sample = torch.randn(self.num_samples, num_feat).to(self.device)
        timesteps = list(range(len(noise_scheduler)))[::-1]
        for i, t in enumerate(timesteps):
            t = torch.from_numpy(np.repeat(t, self.num_samples)).long().to(self.device)
            with torch.no_grad():
                residual = self.mlp(sample, t)        
            sample = noise_scheduler.step(residual, t[0], sample)     
            
        return sample.cpu().numpy().astype(np.float64)