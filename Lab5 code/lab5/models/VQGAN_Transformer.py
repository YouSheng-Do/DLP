import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        _, z_indices, _ = self.vqgan.encode(x)
        return z_indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            def linear(ratio):
                return 1 - ratio
            return linear
        elif mode == "cosine":
            def cosine(ratio):
                return (1 + math.cos(math.pi * ratio)) / 2
            return cosine
        elif mode == "square":
            def square(ratio):
                return 1 - ratio ** 2
            return square
        else:
            raise NotImplementedError
    
##TODO2 step1-3:            
    def forward(self, x):
        z_indices = self.encode_to_z(x).view(x.size(0), -1) #ground truth
        
        ratio = np.random.uniform(0, 1)
        mask_ratio = self.gamma(ratio)
        batch_size, num_tokens = z_indices.size()
        mask = torch.rand(batch_size, num_tokens) < mask_ratio
        masked_z_indices = z_indices.clone()
        masked_z_indices[mask] = self.mask_token_id
        
        logits = self.transformer(masked_z_indices) #transformer predict the probability of tokens
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask_bc, ratio):
        masked_z_indices = z_indices.clone()
        masked_z_indices[mask_bc] = self.mask_token_id
        logits = self.transformer(masked_z_indices)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = torch.softmax(logits, dim=-1)

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(logits, dim=-1)
        z_indices_predict_prob = torch.where(mask_bc, z_indices_predict_prob, torch.tensor(float("Inf"), device=z_indices_predict_prob.device))

        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = torch.distributions.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(z_indices_predict_prob.device)  # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens

        # z_indices_predict[~mask_bc] = z_indices[~mask_bc]
        z_indices_predict = torch.where(mask_bc, z_indices_predict, z_indices)
        
        sorted_conf, sorted_indices = torch.sort(confidence)

        mask_ratio = self.gamma(ratio)
        mask_count = int(mask_ratio * torch.sum(mask_bc).item())

        mask_bc[0][sorted_indices[0][mask_count:]] = False
        new_mask = mask_bc
        
        return z_indices_predict, new_mask
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
