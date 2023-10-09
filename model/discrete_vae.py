import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from model.quantizer import Quantizer

class DiscreteVAE(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=512):
        super(DiscreteVAE, self).__init__()
        self.encoder = Encoder(num_embeddings=num_embeddings)
        self.quantizer = Quantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.decoder = Decoder(embedding_dim=embedding_dim)
    
    
    def get_codebook_indices(self, x):
        # x.shape = B,C,H,W
        enc_logits = self.encoder(x)
        # enc_logits.shape = B,C,H,W
        indices = torch.argmax(enc_logits, dim=1)
        return indices
    
    def decode_from_codebook_indices(self, indices):
        quantized_indices = self.quantizer.quantize_indices(indices)
        return self.decoder(quantized_indices)
        
    def forward(self, x):
        enc = self.encoder(x)
        quant_output, kl, logits, log_qy = self.quantizer(enc)
        out = self.decoder(quant_output)
        return out, kl, log_qy

    


