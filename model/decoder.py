import torch
import torch.nn as nn


class Decoder(nn.Module):
    r"""
    Decoder with couple of residual blocks
    followed by conv transpose relu layers
    """
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()
        
        self.decoder_layers = nn.ModuleList([
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Tanh()
        ])
        
        self.residuals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU())
        ])
        
        self.decoder_quant_conv = nn.Conv2d(embedding_dim, 64, 1)
        
    
    def forward(self, x):
        out = self.decoder_quant_conv(x)
        for layer in self.residuals:
            out = layer(out)+out
        for idx, layer in enumerate(self.decoder_layers):
            out = layer(out)
        return out
        


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import yaml
    decoder = Decoder()
    
    out = decoder(torch.rand((3, 64, 14, 14)))
    print(out.shape)
    


