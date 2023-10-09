import torch
import torch.nn as nn
from model.mingpt import GPT, DallEGPTConfig

class DallE(nn.Module):
    r"""
    Class handling the logic for DallE
    Calls the vae and passes the text and image tokens
    together with target to gpt
    """
    def __init__(self, vae, num_words, image_size, max_text_len, image_vocab_size, gpt_config):
        super(DallE, self).__init__()
        self.vae = vae
        
        # Text Vocab size
        self.num_words = num_words
        # Number of Image tokens
        self.image_size = image_size
        # Maximum Text Sequence Length
        self.max_text_len = max_text_len
        
        # Image tokens vocabulary size (num_of_embeddings)
        image_vocab_size = image_vocab_size
        
        # Length of largest sequence so that we tell gpt
        # to have that as the context size
        max_sequence_len = max_text_len + image_size*image_size
        config = DallEGPTConfig(text_vocab_size=num_words,
                           image_vocab_size=image_vocab_size,
                           max_sequence_len=max_sequence_len,
                           im_size=image_size,
                           **gpt_config)
        self.gpt = GPT(config)
    
    def forward(self, im, text):
        # Call Discrete vae
        image_tokens = self.vae.get_codebook_indices(im).reshape(im.size(0), -1)
        
        # Shift the target image tokens as image tokens + text vocab size
        # Last fc layer will predict 0 to (num_words + num_embeddings) output probabilities
        # We will formulate the target such first num_words-1 are text token probabilities
        # and num_words to num_words+num_embeddings are image token probabilities
        target_image_tokens = image_tokens + self.num_words
        labels = None
        
        if self.training:
            # Pass one position shifted tokens as targets only in training
            labels = torch.cat((text[:, 1:], target_image_tokens), dim=1)
        # Loss of text and Loss image separately so that we can get better images
        logits, loss_text, loss_image = self.gpt(image_tokens, text, targets=labels)
        return logits, loss_text, loss_image
        
