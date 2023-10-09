import numpy as np
import cv2
import os
import random
import matplotlib.colors as mcolors
import torch
import json
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


def get_square_crop(image):
    h,w = image.shape[:2]
    if h > w:
        return image[(h - w)//2:-(h - w)//2, :, :]
    else:
        return image[:, (w - h) // 2:-(w - h) // 2, :]


class MnistVisualLanguageDataset(Dataset):
    r"""
    Minimal visual language dataset class which auto generates fixed format caption
    for each dataset point
    """
    def __init__(self, split, config):
        self.split = split
        self.db_root = config['root_dir']
        self.im_size = config['image_size']
        
        # Probability of randomly dropping background info
        self.drop_background_info_prob = config['drop_background_prob']
        # Probability for dropping font color and background color
        self.drop_font_color_info_prob = config['drop_color_prob']
        
        # Auto generated caption formats
        self.generation_text_format_tokens = '<bos> generate image of {} in {} and a {} background of {} <sep>'
        self.generation_text_format_tokens_drop_bg = '<bos> generate image of {} in {} <pad> <pad> <pad> <pad> <pad> <pad> <sep>'
        self.generation_text_format_tokens_drop_color = '<bos> generate image of {} <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <sep>'
        
        # Validate right amount of padding and ensure all are same length
        assert (len(self.generation_text_format_tokens.split(' ')) == len(self.generation_text_format_tokens_drop_bg.split(' '))
                == len(self.generation_text_format_tokens_drop_color.split(' ')))
        self.max_token_len = len(self.generation_text_format_tokens.split(' '))
        self.visual_language_db = json.load(open(os.path.join(self.db_root, self.split + '.json')))
        
        self.vocab_idx_to_word, self.vocab_word_to_idx = self.build_vocab()
    
    def build_vocab(self):
        r"""
        Method to get dictionary of word to indexes and
        indexes to word to be used for tokenizing
        and for generation purposes
        :return:
        """
        vocab_generation_tokens = [word for word in self.generation_text_format_tokens.split(' ') if word != '{}']
        vocab_generation_tokens += [word for word in self.generation_text_format_tokens_drop_bg.split(' ') if word != '{}']
        vocab_generation_tokens += [word for word in self.generation_text_format_tokens_drop_color.split(' ') if word != '{}']
        vocab_preset = set(vocab_generation_tokens)
        for db_entry in self.visual_language_db:
            if 'texture_name' in db_entry:
                vocab_preset.add(db_entry['texture_name'])
                vocab_preset.add('texture')
            if 'background_color' in db_entry:
                vocab_preset.add(db_entry['background_color'])
                vocab_preset.add('solid')
            vocab_preset.add(db_entry['digit_name'])
            vocab_preset.add(db_entry['digit_color'])
        vocab_tokens = sorted(list(vocab_preset))
        vocab_word_to_idx = { k:v for (k,v) in zip(vocab_tokens, range(len(vocab_tokens)))}
        vocab_idx_to_word = { v:k for (k,v) in zip(vocab_tokens, range(len(vocab_tokens)))}
        return vocab_idx_to_word, vocab_word_to_idx
    
    def __len__(self):
        return len(self.visual_language_db)
    
    def __getitem__(self, index):
        entry = self.visual_language_db[index]
        background_type = 'solid' if 'background_color' in entry else 'texture'
        drop_type = random.choices(['no_drop','drop_bg','drop_bg_and_color'],
                                   weights=[1-self.drop_background_info_prob-self.drop_font_color_info_prob,
                                            self.drop_background_info_prob,
                                            self.drop_font_color_info_prob])[0]
        if drop_type == 'no_drop':
            text = self.generation_text_format_tokens.format(entry['digit_name'],
                                                      entry['digit_color'],
                                                      background_type,
                                                      entry['background_color'] if background_type == 'solid' else
                                                      entry['texture_name'])
        elif drop_type == 'drop_bg':
            text = self.generation_text_format_tokens_drop_bg.format(entry['digit_name'],
                                                      entry['digit_color'])
        else:
            text = self.generation_text_format_tokens_drop_color.format(entry['digit_name'])
        
        text_tokens = [self.vocab_word_to_idx[word] for word in text.split(' ')]
        text_tokens = torch.LongTensor(text_tokens)
        
        digit_im = cv2.imread(os.path.join(self.db_root, entry['digit_image']))
        digit_im = cv2.cvtColor(digit_im, cv2.COLOR_BGR2RGB)
        digit_im = cv2.resize(digit_im, (self.im_size, self.im_size))
        
        # Discretize mnist images to be either 0 or 1
        digit_im[digit_im > 50] = 255
        digit_im[digit_im <= 50] = 0
        mask_val = (digit_im > 0).astype(np.float32)
        color_scale = mcolors.hex2color('tab:{}'.format(entry['digit_color']))
        digit_im = np.concatenate((digit_im[:, :, 0][..., None] * color_scale[0],
                                   digit_im[:, :, 1][..., None] * color_scale[1],
                                   digit_im[:, :, 2][..., None] * color_scale[2]), axis=-1)
        if background_type == 'texture':
            im = cv2.imread(os.path.join(self.db_root, entry['texture_image']))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = get_square_crop(im)
            im = cv2.resize(im, (self.im_size, self.im_size))
        else:
            im = np.ones((self.im_size, self.im_size, 3))
            back_color_scale = mcolors.hex2color('tab:{}'.format(entry['background_color']))
            im[:, :, 0] = 255*back_color_scale[0]
            im[:, :, 1] = 255*back_color_scale[1]
            im[:, :, 2] = 255*back_color_scale[2]
        out_im = mask_val * digit_im + (1 - mask_val) * im
        im_tensor = torch.from_numpy(out_im).permute((2, 0, 1))
        im_tensor = 2 * (im_tensor / 255) - 1
        return {
            "image" : im_tensor,
            "text_tokens" : text_tokens,
            "text" : text,
        }

    
    
    