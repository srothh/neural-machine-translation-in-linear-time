import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import os

from torch import FloatTensor, LongTensor


class WMTLoader(data.Dataset):
    """`WMT dataset loader

        This class enables the loading of the WMT german-english translation

        Args:
            split="train" The train part of 34,8 Million rows
            src_lang="de" German as the source language
            tft_lang="en" English as the target language
            max_length=128
            cache_dir='./wmt19_cache' The cache dir where the downloaded dataset is cached
        """
    def __init__(self, split="train", src_lang="de", tgt_lang="en", max_length=128, cache_dir='./wmt19_cache'):
        self.dataset = load_dataset("wmt19", "de-en", split=split, cache_dir=cache_dir)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        # Cache dir for faster access to WMT-dataset
        self.cache_dir = cache_dir

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        src_text = item[self.src_lang]
        tgt_text = item[self.tgt_lang]

        src_tokens = self.tokenizer(src_text, max_length=self.max_length, padding="max_length", truncation=True,
                                    return_tensors="pt")
        tgt_tokens = self.tokenizer(tgt_text, max_length=self.max_length, padding="max_length", truncation=True,
                                    return_tensors="pt")

        src_ids = src_tokens["input_ids"].squeeze()
        tgt_ids = tgt_tokens["input_ids"].squeeze()

        return src_ids, tgt_ids

    def convert_to_tensor(self, src, trg):
        """
        Checks if source and target are tensor
        If both are not tensor, they are converted to tensors

        :param src:
        :param trg:
        :return:
        """
        if not torch.is_tensor(src):
            src = FloatTensor([src])
        if not torch.is_tensor(trg):
            trg = LongTensor([trg])
        return src, trg

    def load_data(self):
        pass

    def collate_fn(sel, batch):
        """
        Function needed to combine single
        examples together by stacking Tensors
        :param batch:
        :return:
        """
        src_batch, tgt_batch = zip(*batch)
        src_batch = torch.stack(src_batch)
        tgt_batch = torch.stack(tgt_batch)
        return src_batch, tgt_batch

if __name__ == '__main__':
    cache_dir = './wmt19_cache'
    wmt_loader = WMTLoader(split="train", cache_dir=cache_dir)
    # Number of workers provides parallel loading
    num_workers = 4
    data_load = DataLoader(wmt_loader, batch_size=32, collate_fn=wmt_loader.collate_fn, num_workers=num_workers)
    temp = data_load




