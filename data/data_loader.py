import json

import requests
import torch
import pickle
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import os

from torch import FloatTensor, LongTensor


class WMTLoader(data.Dataset):
    """`WMT dataset loader

        This class enables the loading of the WMT german-english translation
        WMTLoader class uses BertTokenizer as tokenizer model, and it is a sub-word-tokenizer,
        which means it splits unknown word into smaller words. For example, 'words' is split
        into the pieces 'word' and 's'.

        Args:
            split="train" The train part of 34,8 Million rows
            src_lang="de" German as the source language
            tft_lang="en" English as the target language
            max_length=128
            cache_dir='./wmt19_cache' The cache dir where the downloaded dataset is cached
        """
    def __init__(self, split="train", src_lang="de", tgt_lang="en", max_length=128, cache_dir='./wmt19_cache'):
        self.dataset = load_dataset("wmt19", "de-en", split=split, cache_dir=cache_dir)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        # Cache dir for faster access to WMT-dataset
        self.cache_dir = cache_dir

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

        self.data = self.download_data(0, 100)

    def download_data(self, offset, length):
        """
        Method for downloading the dataset as JSON
        F.e. if the first 10 rows have to be downloaded, offset has to
        be 0 and length has to be 10

        :param offset: The offset used in the url
        :param length: The length of the selected number of rows in the dataset
        :return:
        """
        url = f"https://datasets-server.huggingface.co/rows?dataset=wmt%2Fwmt19&config=de-en&split=train&offset={offset}&length={length}"
        response = requests.get(url)
        if response.status_code == 200:
            loaded_data = response.json()
            return loaded_data['rows']
        else:
            print(f"Error while downloading data: {response.status_code}")
            return []

    def save_data_to_json(self, data, file_path):
        """
        Writes data into the JSON object

        :param data: The data that has to be writen into file
        :param file_path: The file path where the file has to be saved
        """
        with open(file_path, 'a', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

    def download_entire_de_en_dataset(self):
        # TODO: further implementation of this method
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Function for receiving a source and target element and to tokenize them.
        Both the source and the target text are going to be tokenized, which means they are
        split into several smaller pieces, named tokens and then converted into numerical
        numbers, called token-ids

        :param index:
        :return:
        """
        item = self.dataset[index]
        src_text = item[self.src_lang]
        tgt_text = item[self.tgt_lang]

        # self.tokenizer is a object of type transformers from the Bert model
        # padding="max_length": is used to fill sequence to maximal length
        # truncation = True: Means that the sequence is cutted, if longer than max_length
        # return_tensors="pt": Means that a pytorch tensor is returned
        # the source text is tokenized into smaller elements
        src_tokens = self.tokenizer(src_text, max_length=self.max_length, padding="max_length", truncation=True,
                                  return_tensors="pt")
        # the target text is tokenized into smaller elements
        tgt_tokens = self.tokenizer(tgt_text, max_length=self.max_length, padding="max_length", truncation=True,
                                    return_tensors="pt")
        # Now both the source and the target tokens are converted into numerical data, called token-ids
        # Squeeze is used to remove dimensions with value 1
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
        The results are stacked tensors for the
        source and target batches which are feed
        into the neural network

        :param batch:
        :return:
        """
        src_batch, tgt_batch = zip(*batch)
        src_batch = torch.stack(src_batch)
        tgt_batch = torch.stack(tgt_batch)
        return src_batch, tgt_batch


if __name__ == '__main__':
    # use drive in which to save dataset in cache
    cache_dir = 'D:/wmt19_cache'
    wmt_loader = WMTLoader(split="train", cache_dir=cache_dir)
    # Number of workers provides parallel loading
    num_workers = 4
    data_load = DataLoader(wmt_loader, batch_size=32, collate_fn=wmt_loader.collate_fn, num_workers=num_workers)
    temp = data_load

    for batch in wmt_loader:
        src_batch, tgt_batch = batch
        break




