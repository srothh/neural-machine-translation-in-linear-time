import json

import requests
import torch
import pickle
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import T5ForConditionalGeneration, AutoTokenizer

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


class WMT19JSONLoader:
    def __init__(self, file_path, source_lang='de', target_lang='en', max_length=128):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.file_path = file_path
        self.tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    def load_json_data(self, file_path):
        """
        Function that loads the downloaded JSON file

        :param file_path:
        :return:
        """
        loaded_data = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    loaded_data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Error when line is decoded: {e}")
        return loaded_data

    def convert_to_tensor(self, src, trg):
        """
        Checks if source and target are tensor
        If both are not tensor, they are converted to tensors

        :param src:
        :param trg:
        :return:
        """
        if not torch.is_tensor(src):
            src = torch.Tensor(src)
        if not torch.is_tensor(trg):
            trg = torch.tensor(trg, dtype=torch.int32)
        return src, trg

    def extract_source_target(self, load_data):
        """
        Function that extracts out of the downloaded JSON the
        german rows as source and the english rows as targets

        :param load_data:
        :param source_lang:
        :param target_lang:
        :return:
        """
        source_texts = []
        target_texts = []
        for item in load_data:
            if ('row' in item and 'translation' in item['row'] and
                    self.source_lang in item['row']['translation'] and
                    self.target_lang in item['row']['translation']):
                source_texts.append(item['row']['translation'][self.source_lang])
                target_texts.append(item['row']['translation'][self.target_lang])
        return source_texts, target_texts

    def tokenize_texts(self, texts):
        """
        Function for tokenizing the text data
        Uses BERT-Tokenizer as tokenizer model

        :param texts:
        :return:
        """
        tokenized_texts = []
        for text in texts:
            tokens = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            tokenized_texts.append(tokens['input_ids'].squeeze())
        return tokenized_texts

    def load_and_tokenize(self, json_file_path):
        """
        Function that does the load json data
        and the tokenizing process

        :param json_file_path:
        """
        loaded_data = self.load_json_data(json_file_path)

        source_texts, target_texts = self.extract_source_target(loaded_data)

        # The tokenized source and targets
        # self.tokenizer is a object of type transformers from the Bert model
        # padding="max_length": is used to fill sequence to maximal length
        # truncation = True: Means that the sequence is cutted, if longer than max_length
        # return_tensors="pt": Means that a pytorch tensor is returned
        # the source text is tokenized into smaller elements
        tokenized_source_texts = self.tokenize_texts(source_texts)

        # the target text is tokenized into smaller elements
        tokenized_target_texts = self.tokenize_texts(target_texts)

        #TODO: evetually squeeze as in WMTLoader

        return tokenized_source_texts, tokenized_target_texts


def download_data(offset, length):
    """
    Method for downloading the dataset as JSON
    F.e. if the first 10 rows have to be downloaded, offset has to
    be 0 and length has to be 10

    :param offset: The offset used in the url
    :param length: The length of the selected number of rows in the dataset
    :return:
    """
    url = f"https://datasets-server.huggingface.co/rows?dataset=wmt%2Fwmt19&config=de-en&split=train&offset={offset}&length={length}"
    query_parameters = {"downloadformat": "json"}
    response = requests.get(url, params=query_parameters)
    if response.status_code == 200:
        loaded_data = response.json()
        print(f"Downloading dataset-offset: {offset}")
        return loaded_data['rows']
    else:
        print(f"Error while downloading data: {response.status_code}")
        return []


def save_data_to_json(load_data, file_path):
    """
    Writes data into the JSON object

    :param load_data: The data that has to be writen into file
    :param file_path: The file path where the file has to be saved
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        for item in load_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def download_batch_and_save(offset, length, output_file):
    """
    Downloads and saves the batch

    :param offset: The offset which is currently used to download
    :param length: The length is defined with 100
    :param output_file: The name of the file to be saved
    """
    loaded_data = download_data(offset, length)
    save_data_to_json(loaded_data, output_file)


def download_entire_de_en_dataset(batch_size, output_dir, num_workers):
    """
    Downloads the entire WMT19 dataset. Uses a ThreadPoolExecutor for
    faster download of the dataset.

    :param batch_size:
    :param output_dir:
    :param num_workers:
    """
    offset = 0
    output_file = os.path.join(output_dir, 'wmt_19_de_en.json')
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        while True:
            futures.append(executor.submit(download_batch_and_save, offset, batch_size, output_file))
            offset += batch_size
            # if offset >= 34800000:
            if offset >= 348:
                break

        for future in as_completed(futures):
            future.result()


if __name__ == '__main__':
    # use drive in which to save dataset in cache
    cache_dir = 'F:/wmt19_cache'
    # wmt_loader = WMTLoader(split="train", cache_dir=cache_dir)
    # Number of workers provides parallel loading
    # num_workers = 4
    # data_load = DataLoader(wmt_loader, batch_size=32, collate_fn=wmt_loader.collate_fn, num_workers=num_workers)
    # temp = data_load
    #
    # for batch in wmt_loader:
    #     src_batch, tgt_batch = batch
    #     break

    batch_size = 100
    output_dir = 'F:\\wmt19_json'

    # download_entire_de_en_dataset(batch_size, output_dir, 4)

    wmt_json_loader = WMT19JSONLoader(output_dir)
    tokenized_source_texts, tokenized_target_texts = wmt_json_loader.load_and_tokenize('D:\\wmt19_json\\wmt_19_de_en.json')
    src = tokenized_source_texts
    trgt = tokenized_target_texts