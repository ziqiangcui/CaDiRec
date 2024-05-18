from tqdm import tqdm
from collections import defaultdict
import time
from utils import generate_rating_matrix_valid, generate_rating_matrix_test, neg_sample
from torch.utils.data import Dataset, DataLoader
import random
import copy 
import torch
from torch.utils.data import RandomSampler, SequentialSampler
import numpy as np

class DataGenerator(object):

    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.bs = args.train_batch_size
        self.data_file = args.data_path + args.dataset + ".txt"
        print("========dataset :===========", self.data_file)
        self.create_dataset()
       
    def get_user_seqs(self, data_file):
        lines = open(data_file).readlines()
        user_seq = []
        item_set = set()
        for line in lines:
            user, items = line.strip().split(" ", 1)
            items = items.split(" ")
            items = [int(item) for item in items]
            if len(items) >= 5:
                user_seq.append(items)
            item_set = item_set | set(items)
        max_item = max(item_set)
        num_users = len(lines)
        num_items = max_item + 2

        valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
        test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    
        return user_seq, max_item, valid_rating_matrix, test_rating_matrix
  
    
    def create_dataset(self):
        '''Load train, validation, test dataset'''
        self.user_seq, self.max_item, self.valid_rating_matrix, self.test_rating_matrix = self.get_user_seqs(self.data_file)
        self.item_size = self.max_item + 2
        self.mask_id = self.max_item + 1
        self.args.item_size = self.item_size
        
        self.train_dataset = SASRecDataset(self.args, self.user_seq, data_type="train")
        self.valid_dataset = SASRecDataset(self.args, self.user_seq, data_type="valid")
        self.test_dataset = SASRecDataset(self.args, self.user_seq, data_type="test")
       
        self.train_dataloader = self.make_train_dataloader()
        self.valid_dataloader = self.make_valid_dataloader()
        self.test_dataloader = self.make_test_dataloader()
        
        
        
    def make_train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset,
                                        sampler=RandomSampler(self.train_dataset),
                                        batch_size=self.args.train_batch_size,
                                        drop_last=False,
                                        num_workers=4
                                        ) 
     
        return train_dataloader
    
    
    def make_valid_dataloader(self):
        valid_dataloader = DataLoader(self.valid_dataset,
                                        sampler=SequentialSampler(self.valid_dataset),
                                        batch_size=self.args.test_batch_size,
                                        drop_last=True) 
        return valid_dataloader
        
       
    def make_test_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset,
                                        sampler=SequentialSampler(self.test_dataset),
                                        batch_size=self.args.test_batch_size,
                                        drop_last=True) 
        return test_dataloader
    
    
class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length
        
    def mask_input_ids(self, input_ids, p):
        mask_indices = []
        for token in input_ids:
            if token == 0:
                mask_indices.append(0)  # 元素为0的位置不参与mask
            else:
                mask_indices.append(1 if random.random() < p else 0)
        return mask_indices

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]
        attention_mask = [1.0 if token > 0 else 0.0 for token in input_ids]
        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len
        
        masked_indices0 = self.mask_input_ids(input_ids, self.args.mlm_probability_train) # random mask some items and return the indices. for diffusion train
        # masked_indices1 = self.mask_input_ids(input_ids, self.args.mlm_probability) # random mask some items and return the indices.  for sas aug
        # masked_indices2 = self.mask_input_ids(input_ids, self.args.mlm_probability) # random mask some items and return the indices.  for sas aug
        

        dict1 = {"user_id": torch.tensor(user_id, dtype=torch.long),
                "input_ids":torch.tensor(input_ids, dtype=torch.long),
                "target_pos":torch.tensor(target_pos, dtype=torch.long), 
                "target_neg":torch.tensor(target_neg, dtype=torch.long), 
                "answer":torch.tensor(answer, dtype=torch.long), 
                "masked_indices0": torch.tensor(masked_indices0, dtype=torch.bool), 
                # "masked_indices1": torch.tensor(masked_indices1, dtype=torch.bool),  
                # "masked_indices2": torch.tensor(masked_indices2, dtype=torch.bool), 
                "attention_mask":torch.tensor(attention_mask, dtype=torch.float), 
             }
        return dict1
        

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = 0  # no use

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)

    def __len__(self):
        return len(self.user_seq)
    
    
    

       
    