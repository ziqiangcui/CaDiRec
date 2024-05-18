from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
import torch
import numpy as np
from scipy.sparse import csr_matrix
import random
import copy 

def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_user_seqs(data_file):
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


def get_interaction(fname, filter_num, aug=True):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('datasets/%s.txt' % fname, 'r')
   
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback >= filter_num:  
            # user_train[user] = User[user]
            # user_valid[user] = []
            # user_test[user] = []
       
            user_train[user] = User[user][:-2]
            
            user_valid[user] = User[user][:-1]
          
            user_test[user] = User[user][:]
           
            
    return [user_train, user_valid, user_test, usernum, itemnum]


def unzip_data(data, aug=True, min_len=3):

    res = []
    
    if aug:
        for user in data.keys():
            user_seq = data[user]
            seq_len = len(user_seq)
            if seq_len > min_len:
                for i in range(min_len, seq_len+1):
                    res.append(user_seq[:i])
    else:
        for user in data.keys():
            user_seq = data[user]
            seq_len = len(user_seq)
            if seq_len > min_len:
                res.append(user_seq)
    return res


def filter_data(data, thershold=5):
    '''Filter out the sequence shorter than threshold'''
    res = []

    for user in data:

        if len(user) > thershold:
            res.append(user)
        else:
            continue
    
    return res


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def neg_sample(item_set, item_size): 
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class SASTrainDataset(Dataset):

    def __init__(self, data, item_num, max_len, neg_sample=False) -> None:

        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_sample = neg_sample
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        labels = self.data[index][-1]
        seq = self.data[index][:-1]
        
        neg = random_neq(1, self.item_num + 1, set(self.data[index]))
        
        seq = seq[-self.max_len:]
        seq_len = len(seq)  
        pad_len = self.max_len - seq_len  
        
        padded_seq = seq + [0] * pad_len 
        attention_mask = [1.0] * seq_len + [0.0] * pad_len 
        attention_mask = torch.tensor(np.array(attention_mask), dtype=torch.float32)
        # neg100 = torch.tensor(np.array(neg100), dtype=torch.long)
        
        padded_seq = torch.tensor(np.array(padded_seq), dtype=torch.long)
        labels = torch.tensor(np.array(labels), dtype=torch.long)
        seq_len = torch.tensor(np.array(seq_len), dtype=torch.long)
        
        return {"input_ids": padded_seq, "labels": labels, "seq_len": seq_len, "attention_mask": attention_mask, "neg": neg}

       
class DataCollatorForDiffusion:
    def __init__(self, 
                mask_id,
                pad_id,
                mlm_probability = 0.15
                ):
        self.mask_id = mask_id
        self.pad_id = pad_id
        self.mlm_probability = mlm_probability

    def generate_masked_indices(self, inputs):
        # Sample Masked position. 
        probability_matrix = torch.full(inputs.shape, self.mlm_probability)
        padding_mask = inputs.eq(self.pad_id) 
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        return masked_indices

    def __call__(self, examples):
       
        # print("======================examples====================", examples[0]['input_ids'].shape)
        inputs = torch.cat([a['input_ids'].reshape(1, -1) for a in examples], axis=0)
        attention_mask = torch.cat([a['attention_mask'].reshape(1, -1) for a in examples], axis=0)
        labels = torch.cat([a['labels'].reshape(-1) for a in examples], axis=0)
        seq_len = torch.cat([a['seq_len'].reshape(-1) for a in examples], axis=0)
        
        masked_inputs1 = inputs.clone()
        masked_indices1 = self.generate_masked_indices(masked_inputs1)
        masked_inputs1[masked_indices1] = self.mask_id
        
        masked_inputs2 = inputs.clone()
        masked_indices2 = self.generate_masked_indices(masked_inputs2)
        masked_inputs2[masked_indices2] = self.mask_id
        # n_sample = int(masked_indices.sum()) # 15% --320
        # print("debug inputs", inputs)
        # print("debug masked_inputs1", masked_inputs1)
        # print("debug masked_inputs2", masked_inputs2)
        # print("debug labels", labels)
        # print("debug seq_len", seq_len)
       
        return {"input_ids": inputs.long(), "masked_inputs1": masked_inputs1.long(), "masked_inputs2": masked_inputs2.long(), 
                "attention_mask":attention_mask.float(),
                "labels": labels.long(), "seq_len": seq_len.long()}
        
        
     
##### from ICLRec ####
class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

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

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

   
        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )
        
        

        return cur_rec_tensors

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

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
    
     
class DataCollatorForCL:
    def __init__(self, 
                mask_id,
                pad_id,
                mlm_probability = 0.15
                ):
        self.mask_id = mask_id
        self.pad_id = pad_id
        self.mlm_probability = mlm_probability

    def generate_masked_indices(self, inputs):
        # Sample Masked position. 
        probability_matrix = torch.full(inputs.shape, self.mlm_probability)
        padding_mask = inputs.eq(self.pad_id) 
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        return masked_indices

    def __call__(self, examples):
        user_id = torch.cat([a[0].reshape(-1) for a in examples], axis=0)
        input_ids = torch.cat([a[1].reshape(1, -1) for a in examples], axis=0)
        target_pos = torch.cat([a[2].reshape(1, -1) for a in examples], axis=0)
        target_neg = torch.cat([a[3].reshape(1, -1) for a in examples], axis=0)
        answer = torch.cat([a[4].reshape(-1) for a in examples], axis=0)
        
        
        masked_inputs1 = copy.deepcopy(input_ids)
        masked_indices1 = self.generate_masked_indices(masked_inputs1)
        masked_inputs1[masked_indices1] = self.mask_id
        
        masked_inputs2 = copy.deepcopy(input_ids)
        masked_indices2 = self.generate_masked_indices(masked_inputs2)
        masked_inputs2[masked_indices2] = self.mask_id
       
        return (user_id, input_ids, target_pos, target_neg, answer, masked_inputs1, masked_inputs2)
        