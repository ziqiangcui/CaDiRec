import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from utils import q_xt_x0, p_xt
from .modules import DiffNet, TransformerEncoder, info_nce, mask_correlated_samples
from transformers.models.deberta.modeling_deberta import DebertaEncoder
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch as th
from utils import (
    SiLU,
    linear,
    timestep_embedding
)

class CaDiRec(nn.Module):

    def __init__(self, device, args):
        super(CaDiRec, self).__init__()
        self.item_size = args.item_size
        self.batch_size = args.train_batch_size
        self.hidden_size = args.hidden_size
        self.device = device
        self.args = args
        
        #************params for sasrec****************
        self.max_seq_length = args.max_seq_length
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.inner_size = args.inner_size
        self.hidden_dropout_prob = args.sasrec_dropout_prob
        self.attn_dropout_prob = args.sasrec_dropout_prob
        self.hidden_act = args.hidden_act
        self.layer_norm_eps = args.layer_norm_eps
        self.initializer_range = args.initializer_range
        self.batch_size = args.train_batch_size
        
        
        self.item_embedding = torch.nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        # self.place_embedding = nn.Embedding(1, args.hidden_size)

        self.nce_fct = nn.CrossEntropyLoss()
        self.rec_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
                
        self.input_dims = self.hidden_size
        self.hidden_t_dim = self.hidden_size
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(
            linear(self.hidden_size, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, self.hidden_size),
        )
        self.bert_encoder = DebertaEncoder(args)
        self.ln_bert = nn.LayerNorm(args.hidden_size)
        self.bias = nn.Parameter(torch.zeros(self.item_size))
        self.bert_loss_fct = CrossEntropyLoss()  # -100 index = padding token
        self.logits_mode = 1
        self.register_buffer("position_ids", torch.arange(self.max_seq_length).expand((1, -1)))
        self.softmax_loss_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)
        
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def get_embeds(self, item_ids):
        return self.item_embedding(item_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            test_items_emb = self.item_embedding.weight
            scores = torch.matmul(hidden_repr, test_items_emb.transpose(0, 1))  # [B n_items+2]
            return scores
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError
        

    def diffusion_reverse(self, x, timesteps, attention_mask):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))
        emb_x = x
    
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
      
        # emb_inputs = self.position_embedding(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.ln_bert(emb_inputs))

        input_trans_hidden_states = self.bert_encoder(emb_inputs,attention_mask,output_hidden_states=False,output_attentions=False,return_dict=False,)
        h = input_trans_hidden_states[0]
        # print(" debug: ", h.shape)
        h = h.type(x.dtype)
        return h

  
    
    def add_position_embedding(self, sequence, seq_emb=None):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        if seq_emb == None:
            item_embeddings = self.item_embedding(sequence)
        else:
            item_embeddings = seq_emb
        position_embeddings = self.position_embedding(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb
    

    def get_extended_attention_mask(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        subsequent_mask = subsequent_mask.cuda().to(input_ids.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


    def forward(self, seq_emb, extended_attention_mask):
        item_encoded_layers = self.rec_trm_encoder(seq_emb, extended_attention_mask, output_all_encoded_layers=True)
        sequence_output = item_encoded_layers[-1]
        return sequence_output
    
    
    def calculate_rec_loss(self, item_seq, target_pos, target_neg):
        extended_attention_mask = self.get_extended_attention_mask(item_seq)
        sequence_emb = self.add_position_embedding(item_seq)
        seq_output = self.forward(sequence_emb, extended_attention_mask)
        loss = self.cross_entropy(seq_output, target_pos, target_neg)

               
        # test_item_emb = self.item_embedding.weight
        # logits = torch.matmul(seq_output[:,-1,:], test_item_emb.transpose(0, 1))
        # pos_items = target_pos[:, -1]
        # loss = self.softmax_loss_fct(logits, pos_items)

        return loss
    
    
    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.item_embedding(pos_ids)
        neg_emb = self.item_embedding(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss


    def full_sort_predict(self, item_seq):
        extended_attention_mask = self.get_extended_attention_mask(item_seq)
        sequence_emb = self.add_position_embedding(item_seq)
        seq_output = self.forward(sequence_emb, extended_attention_mask)
        seq_output = seq_output[:,-1,:]
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items+2]
        return scores


    def calculate_cl_loss(self, aug_seq1, aug_seq2, emb1=None, emb2=None):
        
        extended_attention_mask = self.get_extended_attention_mask(aug_seq1)
        aug_seq1_emb = self.add_position_embedding(aug_seq1, emb1)
        seq1_output = self.forward(aug_seq1_emb, extended_attention_mask)[:,-1,:]
        aug_seq2_emb = self.add_position_embedding(aug_seq2, emb2)
        seq2_output = self.forward(aug_seq2_emb, extended_attention_mask)[:,-1,:]
        
        nce_logits, nce_labels = info_nce(seq1_output, seq2_output, temp=self.args.temperature, batch_size=aug_seq1.shape[0], sim="dot")
        nce_loss = self.nce_fct(nce_logits, nce_labels)
        return nce_loss
        
        