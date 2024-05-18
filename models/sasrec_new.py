import torch
from torch import nn
# import sys
# sys.path.append('./')
from .modules import TransformerEncoder


class SASRec(nn.Module):

    def __init__(self, config):
        super(SASRec, self).__init__()
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size
        self.inner_size = config.inner_size
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attn_dropout_prob = config.attn_dropout_prob
        self.hidden_act = config.hidden_act
        self.layer_norm_eps = config.layer_norm_eps

        self.initializer_range = config.initializer_range
        self.loss_type = config.loss_type
        self.max_seq_length = config.max_seq_length

        # define layers and loss
        self.item_embedding = nn.Embedding(config.item_size, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
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
        self.batch_size = config.train_batch_size
        self.loss_fct = nn.CrossEntropyLoss()
        # parameters initialization
        self.apply(self._init_weights)
        # print(" self.item_embedding",  self.item_embedding.weight)
        

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            # module.weight.data = self.truncated_normal_(tensor=module.weight.data, mean=0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        # print("debug attention_mask", attention_mask.shape)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def get_bi_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


        # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embedding(sequence)
        position_embeddings = self.position_embedding(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb
    
    # model same as SASRec
    def forward(self, input_ids):

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

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.trm_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output
    
    
    def calculate_loss(self, item_seq, target_pos, target_neg):
        
        seq_output = self.forward(item_seq)
        
        
        loss = self.cross_entropy(seq_output, target_pos, target_neg)
        
        
        # test_item_emb = self.item_embedding.weight
        # logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        # print("debug logits", logits.shape)
        # loss = self.loss_fct(logits, target_pos[:, -1])

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
        seq_output = self.forward(item_seq)
        seq_output = seq_output[:,-1,:]
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items+2]
        return scores
    
    
    
class SASRecCL(SASRec):
    def __init__(self, config):
        super(SASRecCL, self).__init__(config)
        self.config = config 
        self.nce_fct = nn.CrossEntropyLoss()
        
    def calculate_cl_loss(self, masked_seq1, masked_seq2):
        aug_seq1_output = self.forward(masked_seq1)[:,-1,:]
        aug_seq2_output = self.forward(masked_seq2)[:,-1,:]
        
        nce_logits, nce_labels = self.info_nce(aug_seq1_output, aug_seq2_output, temp=self.config.temperature, batch_size=aug_seq1_output.shape[0], sim="dot")
        nce_loss = self.nce_fct(nce_logits, nce_labels)
        return nce_loss
        
        
    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
            """
            We do not sample negative examples explicitly.
            Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
            """
            N = 2 * batch_size
        
            z = torch.cat((z_i, z_j), dim=0)
        
            if sim == 'cos':
                sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
            elif sim == 'dot':
                sim = torch.mm(z, z.T) / temp
        
            sim_i_j = torch.diag(sim, batch_size)
            sim_j_i = torch.diag(sim, -batch_size)
        
            positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        
            mask = self.mask_correlated_samples(batch_size)
            negative_samples = sim[mask].reshape(N, -1)
        
            labels = torch.zeros(N).to(positive_samples.device).long()
            logits = torch.cat((positive_samples, negative_samples), dim=1)
            return logits, labels
        
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
        


class SASRecGenCL(nn.Module):
    def __init__(self, config, item_embedding):
        super(SASRecGenCL, self).__init__()
        self.config = config
        self.nce_fct = nn.CrossEntropyLoss()
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size
        self.inner_size = config.inner_size
        self.hidden_dropout_prob = config.sasrec_dropout_prob
        self.attn_dropout_prob = config.sasrec_dropout_prob
        self.hidden_act = config.hidden_act
        self.layer_norm_eps = config.layer_norm_eps

        self.initializer_range = config.initializer_range
        self.loss_type = config.loss_type
        self.max_seq_length = config.max_seq_length

        # define layers and loss
        self.item_embedding = item_embedding # the embedding_layer is shared
        
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
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
        self.batch_size = config.train_batch_size

        # parameters initialization
        self.apply(self._init_weights)
        # print("self.item_embedding", self.item_embedding.weight)
        

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


        # Positional Embedding
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
    
    # model same as SASRec
    def forward(self, input_ids):
        extended_attention_mask = self.get_extended_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.trm_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
        sequence_output = item_encoded_layers[-1]
        return sequence_output
    
    
    def calculate_loss(self, item_seq, target_pos, target_neg):
        seq_output = self.forward(item_seq)
        loss = self.cross_entropy(seq_output, target_pos, target_neg)
        
        # test_item_emb = self.item_embedding.weight
        # logits = torch.matmul(seq_output[:, -1, :], test_item_emb.transpose(0, 1))
        # # print("logits", logits)
        # # print("labels", labels)
        # loss = self.nce_fct(logits, labels)

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
        seq_output = self.forward(item_seq)
        seq_output = seq_output[:,-1,:]
        
        test_items_emb = self.item_embedding.weight
        # test_items_emb = self.item_embedding.embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items+2]
        return scores

    def calculate_cl_loss(self, input_ids, aug_seq1_emb, aug_seq2_emb):
        extended_attention_mask = self.get_extended_attention_mask(input_ids)
        aug_seq1_emb = self.add_position_embedding(input_ids, aug_seq1_emb)
        aug_seq2_emb = self.add_position_embedding(input_ids, aug_seq2_emb)
        
        aug_seq1_output = self.trm_encoder(aug_seq1_emb, extended_attention_mask, output_all_encoded_layers=True)[-1][:,-1,:]
        aug_seq2_output = self.trm_encoder(aug_seq2_emb, extended_attention_mask, output_all_encoded_layers=True)[-1][:,-1,:]
        
        nce_logits, nce_labels = self.info_nce(aug_seq1_output, aug_seq2_output, temp=self.config.temperature, batch_size=aug_seq1_output.shape[0], sim="dot")
        nce_loss = self.nce_fct(nce_logits, nce_labels)
        return nce_loss
        
        
    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
            """
            We do not sample negative examples explicitly.
            Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
            """
            N = 2 * batch_size
        
            z = torch.cat((z_i, z_j), dim=0)
        
            if sim == 'cos':
                sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
            elif sim == 'dot':
                sim = torch.mm(z, z.T) / temp
        
            sim_i_j = torch.diag(sim, batch_size)
            sim_j_i = torch.diag(sim, -batch_size)
        
            positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        
            mask = self.mask_correlated_samples(batch_size)
            negative_samples = sim[mask].reshape(N, -1)
        
            labels = torch.zeros(N).to(positive_samples.device).long()
            logits = torch.cat((positive_samples, negative_samples), dim=1)
            return logits, labels

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask