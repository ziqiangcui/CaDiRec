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
        self.item_embedding = nn.Embedding(config.item_num+2, self.hidden_size, padding_idx=0)
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
        print("loss_type", self.loss_type)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.batch_size = config.train_batch_size
        self.nce_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)
        

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

    def forward(self, item_seq, seq_len):
        # print("item_seq", item_seq)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        # extended_attention_mask = self.get_bi_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]  # [B L H]
        output = self.gather_indexes(output, seq_len - 1)
        # print("debug output", output.shape)
        # output = output[:, -1, :]
        return output  # [B H]

    def calculate_loss(self, item_seq, pos_items, seq_len):
        seq_output = self.forward(item_seq, seq_len)
        # pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            # print("pos_items", pos_items)
            neg_items = torch.roll(pos_items, 1)
            # print("neg_items", neg_items)
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            # print("debug seq_output", seq_output.shape)
            # print("debug pos_items", pos_items)
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            # print("debug logits", logits.shape)
            loss = self.loss_fct(logits, pos_items)
            # print("debug loss", loss)

        return loss
    
    def calculate_constrastive_loss(self, masked_seq1, masked_seq2, seq_len):
        aug_seq1_output = self.forward(masked_seq1, seq_len)
        aug_seq2_output = self.forward(masked_seq2, seq_len)
        nce_logits, nce_labels = self.info_nce(aug_seq1_output, aug_seq2_output, temp=1, batch_size=aug_seq1_output.shape[0], sim="dot")
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
    
    
    def sample_predict(self, item_seq, seq_len, pos, negs):
       
        candidate_items = torch.cat([pos.reshape(-1,1), negs], axis=1)
        seq_output = self.forward(item_seq, seq_len)  # [B,h]
        candidate_item_emb = self.item_embedding(candidate_items)
       
        # 将 tensor1 扩展为与 tensor2 相同的形状，以便进行点积计算
        expanded_tensor1 = seq_output.unsqueeze(1).expand(-1, 101, -1)  # 形状变为 [B, 101, h]

        # 计算点积
        scores = torch.sum(expanded_tensor1 * candidate_item_emb, dim=-1)  # 在最后一个维度上计算点积，形状为 [B, 101]
        # scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B, n_neg+1]
        return scores

    def full_sort_predict(self, item_seq, seq_len):
        seq_output = self.forward(item_seq, seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
    
    
    
class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss