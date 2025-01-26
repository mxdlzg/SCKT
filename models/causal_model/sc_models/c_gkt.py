import torch
import torch.nn as nn
import torch.nn.functional as F
from pykt.models.gkt import GKT, device

class TimeCausalRegulator(nn.Module):
    def __init__(self, concept_num, emb_size=200, max_len=200, temperature=0.1, l1_lambda=0.001):
        super(TimeCausalRegulator, self).__init__()
        self.temperature = temperature
        self.time_causal_matrix = nn.Parameter(
            torch.zeros(max_len)
        )
        self.concept_causal_matrix = nn.Parameter(
            torch.zeros(concept_num)
        )
        nn.init.constant_(self.time_causal_matrix, 0.1)
        nn.init.constant_(self.concept_causal_matrix, 0.1)

        self.l1_lambda = l1_lambda  # 保存正则化系数
        self.step_norm_loss = 0

    def forward(self, concepts, concept_embs, sample_type="bernoulli", epoch=None):
        if not self.training:
            if sample_type == "bernoulli":
                return concept_embs
        self.step_norm_loss = None
        batch_size, seq_len, emb_size = concept_embs.size()

        # 1. 获取时间和概念的因果权重并加入温度系数
        time_weights = self.time_causal_matrix / self.temperature
        concept_weights = self.concept_causal_matrix / self.temperature

        # 2. 构建因果矩阵 (seq_len, concept_num)
        #    使用 outer product, 可以节省显存
        causal_matrix = torch.outer(time_weights, concept_weights)#torch.sigmoid(torch.outer(time_weights, concept_weights))

        if self.training:
            self.step_norm_loss = self.l1_lambda * torch.norm(causal_matrix)

        # 3. 根据sample_type进行采样
        if sample_type == "gumbel":
            mask, out_concept_weight = self.gumbel_sample(causal_matrix, concepts)
        elif sample_type == "bernoulli":
            mask, out_concept_weight = self.bernoulli_sample(causal_matrix, concepts)
        else:
            raise ValueError("Invalid sample_type. Choose 'gumbel' or 'bernoulli'.")

        # 4. 应用掩码到概念
        # 将mask扩充到batch_size维度
        cal_concept_weight = out_concept_weight[:,:seq_len].unsqueeze(-1)
        out_concept_embs = concept_embs * cal_concept_weight

        return out_concept_embs

    def gumbel_sample(self, logits, concepts):
        batch_size, seq_len = concepts.size()

        # 5. Gumbel 采样
        y = logits
        noise = torch.rand_like(y)
        y = y + torch.log(-torch.log(noise))
        y_softmax = torch.sigmoid(y)

        mask = (y_softmax > 0.5).float()
        # y_softmax = y_softmax.view(batch_size, -1, 1)
        weights = y_softmax[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
        weight_matrix = torch.gather(weights, 2, concepts.unsqueeze(-1)).squeeze(-1)

        return mask, weight_matrix

    def bernoulli_sample(self, logits, concepts):
        batch_size, seq_len = concepts.size()
        # Bernoulli 采样
        probabilities = torch.sigmoid(logits) # 将logits转为概率
        mask = torch.bernoulli(probabilities) # 根据概率进行二项分布采样

        # 将采样后的mask转换成权重矩阵，这里我们假设被选中的为1，否则为0
        weights = mask.float() # 将mask转为float类型方便计算
        weights = weights[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
        weight_matrix = torch.gather(weights, 2, concepts.unsqueeze(-1)).squeeze(-1)
        return mask, weight_matrix

class cGKT(GKT):
    def __init__(self, num_c, hidden_dim, emb_size, graph_type="dense", graph=None, dropout=0.5, emb_type="qid", emb_path="",bias=True):
        super().__init__(num_c, hidden_dim, emb_size, graph_type, graph, dropout, emb_type, emb_path,bias)
        self.model_name = "cgkt"
        self.time_pid_regulator = TimeCausalRegulator(self.res_len * num_c, emb_size)
        self.time_regulator = TimeCausalRegulator(num_c + 1, emb_size)

    def _aggregate(self, xt, qt, ht, batch_size):
        r"""
        Parameters:
            xt: input one-hot question answering features at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
            ht: hidden representations of all concepts at the current timestamp
            batch_size: the size of a student batch
        Shape:
            xt: [batch_size]
            qt: [batch_size]
            ht: [batch_size, num_c, hidden_dim]
            tmp_ht: [batch_size, num_c, hidden_dim + emb_size]
        Return:
            tmp_ht: aggregation results of concept hidden knowledge state and concept(& response) embedding
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        x_idx_mat = torch.arange(self.res_len * self.num_c, device=device)
        x_embedding = self.interaction_emb(x_idx_mat)  # [res_len * num_c, emb_size]#the emb for each concept with answer?
        x_embedding = self.time_pid_regulator(x_idx_mat, x_embedding)
        # print(xt[qt_mask])
        # print(self.one_hot_feat)
        masked_feat = F.embedding(xt[qt_mask], self.one_hot_feat)  # [mask_num, res_len * num_c] A simple lookup table that looks up embeddings in a fixed dictionary and size.
        #nn.functional.embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
        res_embedding = masked_feat.mm(x_embedding)  # [mask_num, emb_size]
        mask_num = res_embedding.shape[0]

        concept_idx_mat = self.num_c * torch.ones((batch_size, self.num_c), device=device).long()
        concept_idx_mat[qt_mask, :] = torch.arange(self.num_c, device=device)
        concept_embedding = self.emb_c(concept_idx_mat)  # [batch_size, num_c, emb_size]
        concept_embedding = self.time_regulator(concept_idx_mat, concept_embedding)

        index_tuple = (torch.arange(mask_num, device=device), qt[qt_mask].long())
        concept_embedding[qt_mask] = concept_embedding[qt_mask].index_put(index_tuple, res_embedding)
        tmp_ht = torch.cat((ht, concept_embedding), dim=-1)  # [batch_size, num_c, hidden_dim + emb_size]
        return tmp_ht