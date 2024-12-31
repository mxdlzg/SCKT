import torch
import torch.nn as nn
import torch.nn.functional as F
from pykt.models.dkt import DKT

class CausalRegulator(nn.Module):
    def __init__(self, n_knowledge, hidden_dim, dropout=0.2, alpha=1, temperature=0.3):
        super(CausalRegulator, self).__init__()
        self.n_knowledge = n_knowledge
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.temperature = temperature

        # 因果关系矩阵作为可训练参数
        self.causal_matrix = nn.Parameter(
            torch.zeros(n_knowledge, n_knowledge)
        )

        # 因果发现网络
        self.causal_discovery = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 知识点调节网络
        self.knowledge_regulate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, kc_embeddings):
        # 计算动态因果关系
        batch_size = kc_embeddings.size(0)

        noise = -torch.log(-torch.log(torch.rand_like(self.causal_matrix) + 1e-10) + 1e-10)
        dynamic_causal_matrix = F.softmax((self.causal_matrix + noise) / self.temperature, dim=-1)

        # 基于因果关系调节知识点表征
        causal_context = torch.bmm(dynamic_causal_matrix.unsqueeze(0).expand(batch_size, -1, -1), kc_embeddings)
        regulated_embeds = self.knowledge_regulate(
            torch.cat([kc_embeddings, causal_context], dim=-1)
        )

        # 动态融合
        final_embeds = (1 - self.alpha) * kc_embeddings + self.alpha * regulated_embeds

        return final_embeds, dynamic_causal_matrix

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
        nn.init.constant_(self.time_causal_matrix, 0.5)
        nn.init.constant_(self.concept_causal_matrix, 0.5)

        self.l1_lambda = l1_lambda  # 保存正则化系数
        self.step_norm_loss = 0

    def forward(self, concepts, concept_embs, sample_type="gumbel"):
        self.step_norm_loss = None
        batch_size, seq_len, emb_size = concept_embs.size()

        # 1. 获取时间和概念的因果权重并加入温度系数
        time_weights = self.time_causal_matrix / self.temperature
        concept_weights = self.concept_causal_matrix / self.temperature

        # 2. 构建因果矩阵 (seq_len, concept_num)
        #    使用 outer product, 可以节省显存
        causal_matrix = torch.outer(time_weights, concept_weights)#torch.sigmoid(torch.outer(time_weights, concept_weights))
        self.step_norm_loss = self.l1_lambda * torch.norm(causal_matrix)

        # 3. 根据sample_type进行采样
        if sample_type == "gumbel":
            mask, out_concept_weight = self.gumbel_sample(causal_matrix, concepts)
        elif sample_type == "bernoulli":
            mask = self.bernoulli_sample(causal_matrix, concept_embs)
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

    def bernoulli_sample(self, probs):
        """使用Bernoulli分布进行采样"""
        mask = torch.bernoulli(probs)
        return mask


class cDKT(DKT):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__(num_c, emb_size, dropout, emb_type, emb_path, pretrain_dim)
        self.model_name = "cdkt"
        self.time_regulator = TimeCausalRegulator(2*num_c+1, emb_size)

    def forward(self, q, r):
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
            xemb = self.time_regulator(x, xemb)
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y