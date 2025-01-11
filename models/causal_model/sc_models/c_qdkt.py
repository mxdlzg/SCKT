import torch
import torch.nn as nn
import torch.nn.functional as F
from pykt.models.qdkt import QDKT, QDKTNet
from pykt.models.que_base_model import QueBaseModel

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
        if not self.training or epoch > 60:
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

class cQDKTNet(QDKTNet):
    def __init__(self, num_q,num_c,emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,device='cpu',mlp_layer_num=1,other_config={},time_regulator=None):
        super().__init__(num_q, num_c, emb_size, dropout, emb_type, emb_path, pretrain_dim, device, mlp_layer_num, other_config)
        self.model_name = "cqdkt"
        self.time_regulator = time_regulator

    def forward(self, q, c ,r,data=None):
        x = (q + self.num_q * r)[:,:-1]
        xemb = self.interaction_emb(x)

        xemb = self.time_regulator(x, xemb)

        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        y = (y * F.one_hot(data['qshft'].long(), self.num_q)).sum(-1)
        outputs = {"y":y}
        return outputs

class cQDKT(QueBaseModel):
    def __init__(self, num_q, num_c, emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,
                 device='cpu', seed=0, mlp_layer_num=1, other_config={}, **kwargs):
        model_name = "cqdkt"

        super().__init__(model_name=model_name, emb_type=emb_type, emb_path=emb_path, pretrain_dim=pretrain_dim,
                         device=device, seed=seed)
        self.time_regulator = TimeCausalRegulator(2*num_q, emb_size, **other_config)

        self.model = cQDKTNet(num_q=num_q, num_c=num_c, emb_size=emb_size, dropout=dropout, emb_type=emb_type,
                             emb_path=emb_path, pretrain_dim=pretrain_dim, device=device, mlp_layer_num=mlp_layer_num,
                             other_config=other_config, time_regulator=self.time_regulator)

        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")

    def train_one_step(self, data, process=True, return_all=False):
        outputs, data_new = self.predict_one_step(data, return_details=True, process=process)
        loss = self.get_loss(outputs['y'], data_new['rshft'], data_new['sm'])
        return outputs['y'], loss  # y_question没用

    def predict_one_step(self, data, return_details=False, process=True, return_raw=False):
        data_new = self.batch_to_device(data, process=process)
        outputs = self.model(data_new['cq'].long(), data_new['cc'], data_new['cr'].long(), data=data_new)
        if return_details:
            return outputs, data_new
        else:
            return outputs['y']
