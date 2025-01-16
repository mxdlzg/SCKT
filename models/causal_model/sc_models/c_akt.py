import torch
import torch.nn as nn
from pykt.models.akt import AKT


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
        self.step_norm_loss = torch.tensor(0)

    def forward(self, concepts, concept_embs, sample_type="bernoulli", epoch=None):
        self.step_norm_loss = torch.tensor(0)
        if not self.training or epoch > 60:
            return concept_embs
        batch_size, seq_len, emb_size = concept_embs.size()

        # 1. 获取时间和概念的因果权重并加入温度系数
        time_weights = self.time_causal_matrix / self.temperature
        concept_weights = self.concept_causal_matrix / self.temperature

        # 2. 构建因果矩阵 (seq_len, concept_num)
        #    使用 outer product, 可以节省显存
        causal_matrix = torch.outer(time_weights,
                                    concept_weights)  # torch.sigmoid(torch.outer(time_weights, concept_weights))

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
        cal_concept_weight = out_concept_weight[:, :seq_len].unsqueeze(-1)
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
        probabilities = torch.sigmoid(logits)  # 将logits转为概率
        mask = torch.bernoulli(probabilities)  # 根据概率进行二项分布采样

        # 将采样后的mask转换成权重矩阵，这里我们假设被选中的为1，否则为0
        weights = mask.float()  # 将mask转为float类型方便计算
        weights = weights[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
        weight_matrix = torch.gather(weights, 2, concepts.unsqueeze(-1)).squeeze(-1)
        return mask, weight_matrix


class cAKT(AKT):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256,
                 kq_same=1, final_fc_dim=512, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="",
                 pretrain_dim=768, use_time=True, use_time_q=True, use_time_pid=True, seq_len=200):
        super().__init__(n_question, n_pid, d_model, n_blocks, dropout, d_ff,
                         kq_same, final_fc_dim, num_attn_heads, separate_qa, l2, emb_type, emb_path, pretrain_dim)
        self.model_name = "cakt"
        self.use_time = use_time
        self.use_time_q = use_time_q
        self.use_time_pid = use_time_pid
        self.time_regulator = TimeCausalRegulator(n_question, emb_size=d_model, max_len=seq_len)
        self.time_q_regulator = TimeCausalRegulator(3, emb_size=d_model, max_len=seq_len)
        self.time_pid_regulator = TimeCausalRegulator(n_pid, emb_size=d_model, max_len=seq_len)

    def forward(self, q_data, target, pid_data=None, qtest=False, epoch=None):
        emb_type = self.emb_type
        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.n_pid > 0:  # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                           q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            if self.use_time:
                q_embed_data = self.time_regulator(q_data, q_embed_data, epoch=epoch)

            qa_embed_diff_data = self.qa_embed_diff(
                target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                                qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                                (
                                            qa_embed_diff_data + q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
            if self.use_time_q:
                qa_embed_data = self.time_q_regulator(target, qa_embed_data, epoch=epoch)
            if self.use_time_pid:
                pid_embed_data = self.time_pid_regulator(pid_data, pid_embed_data, epoch=epoch)
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2  # rasch部分loss
        else:
            c_reg_loss = 0.

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)
        if not qtest:
            return preds, c_reg_loss
        else:
            return preds, c_reg_loss, concat_q
