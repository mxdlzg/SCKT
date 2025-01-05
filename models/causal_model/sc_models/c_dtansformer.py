import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from pykt.models.dtransformer import DTransformer, MIN_SEQ_LEN, device


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


class cDtransformer(DTransformer):
    def __init__(self, n_question, n_pid, d_model=128, d_ff=256, num_attn_heads=8, n_know=16, n_blocks=3, dropout=0.3,
                 lambda_cl=0.1, proj=False, hard_neg=False, window=1, shortcut=False, separate_qa=False, emb_type="qid",
                 emb_path=""):
        super().__init__(n_question, n_pid, d_model, d_ff, num_attn_heads, n_know, n_blocks, dropout, lambda_cl, proj,
                         hard_neg, window, shortcut, separate_qa, emb_type, emb_path)
        self.model_name = "cdtransformer"
        self.time_regulator = TimeCausalRegulator(n_question, emb_size=d_model, max_len=200)
        self.time_q_regulator = TimeCausalRegulator(3, emb_size=d_model, max_len=200)
        self.time_pid_regulator = TimeCausalRegulator(n_pid, emb_size=d_model, max_len=200)

    def embedding(self, q_data, target, pid_data=None, epoch=None):
        lens = (target >= 0).sum(dim=1)
        if self.emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.n_pid > 0:  # have problem id
            q_embed_diff_data = self.q_diff_embed(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.p_diff_embed(pid_data)  # uq 当前problem的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                           q_embed_diff_data  # uq *d_ct + c_ct # question encoder
            q_embed_data = self.time_regulator(q_data, q_embed_data, epoch=epoch)

            qa_embed_diff_data = self.s_diff_embed(
                target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                                qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                                (qa_embed_diff_data + q_embed_diff_data)
            qa_embed_data = self.time_q_regulator(target, qa_embed_data, epoch=epoch)
            pid_embed_data = self.time_pid_regulator(pid_data, pid_embed_data, epoch=epoch)
        return q_embed_data, qa_embed_data, lens, pid_embed_data

    def get_cl_loss(self, q, s, pid=None, epoch=None):
        bs = s.size(0)

        # Input data preprocess
        if pid.size(1) ==0:
            pid = None
        q = q.to(device)
        s = s.to(device)
        if pid is not None:
            pid = pid.to(device)

        # skip CL for batches that are too short
        lens = (s >= 0).sum(dim=1)
        minlen = lens.min().item()
        if minlen < MIN_SEQ_LEN:

            return self.get_loss(q, s, pid,True)

        # augmentation
        q_ = q.clone()
        s_ = s.clone()

        if pid is not None:
            pid_ = pid.clone()
        else:
            pid_ = None

        # manipulate order
        for b in range(bs):
            idx = random.sample(
                range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
            )
            for i in idx:
                q_[b, i], q_[b, i + 1] = q_[b, i + 1], q_[b, i]
                s_[b, i], s_[b, i + 1] = s_[b, i + 1], s_[b, i]
                if pid_ is not None:
                    pid_[b, i], pid_[b, i + 1] = pid_[b, i + 1], pid_[b, i]

        # hard negative
        s_flip = s.clone() if self.hard_neg else s_
        for b in range(bs):
            # manipulate score
            idx = random.sample(
                range(lens[b]), max(1, int(lens[b] * self.dropout_rate))
            )
            for i in idx:
                s_flip[b, i] = 1 - s_flip[b, i]
        if not self.hard_neg:
            s_ = s_flip

    #     # model
        # logits, z_1, q_emb, reg_loss, _ = self.predict(q, s, pid)  #预测模型
        logits, concat_q, z_1, q_emb, reg_loss, _ = self.predict(q, s, pid, epoch=epoch)
        # masked_logits = logits[s >= 0]

        # extract forward
        # print(f"q_ shape:{q_.shape} s_ shape:{s_.shape}")
        _, _,  z_2, *_ = self.predict(q_, s_, pid_, epoch=epoch)

        if self.hard_neg:
           _, _, z_3, *_ = self.predict(q, s_flip, pid, epoch=epoch)

        # CL loss
        # print(f"z1 shape:{z_1.shape} z2 shape:{z_2.shape}")
        # import sys
        # sys.exit()
        input = self.sim(z_1[:, :minlen, :], z_2[:, :minlen, :])
        if self.hard_neg:
            hard_neg = self.sim(z_1[:, :minlen, :], z_3[:, :minlen, :])
            input = torch.cat([input, hard_neg], dim=1)
        target = (
            torch.arange(s.size(0))[:, None]
            .to(self.know_params.device)
            .expand(-1, minlen)
        )
        cl_loss = F.cross_entropy(input, target)

        #prediction loss
        # masked_labels = s[s >= 0].float()
        # pred_loss = F.binary_cross_entropy_with_logits(
        #     masked_logits, masked_labels, reduction="mean"
        # )

        for i in range(1, self.window):
            label = s[:, i:]
            query = q_emb[:, i:, :]
            h = self.readout(z_1[:, : query.size(1), :], query)
            y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)

            pred_loss += F.binary_cross_entropy_with_logits(
                y[label >= 0], label[label >= 0].float()
            )

        m = nn.Sigmoid()
        preds = m(logits)

        return preds, cl_loss * self.lambda_cl + reg_loss

    def predict(self, q, s, pid=None, n=1, epoch=None):
        # 判断张量是否为空
        if pid is None:
            pass
        else:
            if pid.nelement() == 0:
                # print("pid.nelement()")
                pid = None
            else:
                pass

        q_emb, s_emb, lens, p_diff = self.embedding(q, s, pid, epoch=epoch)
        # print(f"q_emb {q_emb.shape} q_emb {q_emb.shape} lens:{len}")
        z, q_scores, k_scores = self(q_emb, s_emb, lens)
        # print(f"z {z.shape} q_scores {q_scores.shape} k_scores:{k_scores.shape}")

        # predict T+N
        if self.shortcut:
            assert n == 1, "AKT does not support T+N prediction"
            h = z
        else:
            query = q_emb[:, n - 1:, :]
            h = self.readout(z[:, : query.size(1), :], query)

        # import sys
        # sys.exit()
        concat_q = torch.cat([q_emb, h], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        # print(f"yshape:{y.size()}")

        if pid is not None:
            return output, concat_q, z, q_emb, (p_diff ** 2).mean() * 1e-3, (q_scores, k_scores)
        else:
            return output, concat_q, z, q_emb, 0.0, (q_scores, k_scores)