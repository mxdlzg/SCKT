import torch
import torch.nn as nn
import torch.nn.functional as F
from pykt.models.rekt import ReKT


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

    def forward(self, concepts, concept_embs, sample_type="gumbel", epoch=None):
        self.step_norm_loss = torch.tensor(0)
        if not self.training or (epoch is not None and epoch > 60):
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

class cReKT(ReKT):
    def __init__(self, skill_max, pro_max, d, dropout, emb_type="qid", seq_len=200):
        super(cReKT, self).__init__(skill_max, pro_max, d, dropout, emb_type)
        self.model_name = "crekt"
        self.time_regulator = TimeCausalRegulator(skill_max, emb_size=d, max_len=seq_len)
        self.time_q_regulator = TimeCausalRegulator(3, emb_size=d, max_len=seq_len)
        self.time_pid_regulator = TimeCausalRegulator(pro_max, emb_size=d, max_len=seq_len)

    def forward(self, dcur, qtest=False, train=False, epoch=None):
        """
        last_*:表示去掉序列最后面元素
        next_*:表示去掉序列最前面元素
        """
        last_problem, last_skill, last_ans = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        next_problem, next_skill, next_ans = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur[
            "shft_rseqs"].long()

        device = last_problem.device
        batch = last_problem.shape[0]
        seq = last_problem.shape[-1]

        pro_emb = F.embedding(next_problem, self.pro_embed)
        pro_emb = self.time_pid_regulator(next_problem, pro_emb)
        skill_emb = F.embedding(next_skill, self.skill_embed)
        skill_emb = self.time_regulator(next_skill, skill_emb)
        pro_diff_emb = F.embedding(next_problem, self.akt_pro_diff)
        pro_diff_emb = self.time_pid_regulator(next_problem, pro_diff_emb)
        pro_change_emb = F.embedding(next_skill, self.akt_pro_change)
        pro_change_emb = self.time_regulator(next_skill, pro_change_emb)
        next_pro_embed = (pro_emb
                          + skill_emb
                          + pro_diff_emb
                          * pro_change_emb ) # 对应论文中Et+1 = Qqt+1 + Cct+1 + diffqt+1 ∗ Vct+1

        next_ans_emb = F.embedding(next_ans.long(), self.ans_embed)
        next_ans_emb = self.time_q_regulator(next_ans, next_ans_emb)
        next_X = next_pro_embed + next_ans_emb  # Xt+1 = Et+1 + Rrt+1

        last_pro_time = torch.zeros((batch, self.pro_max)).to(device)  # batch pro
        last_skill_time = torch.zeros((batch, self.skill_max)).to(device)  # batch skill

        pro_state = self.pro_state.unsqueeze(0).repeat(batch, 1, 1)  # batch seq d 问题状态
        skill_state = self.skill_state.unsqueeze(0).repeat(batch, 1, 1)  # batch seq d 概念状态

        all_state = self.ls_state.repeat(batch, 1)  # batch d 领域状态

        last_pro_state = self.pro_state.unsqueeze(0).repeat(batch, 1, 1)  # batch seq d
        last_skill_state = self.skill_state.unsqueeze(0).repeat(batch, 1, 1)  # batch seq d

        batch_index = torch.arange(batch).to(device)

        all_time_gap = torch.ones((batch, seq)).to(device)
        all_time_gap_embed = F.embedding(all_time_gap.long(), self.time_embed)  # batch seq d

        res_p = []
        concat_q = []

        for now_step in range(seq):
            now_pro_embed = next_pro_embed[:, now_step]  # batch d

            now_item_pro = next_problem[:, now_step]  # batch
            now_item_skill = next_skill[:, now_step]

            last_batch_pro_time = last_pro_time[batch_index, now_item_pro]  # batch
            last_batch_pro_state = pro_state[batch_index, last_batch_pro_time.long()]  # batch d

            time_gap = now_step - last_batch_pro_time  # batch
            time_gap_embed = F.embedding(time_gap.long(), self.time_embed)  # batch d 问题时间间隔嵌入

            last_batch_skill_time = last_skill_time[batch_index, now_item_skill]  # batch
            last_batch_skill_state = skill_state[batch_index, last_batch_skill_time.long()]  # batch d

            skill_time_gap = now_step - last_batch_skill_time  # batch
            skill_time_gap_embed = F.embedding(skill_time_gap.long(), self.time_embed)  # batch d 概念时间间隔嵌入

            item_pro_state_forget = self.obtain_pro_forget(
                self.dropout(
                    torch.cat([last_batch_pro_state, time_gap_embed], dim=-1)))  # 遗忘模块 ft = Sigmoid([Zt−α ⊕ Iα]W1 + b1)
            last_batch_pro_state = last_batch_pro_state * item_pro_state_forget  # Responset = Zt−α ∗ ft

            item_skill_state_forget = self.obtain_skill_forget(
                self.dropout(torch.cat([last_batch_skill_state, skill_time_gap_embed], dim=-1)))
            last_batch_skill_state = last_batch_skill_state * item_skill_state_forget

            item_all_state_forget = self.obtain_all_forget(
                self.dropout(torch.cat([all_state, all_time_gap_embed[:, now_step]], dim=-1)))
            last_batch_all_state = all_state * item_all_state_forget

            last_pro_state[:, now_step] = last_batch_pro_state
            last_skill_state[:, now_step] = last_batch_skill_state

            final_state = torch.cat(
                [last_batch_all_state, last_batch_pro_state, last_batch_skill_state, now_pro_embed], dim=-1)

            P = torch.sigmoid(self.out(self.dropout(final_state))).squeeze(-1)

            concat_q.append(final_state)
            res_p.append(P)

            item_all_obtain = self.obtain_all_state(
                self.dropout(torch.cat([last_batch_all_state, next_X[:, now_step]], dim=-1)))
            item_all_state = last_batch_all_state + torch.tanh(
                item_all_obtain)  # 状态更新模块 Zt = Responset + T anh([Responset ⊕ Xt]W2 + b2)

            all_state = item_all_state

            pro_get = next_X[:, now_step]
            skill_get = next_X[:, now_step]

            item_pro_obtain = self.obtain_pro_state(
                self.dropout(torch.cat([last_batch_pro_state, pro_get], dim=-1)))
            item_pro_state = last_batch_pro_state + torch.tanh(item_pro_obtain)

            item_skill_obtain = self.obtain_skill_state(
                self.dropout(torch.cat([last_batch_skill_state, skill_get], dim=-1)))
            item_skill_state = last_batch_skill_state + torch.tanh(item_skill_obtain)

            last_pro_time[batch_index, now_item_pro] = now_step
            pro_state[:, now_step] = item_pro_state

            last_skill_time[batch_index, now_item_skill] = now_step
            skill_state[:, now_step] = item_skill_state

        concat_q = torch.vstack(res_p).T
        res_p = torch.vstack(res_p).T
        if train:
            return res_p
        else:
            if qtest:
                return res_p, concat_q
            else:
                return res_p