import pandas as pd
from pykt.models.evaluate_model import predict_each_group2, get_info_dkt_forget, save_currow_question_res, \
    predict_each_group, save_each_question_res, cal_predres
from wandb.wandb_torch import torch

device = "cpu" if not torch.cuda.is_available() else "cuda"

def get_cur_teststart(is_repeat, train_ratio):
    curl = len(is_repeat)
    # print(is_repeat)
    qlen = is_repeat.count(0)
    qtrainlen = int(qlen * train_ratio)
    qtrainlen = 1 if qtrainlen == 0 else qtrainlen
    qtrainlen = qtrainlen - 1 if qtrainlen == qlen else qtrainlen
    # get real concept len
    ctrainlen, qidx = 0, 0
    i = 0
    while i < curl:
        if is_repeat[i] == 0:
            qidx += 1
        # print(f"i: {i}, curl: {curl}, qidx: {qidx}, qtrainlen: {qtrainlen}")
        # qtrainlen = 7 if qlen>7 else qtrainlen
        if qidx == qtrainlen:
            break
        i += 1
    for j in range(i+1, curl):
        if is_repeat[j] == 0:
            ctrainlen = j
            break
    return qlen, qtrainlen, ctrainlen

def evaluate_splitpred_question_tpn(model, data_config, testf, model_name, save_path="", use_pred=False, atkt_pad=False,
                                    t_length=50, num_n=20, ratio_array=None):
    if save_path != "":
        fout = open(save_path, "w", encoding="utf8")

    if ratio_array is None:
        ratio_array = [0.8]

    d_final_dict = {}
    for ratio in ratio_array:
        with torch.no_grad():
            y_trues = []
            y_scores = []
            dres = dict()
            idx = 0
            df = pd.read_csv(testf, encoding='utf-8')
            dcres, dqres = {"trues": [], "preds": []}, {"trues": [], "late_mean": [], "late_vote": [], "late_all": []}

            model.eval()
            for i, row in df.iterrows():
                concepts, responses = row["concepts"].split(","), row["responses"].split(",")
                dforget = dict() if model_name not in ["dkt_forget", "bakt_time"] else get_info_dkt_forget(row,
                                                                                                           data_config)
                ###
                # for AAAI competation
                rs = []
                for item in responses:
                    newr = item if item != "-1" else "0" # default -1 to 0
                    rs.append(newr)
                responses = rs
                ###

                data_seq_length = len(responses)

                # set repeat
                is_repeat = ["0"] * data_seq_length if "is_repeat" not in row else row["is_repeat"].split(",")
                is_repeat = [int(s) for s in is_repeat]
                questions = [] if "questions" not in row else row["questions"].split(",")
                times = [] if "timestamps" not in row else row["timestamps"].split(",")

                qlen, qtrainlen, ctrainlen = get_cur_teststart(is_repeat, ratio)
                cq = torch.tensor([int(s) for s in questions]).to(device)
                cc = torch.tensor([int(s) for s in concepts]).to(device)
                cr = torch.tensor([int(s) for s in responses]).to(device)
                ct = torch.tensor([int(s) for s in times]).to(device)
                dtotal = {"cq": cq, "cc": cc, "cr": cr, "ct": ct}

                curcin, currin = cc[0:ctrainlen].unsqueeze(0), cr[0:ctrainlen].unsqueeze(0)
                curqin = cq[0:ctrainlen].unsqueeze(0) if cq.shape[0] > 0 else cq
                curtin = ct[0:ctrainlen].unsqueeze(0) if ct.shape[0] > 0 else ct

                dcur = {"curqin": curqin, "curcin": curcin, "currin": currin, "curtin": curtin}
                t = ctrainlen

                curdforget = dict()
                for key in dforget:
                    dforget[key] = torch.tensor(dforget[key]).to(device)
                    curdforget[key] = dforget[key][0:ctrainlen].unsqueeze(0)

                ### 如果不是next by next，那直接并行。否则用前一步逐步循环预测后一步。
                if not use_pred:
                    uid, end = row["uid"], data_seq_length
                    qidx = qtrainlen
                    qidxs, ctrues, cpreds = predict_each_group2(dtotal, dcur, dforget, curdforget, is_repeat, qidx, uid,
                                                                idx, model_name, model, t, end, fout, atkt_pad)
                    # 计算
                    save_currow_question_res(idx, dcres, dqres, qidxs, ctrues, cpreds, uid, fout)
                else:
                    qidx = qtrainlen
                    while t < data_seq_length:
                        rtmp = [t]
                        for k in range(t + 1, data_seq_length):
                            if is_repeat[k] != 0:
                                rtmp.append(k)
                            else:
                                break

                        end = rtmp[-1] + 1
                        uid = row["uid"]
                        if model_name == "lpkt":
                            curqin, curcin, currin, curtin, curitin, curdforget, ctrues, cpreds = predict_each_group(dtotal,
                                                                                                                     dcur,
                                                                                                                     dforget,
                                                                                                                     curdforget,
                                                                                                                     is_repeat,
                                                                                                                     qidx,
                                                                                                                     uid,
                                                                                                                     idx,
                                                                                                                     model_name,
                                                                                                                     model,
                                                                                                                     t, end,
                                                                                                                     fout,
                                                                                                                     atkt_pad)
                            dcur = {"curqin": curqin, "curcin": curcin, "currin": currin, "curtin": curtin,
                                    "curitin": curitin}
                        elif model_name == "dimkt":
                            curqin, curcin, currin, curtin, cursdin, curqdin, ctrues, cpreds = predict_each_group(dtotal,
                                                                                                                  dcur,
                                                                                                                  dforget,
                                                                                                                  curdforget,
                                                                                                                  is_repeat,
                                                                                                                  qidx, uid,
                                                                                                                  idx,
                                                                                                                  model_name,
                                                                                                                  model, t,
                                                                                                                  end, fout,
                                                                                                                  atkt_pad)
                            dcur = {"curqin": curqin, "curcin": curcin, "currin": currin, "curtin": curtin,
                                    "cursdin": cursdin, "curqdin": curqdin}
                        else:
                            curqin, curcin, currin, curtin, curdforget, ctrues, cpreds = predict_each_group(dtotal, dcur,
                                                                                                            dforget,
                                                                                                            curdforget,
                                                                                                            is_repeat, qidx,
                                                                                                            uid, idx,
                                                                                                            model_name,
                                                                                                            model, t, end,
                                                                                                            fout, atkt_pad)
                            dcur = {"curqin": curqin, "curcin": curcin, "currin": currin, "curtin": curtin}
                        late_mean, late_vote, late_all = save_each_question_res(dcres, dqres, ctrues, cpreds)

                        fout.write("\t".join(
                            [str(idx), str(uid), str(qidx), str(late_mean), str(late_vote), str(late_all)]) + "\n")
                        t = end
                        qidx += 1
                idx += 1
            try:
                dfinal = cal_predres(dcres, dqres)
                for key in dfinal:
                    fout.write(key + "\t" + str(dfinal[key]) + "\n")
            except:
                print(f"can't output auc and accuracy!")
                dfinal = dict()
        d_final_dict[ratio] = dfinal
    return d_final_dict