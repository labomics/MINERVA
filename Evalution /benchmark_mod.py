# %%
import os
from os.path import join as pj
import argparse
import sys
sys.path.append("modules")
import utils
import numpy as np
import torch as th
import scib
import scib.metrics as me
import anndata as ad
import pandas as pd
import copy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from scipy.stats import pearsonr

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='bmmc_sub100')
parser.add_argument('--experiment', type=str, default='null')
parser.add_argument('--model', type=str, default='default')
parser.add_argument('--method', type=str, default='embed')
o, _ = parser.parse_known_args()  # for python interactive
test_dir = "predict"
start = 999
end = 1000
step = 100

# %%
data_dir = pj("/dev/shm/processed/", o.task)
data_config = utils.load_toml("configs/data.toml")[o.task]
for k, v in data_config.items():
    vars(o)[k] = v
model_config = utils.load_toml("configs/model.toml")["default"]
if o.model != "default":
    model_config.update(utils.load_toml("configs/model.toml")[o.model])
for k, v in model_config.items():
    vars(o)[k] = v
o.s_joint, o.combs, *_ = utils.gen_all_batch_ids(o.s_joint, o.combs)

# %%
def mod_evaluate(o, pred, data_dir, result_dir):
    output_type = "embed"
    embed = "X_emb"
    batch_key = "batch"
    label_key = "label"
    mod_key = "modality"
    cluster_key = "cluster"
    si_metric = "euclidean"
    verbose = False

    results1 = {
        "asw_mod": {},
        "foscttm": {},
        "f1": {},
    }

    results2 = {
        "pearson_rna": {},
        "pearson_adt": {},
        "RMSLE_rna": {},
        "RMSLE_adt": {},
    }

    results3 = {
        "pearson_rna": {},
        "pearson_adt": {},
    }

    results4= {
        "RMSLE_rna": {},
        "RMSLE_adt": {},
    }

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

    for batch_id in pred.keys():
        
        print("Processing batch: ", batch_id)
        x = pred[batch_id]["x"]
        x_impt = pred[batch_id]["x_impt"]

        mask_dir = pj("/dev/shm/processed", o.task, "subset_"+str(batch_id), "mask")
        batch = str(batch_id) # toml dict key must be str

        results1["foscttm"][batch], results1["f1"][batch] = {}, {}
        results2["pearson_rna"][batch], results2["pearson_adt"][batch], results2["RMSLE_rna"][batch], results2["RMSLE_adt"][batch] = {}, {}, {}, {}
        results3["pearson_rna"][batch], results3["pearson_adt"][batch] = {}, {}
        results4["RMSLE_rna"][batch], results4["RMSLE_adt"][batch] = {}, {}

        for m in c.keys() - {"joint"}:
            for m_ in set(c.keys()) - {m, "joint"}:
                k = m+"_to_"+m_
                results1["foscttm"][batch][k] = 1 - utils.calc_foscttm(th.from_numpy(c[m]), th.from_numpy(c[m_]))
                
                # print("Computing f1")
                knn.fit(c[m], label)
                label_pred = knn.predict(c[m_])
                results1["f1"][batch][k] = f1_score(label, label_pred, average='micro')
                
                if m_ in ["rna", "adt"]:
                    mask = np.array(utils.load_csv(pj(mask_dir, m_+".csv"))[1][1:]).astype(bool)
                    x_gt = x[m_][:, mask].reshape(-1)
                    x_pred = x_trans[k][:, mask].reshape(-1)
                    # print("Computing pearson_"+m_)
                    results2["pearson_"+m_][batch][k] = pearsonr(x_gt.reshape(-1), x_pred.reshape(-1))[0]
                    results2["RMSLE_"+m_][batch][k] = np.sqrt(np.mean(np.power(np.log(x_gt + 1) - np.log(x_pred + 1), 2)))
                    results3["pearson_"+m_][batch] = [pearsonr(x_gt[:,i].reshape(-1), x_pred[:,i].reshape(-1))[0] for i in np.arange(x_pred.shape[1])]
                    results4["RMSLE_"+m_][batch] = [np.sqrt(np.mean(np.power(np.log(x_gt[:,i] + 1) - np.log(x_pred[:,i] + 1), 2))) for i in np.arange(x_pred.shape[1])]            
                    

    results1_avg = {metric: np.mean(utils.extract_values(v)) for metric, v in results1.items()}
    df2 = pd.DataFrame({
        'Pearson_RNA':      [results2_avg['pearson_rna']],
        'Pearson_ADT':      [results2_avg['pearson_adt']],
        'RMSLE_RNA':        [results2_avg['RMSLE_rna']],
        'RMSLE_ADt':        [results2_avg['RMSLE_adt']]
    })
    print(df2)

    merged_dict = {k: list(v.values()) for k, v in results3.items()}
    for key, value in merged_dict.items():
        csv_file = f'{result_dir}/{key}.csv'
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(value)

    merged_dict = {k: list(v.values()) for k, v in results4.items()}
    for key, value in merged_dict.items():
        csv_file = f'{result_dir}/{key}.csv'
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(value)

    return df2


# %%
o.mods = ["rna", "adt"]
df_mod_embed, df_impu_x = [], []
for i in list(range(start, end, step)):
    init_model = "sp_0000{:04d}".format(i)
    result_dir = pj("result", "comparison", o.task, o.experiment)
    # Load predicted latent variables
    o.pred_dir = pj("result", o.task, o.experiment, o.model, "predict", init_model)
    pred = utils.load_predicted(o, mod_latent=True, translate=True, input=True, group_by="subset")
    df_mod, df_impu = mod_evaluate(o, pred, data_dir, result_dir)
    df_mod["Model"] = "Step_" + "{:04d}".format(i + 1)
    df_impu["Model"] = "Step_" + "{:04d}".format(i + 1)
    df_mod_embed.append(df_mod)
    df_impu_x.append(df_impu)


# %%
df_mod_cat = pd.concat(df_mod_embed, axis = 0)
df_mod_mean_cat = copy.deepcopy(df_mod_cat)
df_mod_mean_cat["mod_score"] = df_mod_cat[["ASW_mod", "FOSCTTM", "Label_transfer"]].mean(axis = 1)
df_mod_mean_cat = df_mod_mean_cat[["Model", "ASW_mod", "FOSCTTM", "Label_transfer", "mod_score"]]
df_mod_mean_cat.to_csv(pj(result_dir, "metrics_mod_train.csv"), index = False)
print(df_mod_mean_cat)


# %%
df_impu_cat = pd.concat(df_impu_x, axis = 0)
df_impu_mean_cat = copy.deepcopy(df_impu_cat)
df_impu_mean_cat["impu_score"] = df_impu_cat[["Pearson_RNA", "Pearson_ADT"]].mean(axis = 1)
df_impu_mean_cat = df_impu_mean_cat[["Model", "Pearson_RNA", "Pearson_ADT", "impu_score"]]
df_impu_mean_cat.to_csv(pj(result_dir, "metrics_impu_train.csv"), index = False)
print(df_impu_mean_cat)
