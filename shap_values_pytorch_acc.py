import argparse
import os

import numpy as np
import pandas as pd
import torch
import shap
from tqdm import tqdm
import matplotlib.pyplot as plt

from resnet import resnet34
from utils import prepare_input


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/CPSC', help='Data directory')
    parser.add_argument('--leads', type=str, default='all')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use GPU')
    parser.add_argument('--model-path' , default = None)
    parser.add_argument('--label-file-name', default=None)
    parser.add_argument('--reference-file-name', default=None)
    parser.add_argument('--shuffle' , default = False , action = 'store_true')
    parser.add_argument("--folder-to-save" , default = None)
    return parser.parse_args()


def plot_shap(ecg_data, sv_data, top_leads, patient_id, label , database):
    # patient-level interpretation along with raw ECG data
    leads = np.array(['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
    nleads = len(top_leads)
    if nleads == 0:
        return
    nsteps = 5000 # ecg_data.shape[1], visualize last 10 s since many patients' ECG are <=10 s
    x = range(nsteps)
    ecg_data = ecg_data[:, -nsteps:]
    sv_data = sv_data[:, -nsteps:]
    threshold = 0.001 # set threshold to highlight features with high shap values
    fig, axs = plt.subplots(nleads, figsize=(9, nleads))
    fig.suptitle(label)
    for i, lead in enumerate(top_leads):
        sv_upper = np.ma.masked_where(sv_data[lead] >= threshold, ecg_data[lead])
        sv_lower = np.ma.masked_where(sv_data[lead] < threshold, ecg_data[lead])
        if nleads == 1:
            axe = axs
        else:
            axe = axs[i]
        axe.plot(x, sv_upper, x, sv_lower)
        axe.set_xticks([])
        axe.set_yticks([])
        axe.set_ylabel(leads[lead])
    plt.savefig(f'shap/{args.folder_to_save}/shap1-{patient_id}.png')
    plt.close(fig)


def summary_plot(svs, y_scores,database):
    leads = np.array(['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
    svs2 = []
    n = y_scores.shape[0]
    for i in tqdm(range(n)):
        label = np.argmax(y_scores[i])
        sv_data = svs[label, i]
        svs2.append(np.sum(sv_data, axis=1))
    svs2 = np.vstack(svs2)
    svs_data = np.mean(svs2, axis=0)
    plt.plot(leads, svs_data)
    plt.savefig('./shap/{args.folder_to_save}/summary.png')
    plt.clf()


def plot_shap2(svs, y_scores, cmap=plt.cm.Blues , database = None):
    # population-level interpretation
    leads = np.array(['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
    n = y_scores.shape[0]
    results = [[], [], [], [], [], [], [], [], []]
    
    for i in tqdm(range(n)):
        label = np.argmax(y_scores[i])
        results[label].append(svs[label, i])
    ys = []
    for label in range(y_scores.shape[1]):
        result = np.array(results[label])
        y = []
        for i, _ in enumerate(leads):
            try:
                y.append(result[:,i].sum())
            except:
                y.append(0)
        y = np.array(y) / np.sum(y)
        ys.append(y)
        plt.plot(leads, y)
    ys.append(np.array(ys).mean(axis=0))
    ys = np.array(ys)
    fig, axs = plt.subplots()
    im = axs.imshow(ys, cmap=cmap)
    axs.figure.colorbar(im, ax=axs)
    fmt = '.2f'
    xlabels = leads
    ylabels = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE'] + ['AVG']
    axs.set_xticks(np.arange(len(xlabels)))
    axs.set_yticks(np.arange(len(ylabels)))
    axs.set_xticklabels(xlabels)
    axs.set_yticklabels(ylabels)
    thresh = ys.max() / 2
    for i in range(ys.shape[0]):
        for j in range(ys.shape[1]):
            axs.text(j, i, format(ys[i, j], fmt),
                    ha='center', va='center',
                    color='white' if ys[i, j] > thresh else 'black')
    np.set_printoptions(precision=2)
    fig.tight_layout()
    plt.savefig(f'./shap/{args.folder_to_save}/shap2.png')
    plt.clf()
    

if __name__ == '__main__':
    args = parse_args()
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)
    if not args.model_path:
        args.model_path = f'models/resnet34_{args.folder_to_save}_{args.leads}_{args.seed}.pth'
    if not args.label_file_name:
        args.label_file_name = f'labels.csv'
    if not args.reference_file_name:
        args.reference_file_name = f'reference.csv'
    if not args.folder_to_save:
        args.folder_to_save = f'{database}'
    label_csv = os.path.join(data_dir, args.label_file_name)
    # reference_csv = os.path.join(data_dir, 'reference.csv')
    lleads = np.array(['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
    classes = np.array(['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE'])
    if args.use_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = 'cpu'
    if args.leads == 'all':
        leads = 'all'
        nleads = 12
    else:
        leads = args.leads.split(',')
        nleads = len(leads)
    
    model = resnet34(input_channels=nleads).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    background = 100
    result_path = f'results_{args.folder_to_save}/A{background * 2}.npy'

    df_labels = pd.read_csv(label_csv , dtype = {"patient_id" : str})
    # df_reference = pd.read_csv(os.path.join(args.data_dir, args.reference_file_name) , dtype = {"patient_id" : str})
    # df = pd.merge(df_labels, df_reference[['patient_id', 'age', 'sex', 'signal_len']], on='patient_id', how='left')
    df = df_labels
    # df =  df[df['signal_len'] >= 15000]

    patient_ids = df['patient_id'].to_numpy()
    random_indices = np.random.choice([i for i in range(len(patient_ids))] , background*2 , replace= False)
    to_explain = patient_ids[:background * 2] if not args.shuffle else patient_ids[random_indices] 
    background_patient_ids = df.head(background)['patient_id'].to_numpy()
    if database != 'combined_data':
        background_inputs = [os.path.join(data_dir, patient_id) for patient_id in background_patient_ids]
    else:
        background_inputs = [os.path.join('data/CPSC', patient_id) if 'A' in patient_id else os.path.join('data/PTB_XL', patient_id) for patient_id in background_patient_ids]
    background_inputs = torch.stack([torch.from_numpy(prepare_input(input)).float() for input in background_inputs]).to(device)
    
    e = shap.GradientExplainer(model, background_inputs)

    svs = []
    y_scores = []
    if not os.path.exists(result_path):
        for patient_id in tqdm(to_explain):
            if database != "combined_data":
                input = os.path.join(data_dir, patient_id)
            else:
                if 'A' in patient_id:
                    input = os.path.join('data/CPSC', patient_id)
                else:
                    input = os.path.join('data/PTB_XL', patient_id)
            inputs = torch.stack([torch.from_numpy(prepare_input(input)).float()]).to(device)
            y_scores.append(torch.sigmoid(model(inputs)).detach().cpu().numpy())
            sv = np.array(e.shap_values(inputs)) # (n_classes, n_samples, n_leads, n_points)
            svs.append(sv)
        svs = np.concatenate(svs, axis=1)
        y_scores = np.concatenate(y_scores, axis=0)
        # np.save(result_path, (svs, y_scores))
    # svs, y_scores = np.load(result_path, allow_pickle=True)
    svs = np.array(svs)
    y_scores = np.array(y_scores)
    
    # summary_plot(svs, y_scores)
    plot_shap2(svs, y_scores , database = database)

    preds = []
    top_leads_list = []
    with open(f'shap/{args.folder_to_save}/top_leads_list.txt', 'w') as f:
        for i, patient_id in enumerate(to_explain):
            if database != "combined_data":
                ecg_data = prepare_input(os.path.join(data_dir, patient_id))
            else:
                if 'A' in patient_id:
                    ecg_data = prepare_input(os.path.join('data/CPSC', patient_id))
                else:
                    ecg_data = prepare_input(os.path.join('data/PTB_XL', patient_id))
            label_idx = np.argmax(y_scores[i])
            sv_data = svs[label_idx, i]
            
            sv_data_mean = np.mean(sv_data, axis=1)
            top_leads = np.where(sv_data_mean > 1e-4)[0] # select top leads
            preds.append(classes[label_idx])
            f.write(f'{patient_id} , {classes[label_idx]} , {lleads[top_leads]}')
            f.write("\n")

        plot_shap(ecg_data, sv_data, top_leads, patient_id, classes[label_idx] , database)
