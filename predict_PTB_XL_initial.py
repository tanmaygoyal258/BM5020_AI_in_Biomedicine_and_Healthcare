import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from resnet import resnet34
from dataset_PTB_XL import PTB_XL_dataset
from utils import cal_scores, find_optimal_threshold, split_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/PTB_XL', help='Directory to data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs to identify model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers to load data')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use gpu')
    parser.add_argument('--model-path', type=str, default='', help='Path to saved model')
    parser.add_argument('--threshold-path' , type=str , default='', help="Path to saved thresholds")
    return parser.parse_args()


def get_thresholds(val_loader, net, device, threshold_path):
    print('Finding optimal thresholds...')
    if os.path.exists(threshold_path):
        return pickle.load(open(threshold_path, 'rb'))
    output_list, label_list = [], []
    for _, (data, label) in enumerate(tqdm(val_loader)):
        data, labels = data.to(device), label.to(device)
        output = net(data)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(label_list)
    y_scores = np.vstack(output_list)
    thresholds = []
    for i in range(y_trues.shape[1]):
        y_true = y_trues[:, i]
        y_score = y_scores[:, i]
        threshold = find_optimal_threshold(y_true, y_score)
        thresholds.append(threshold)
    # pickle.dump(thresholds, open(threshold_path, 'wb'))
    return thresholds


def apply_thresholds(test_loader, net, device, thresholds):
    output_list, label_list = [], []
    for _, (data, label) in enumerate(tqdm(test_loader)):
        data, labels = data.to(device), label.to(device)
        output = net(data)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(label_list)
    y_scores = np.vstack(output_list)
    y_preds = []
    scores = [] 
    for i in range(len(thresholds)):
        y_true = y_trues[:, i]
        y_score = y_scores[:, i]
        y_pred = (y_score >= thresholds[i]).astype(int)
        scores.append(cal_scores(y_true, y_pred, y_score))
        y_preds.append(y_pred)
    y_preds = np.array(y_preds).transpose()
    scores = np.array(scores)
    print('Precisions:', scores[:, 0])
    print('Recalls:', scores[:, 1])
    print('F1s:', scores[:, 2])
    print('AUCs:', scores[:, 3])
    print('Accs:', scores[:, 4])
    print(np.mean(scores, axis=0))
    plot_cm(y_trues, y_preds)


def plot_cm(y_trues, y_preds, normalize=True, cmap=plt.cm.Blues):
    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    for i, label in enumerate(classes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=[0, 1], yticklabels=[0, 1],
           title=label,
           ylabel='True label',
           xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), ha="center")

        fmt = '.3f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        np.set_printoptions(precision=3)
        fig.tight_layout()
        plt.savefig(f'results_PTB_XL_initial/{label}.png')
        plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)
    if not args.model_path:
        args.model_path = f'models/resnet34_{database}_{args.leads}_{args.seed}_{args.epochs}.pth'
    if not args.threshold_path:
        args.threshold_path = f'models/{database}-threshold.pkl'
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
    data_dir = args.data_dir
    label_csv = os.path.join(data_dir, 'labels_with_final_mapping.csv')
    
    net = resnet34(input_channels=nleads).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.eval()

    print("Model has been loaded...")

    test_dataset = PTB_XL_dataset('test', data_dir, label_csv, None, leads)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print("Test dataset has been loaded...")
    
    thresholds = get_thresholds(test_loader, net, device, args.threshold_path)   
    print('Thresholds:', thresholds)
    apply_thresholds(test_loader, net, device, thresholds)
        