import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import psutil

from tqdm import tqdm
import numpy as np

from dataset_PTB_XL import PTB_XL_dataset
from resnet import resnet34
from utils import cal_f1s, cal_aucs, split_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/PTB_XL', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use GPU')
    parser.add_argument('--original-model-path', type=str, default='', help='Path to saved model')
    parser.add_argument('--new_model_path', type=str, default='', help='Path to save new model')
    return parser.parse_args()


def train(dataloader, net, args, criterion, epoch, scheduler, optimizer, device):
    print('Training epoch %d:' % epoch)
    net.train()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    # scheduler.step()
    print('Loss: %.4f' % running_loss)
    

def evaluate(dataloader, net, args, criterion, device):
    print('Validating...')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        loss = criterion(output, labels)
        running_loss += loss.item()
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print('Loss: %.4f' % running_loss)
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    f1s = cal_f1s(y_trues, y_scores)
    avg_f1 = np.mean(f1s)
    print('F1s:', f1s)
    print('Avg F1: %.4f' % avg_f1)
    if args.phase == 'train' and avg_f1 > args.best_metric:
        args.best_metric = avg_f1
        torch.save(net.state_dict(), args.new_model_path)
    else:
        aucs = cal_aucs(y_trues, y_scores)
        avg_auc = np.mean(aucs)
        print('AUCs:', aucs)
        print('Avg AUC: %.4f' % avg_auc)


if __name__ == "__main__":
    args = parse_args()
    args.best_metric = 0
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)

    if not args.original_model_path:
        args.original_model_path = f'models/resnet34_{database}_{args.leads}_{args.seed}_{args.epochs}.pth'

    if not args.new_model_path:
        meta_info = args.original_model_path.split('_')
        original_database = meta_info[1]
        original_leads = meta_info[2]
        original_seed = meta_info[3]
        original_epochs = int(meta_info[4].split('.')[0])
        args.new_model_path = f'models/resnet34_{original_database}&{database}_{args.leads}_{args.seed}_{args.epochs + original_epochs}.pth'

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
    
    label_csv = os.path.join(data_dir, 'labels_with_final_mapping.csv')
    
    # we continue with ECG Dataset class since our data has been converted into the same format
    train_folds, val_folds, test_folds = split_data(seed=args.seed)
    train_dataset = PTB_XL_dataset('train', data_dir, label_csv, train_folds, leads)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataset = PTB_XL_dataset('val', data_dir, label_csv, val_folds, leads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataset = PTB_XL_dataset('test', data_dir, label_csv, test_folds, leads)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    net = resnet34(input_channels=nleads).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    
    criterion = nn.BCEWithLogitsLoss()
    
    if args.phase == 'train':
        if args.resume:
            net.load_state_dict(torch.load(args.original_model_path, map_location=device))
        
        for epoch in range(args.epochs):
            train(train_loader, net, args, criterion, epoch, scheduler, optimizer, device)
            evaluate(val_loader, net, args, criterion, device)
            # might have to increase number of files that can be opened using ulimit -n 2560           
    else:
        net.load_state_dict(torch.load(args.new_model_path, map_location=device))
        evaluate(test_loader, net, args, criterion, device)
