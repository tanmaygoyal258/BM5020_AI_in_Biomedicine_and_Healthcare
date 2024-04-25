import os 
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import wfdb
import scipy
from PIL import Image
from resnet import resnet34
from utils import prepare_input
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/combined_data', help='Directory to data dir')
    parser.add_argument('--number-grad-cams', type=int, default=5, help='Number of gradcams to produce')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use gpu')
    parser.add_argument('--model-path', type=str, default='models/resnet34_CPSC_all_42_30.pth', help='Path to saved model')
    parser.add_argument('--folder-to-save' , type=str , default = "gradcams")
    parser.add_argument('--percentile-threshold' , type=int , default=95, help = 'Percentile at which to threshold gradients')
    parser.add_argument('--label-file-name' , type=str , default = 'label.csv')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)
    if args.use_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = 'cpu'

    np.random.seed(args.seed)
    nleads = 12
    
    list_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    list_classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']

    model_path = args.model_path

    model = resnet34(input_channels=nleads).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    label_df = pd.read_csv(os.path.join(data_dir, args.label_file_name)  ,dtype = {"patient_id":str})
    patient_ids = label_df['patient_id'].to_numpy()
    random_indices = np.random.choice([i for i in range(len(patient_ids))] , args.number_grad_cams , replace= False)
    patient_ids = patient_ids[random_indices]

    for id in tqdm(patient_ids):

        file_path = os.path.join("data/CPSC", id) if 'A' in id else os.path.join("data/PTB_XL", id)
        # tranposed data of size 12 x length
        original_input = prepare_input(file_path) 
        input = np.expand_dims(original_input , axis = 0)
        labels = [c for c in label_df.columns if label_df[label_df['patient_id']==id][c].item()==1]
        model_input = torch.from_numpy(input).float().to(device)
        output = model(model_input)

        prediction = torch.argmax(output).item()
        
        output[:,prediction].backward()
        gradients = model.get_activations_gradient()

        pooled_gradients = torch.mean(gradients , axis = [0,1])
        activations = model.get_activations(model_input)[0]
        
        # Weight the channels by corresponding gradients
        heatmap = activations @ pooled_gradients.reshape(-1,1)
        heatmap = heatmap.cpu().detach().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        heatmap[np.where(heatmap < np.percentile(heatmap , args.percentile_threshold))] = 0

        if not os.path.exists(f'{args.folder_to_save}/{id}'):
            os.mkdir(f'{args.folder_to_save}/{id}')

        if 'A' in file_path:
            file_data = wfdb.rdrecord(file_path)
        else:
            file_path = str(file_path) + ".mat"
            file_data = scipy.io.loadmat(file_path)['val']

        for i in range(len(list_leads)):
            fig = plt.figure(figsize=(10, 2))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(f'Patient {id}: True diagnosis {labels} , Predicted diagnosis {list_classes[prediction]}')
            ax.plot(file_data[: , i], color='b')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(list_leads[i])
            yabs_max = abs(max(ax.get_ylim(), key=abs))
            ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
            ax.set_xlim(0 , 1000)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.imshow(heatmap.T, extent=[*xlim, *ylim], cmap='Reds', alpha=0.6, aspect=100)
            plt.grid(False)
            plt.savefig(f'{args.folder_to_save}/{id}/{list_leads[i]}.png')
            plt.close(fig)    



