import torch
import wfdb
import os
from torchvision import models, transforms
from PIL import Image as PilImage
from resnet import resnet34
from omnixai.data.image import Image
from omnixai.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM

classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
cpsc_dir = "data/CPSC"
patient_id = "A0001"
ecg_data, _ = wfdb.rdsamp(os.path.join(cpsc_dir, patient_id))
ecg_data = torch.tensor(ecg_data)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = 'cpu'

nleads = 12

model_path = "models/resnet34_combined_12_42_30.pth"
model = resnet34(input_channels=nleads).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def flatten(model):
    submodules = list(model.children())
    if len(submodules) == 0:
        return [model]
    else:
        res = []
        for module in submodules:
            res += flatten(module)
        return res

res = flatten(model)
for i , m in enumerate(res):
    print(i, m)

explainer = GradCAM(
    model=model_path,
    target_layer=model.layer4[-1],
    preprocess_function = None
)
explanations = explainer.explain(ecg_data)
explanations.ipython_plot(index=0 , class_names=classes)