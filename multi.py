import os
import glob
import random
import argparse
import yaml
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from clip import clip
from dassl.metrics import compute_accuracy
from trainer.ours import *


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)

class AvgMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


class ImageFilter(nn.Module):
    """Filter images based on brightness threshold."""
    def __init__(self, brightness_threshold=0.01):
        super(ImageFilter, self).__init__()
        self.brightness_threshold = brightness_threshold

    def calculate_brightness(self, images):
        """Calculate average brightness of images."""
        grayscale_images = torch.mean(images, dim=1, keepdim=True)
        return grayscale_images.mean((2, 3))

    def forward(self, image_tensor):
        """Select images based on brightness criteria."""
        batch_size = image_tensor.size(0)
        brightness_values = self.calculate_brightness(image_tensor)
        
        indices_with_brightness = [
            i for i, value in enumerate(brightness_values) 
            if value >= self.brightness_threshold
        ]
        
        if len(indices_with_brightness) < batch_size:
            remaining_indices = [
                i for i in range(batch_size) 
                if i not in indices_with_brightness
            ]
            num_additional = batch_size - len(indices_with_brightness)
            additional_indices = random.sample(remaining_indices, 
                                             min(num_additional, len(remaining_indices)))
            return indices_with_brightness + additional_indices
        else:
            return random.sample(indices_with_brightness, batch_size)

class DataTrain(Dataset):
  def __init__(self,train_image_paths,train_domain,train_labels):
    self.image_path=train_image_paths
    self.domain=train_domain
    self.labels=train_labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self,idx):
    image = preprocess(Image.open(self.image_path[idx]))
    domain=self.domain[idx] 
    domain=torch.from_numpy(np.array(domain)) 
    label=self.labels[idx] 
    label=torch.from_numpy(np.array(label)) 
    # print("label",label)
    label_one_hot=F.one_hot(label,49)
  
    return image, domain, label, label_one_hot 

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device='cpu')

with open('prompts/prompts_list_multi.txt', 'r') as file:
    prompt_list = file.readlines()

            
attri_embed = torch.from_numpy(np.load('./attributes/attribute_multi.npy')).to(device).to(torch.float32)
mask_embed = torch.from_numpy(np.load('./attributes/masks_multi.npy')).to(device).to(torch.bool)

# Remove any trailing newline characters
prompt_list = [line.strip() for line in prompt_list]
# random.shuffle(prompt_list)

repeat_transform = transforms.Compose([
    transforms.ToTensor(),
])

class DataTrain(Dataset):
  def __init__(self,train_image_paths,train_domain,train_labels):
    self.image_path=train_image_paths
    self.domain=train_domain
    self.labels=train_labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self,idx):
    image = preprocess(Image.open(self.image_path[idx]))
    domain=self.domain[idx] 
    domain=torch.from_numpy(np.array(domain)) 
    label=self.labels[idx] 
    label=torch.from_numpy(np.array(label)) 
    # print("label",label)
    label_one_hot=F.one_hot(label,49)
  
    return image, domain, label, label_one_hot

parser = argparse.ArgumentParser(description='PACS Domain Adaptation Training')
parser.add_argument('--source_domains', type=str, required=True, 
                    help='Comma-separated source domains')
parser.add_argument('--target_domain', type=str, required=True, 
                    help='Target domain')
parser.add_argument('--shots', type=int, default=1, 
                    help='Number of shots per class')
parser.add_argument('--config', type=str, 
                    help='Path to config file', default='configs/pacs.yaml')
parser.add_argument('--data_root', type=str, default='/users/student/Datasets/domainbed/pacs',
                    help='Root directory for PACS data')
parser.add_argument('--output_dir', type=str, default='./experiments',
                    help='Output directory for results')
args = parser.parse_args()

import yaml

# Load the configuration from the YAML file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Split the source domains string into a list
source_domains = args.source_domains.split(',')
target_domains = args.target_domain

# Set the domains list
domains = source_domains + [target_domains]
target = domains[-1]
shots =args.shots
config["prompt_lr"]= 0.0012
config["project_dim"]= 128 

data_root = args.data_root
output_dir = args.output_dir


'''
############### The source dataset 1 ##################
'''

image_path_dom1=[]
label_class_dom1=[]
label_dom1=[]
class_names1=[]
paths_list={}
class_names4={}
all_classes={}
root1 = '/users/student/Datasets/domainbed/amazon/images'
paths_labels =[]
with open('/users/student/Datasets/domainbed/amazon_split.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('/')
        paths = line.strip().split()[0]
        # print(paths)
        if len(parts) >= 2:
            filename,label = parts[-1].split()
            class_name = parts[-2]
            if class_name not in class_names1 and (int)(label)<48:
                class_names1.append(class_name)
                class_names4[(int)(label)]=class_name         
            filename_with_class_name = f"{class_name}/{filename}"
            paths_list.setdefault(class_name, [])
            # print(os.path.join(root1,paths))
            paths_list[class_name].append((os.path.join(root1,paths),(int)(label)))
for i in class_names1:
    lists_sample = random.sample(paths_list[i],shots)
    paths_labels.extend(lists_sample)
random.shuffle(paths_labels)
for p,l in paths_labels:
    label_class_dom1.append((int)(l))
    image_path_dom1.append(p)
label_dom1.extend([0 for p in image_path_dom1])

# print(len(class_names1))
print(class_names4.keys())


'''
############### The source dataset 2 ##################
'''
paths_list={}
image_path_dom2=[]
label_class_dom2=[]
label_dom2=[]
class_names2=[]
root2 = '/users/student/Datasets/domainbed/visda/train'
paths_labels =[]
with open('/users/student/Datasets/domainbed/visda/visda_split.txt', 'r') as file:
    for line in file:
        # print(line)
        parts = line.strip().split('/')
        # print(parts)
        paths = line.strip().split()[0]
        if len(parts) >= 2:
            filename,label = parts[-1].split()
            # print(label)
            class_name = parts[-2]
            if class_name not in class_names2 and (int)(label)<48 and (int)(label) not in class_names4.keys():
                class_names2.append(class_name)    
                class_names4[(int)(label)]=class_name       
            filename_with_class_name = f"{class_name}/{filename}"
            paths_list.setdefault(class_name, [])
            paths_list[class_name].append((os.path.join(root2,paths),(int)(label)))
for i in class_names2:
    lists_sample = random.sample(paths_list[i],shots)
    paths_labels.extend(lists_sample)
random.shuffle(paths_labels)
for p,l in paths_labels:
    label_class_dom2.append((int)(l))
    image_path_dom2.append(p)
label_dom2.extend([1 for p in image_path_dom2])
print(class_names4.keys())


'''
############### The source dataset 3 ##################
'''

paths_list={}
image_path_dom3=[]
label_class_dom3=[]
label_dom3=[]
class_names3=[]
root3 = '/users/student/Datasets/domainbed/stl10/train'
paths_labels =[]
with open('/users/student/Datasets/domainbed/stl10/stl10_split.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('/')
        paths = line.strip().split()[0]
        if len(parts) >= 2:
            filename,label = parts[-1].split()
            class_name = parts[-2]
            if class_name not in class_names3 and (int)(label)<48:
                class_names3.append(class_name)           
                class_names4[(int)(label)]=class_name
            filename_with_class_name = f"{class_name}/{filename}"
            paths_list.setdefault(class_name, [])
            paths_list[class_name].append((os.path.join(root3,paths),(int)(label)))
for i in class_names3:
    lists_sample = random.sample(paths_list[i],shots)
    paths_labels.extend(lists_sample)
random.shuffle(paths_labels)
for p,l in paths_labels:
    label_class_dom3.append((int)(l))
    image_path_dom3.append(p)
label_dom3.extend([2 for p in image_path_dom3])
print(class_names4.keys())

'''
############### The combining the source dataset ##################
'''   
  
image_path_final=[]
image_path_final.extend(image_path_dom1)
image_path_final.extend(image_path_dom2)
image_path_final.extend(image_path_dom3)
label_class_final=[]
label_class_final.extend(label_class_dom1)
label_class_final.extend(label_class_dom2)
label_class_final.extend(label_class_dom3)
label_dom_final=[]
label_dom_final.extend(label_dom1)
label_dom_final.extend(label_dom2)
label_dom_final.extend(label_dom3)
domain_names=['amazon','synthetic 2D renderings', 'photo']
print("domain_names",domain_names)

'''
############### Test dataset ##################
'''

test_image_path_dom=[]
test_label_class_dom=[]
test_label_dom=[]
test_classnames=[]
# all_classes=[]
test_path_dom='./data/domainnet/'+target_domains[0]
test_domain_name = test_path_dom.split('/')[-1]
c=0
index=0
target_labels = [0, 1, 5, 6, 10, 11, 14, 17, 20, 26] + list(range(31, 37)) + list(range(39, 44)) + list(range(45, 47)) + list(range(48, 68))
known_index_dom = [0, 1, 5, 6, 10, 11, 14, 17, 20, 26] + list(range(31, 37)) + list(range(39, 44)) + list(range(45, 47))
root4 = f'/users/student/Datasets/domainbed/domainnet'
with open(f'/users/student/Datasets/domainbed/{target_domains}_test_new.txt', 'r') as file:
    paths_labels =[]
    for line in file:
        parts = line.strip().split('/')
        paths = line.strip().split()[0]
        if len(parts) >= 3:
            filename,label = parts[-1].split()
            class_name = parts[-2]
            if (int)(label) not in class_names4.keys() and (int)(label)<48:
                class_names4[(int)(label)]=class_name    
            if (int)(label) not in all_classes.keys() and (int)(label)>=48 :
                all_classes[(int)(label)]=class_name       
            filename_with_class_name = f"{class_name}/{filename}"
            # dom1_filenames.append(filename_with_class_name)
            # label_class_dom1.append((int)(label))
            # image_path_dom1.append(os.path.join(root1,paths))
            paths_labels.append((os.path.join(root4,paths),(int)(label)))
    random.shuffle(paths_labels)
    for p,l in paths_labels:
        test_label_class_dom.append((int)(l))
        test_image_path_dom.append(p) 
    test_label_dom.extend([3 for p in test_image_path_dom])

test_image_path_final=[]
test_image_path_final.extend(test_image_path_dom)
# test_classnames.extend(class_names4)

test_label_class_final=[]
test_label_class_final_modified = [label if label <= 47 else 48 for label in test_label_class_dom]
test_label_class_final.extend(test_label_class_final_modified)
# print("test_label_class_final",test_label_class_final)

test_label_dom_final=[]
test_label_dom_final.extend(test_label_dom)

test_domain_names = []
test_domain_names.append(test_domain_name)
test_domain_names.append(test_domain_name)
test_domain_names.append(test_domain_name) 

# Known Classes
# known_class_names = test_classnames.sort()
known_class_names=[]
for i in range(48):
    known_class_names.append(class_names4[i])
known_classes = ",".join(known_class_names)
unknown_class_names=[]
for i in range(48,68):
    unknown_class_names.append(all_classes[i])
print(unknown_class_names)
train_prev_classnames = known_classes.split(",")

print("known_classes: ",known_classes)

batchsize = config["batch_size"] #9
train_prev_ds=DataTrain(image_path_final,label_dom_final,label_class_final)
print(f'length of train_prev_ds: {len(train_prev_ds)}')

train_dl=DataLoader(train_prev_ds,batch_size=batchsize, num_workers=2, shuffle=True)
img_prev, domain_prev, label_prev, label_prev_one_hot = next(iter(train_dl))

domain_prev = domain_prev.to(device)

# train_prev_classnames = class_names[:54]
image_filter = ImageFilter(brightness_threshold=0.01)
def train_epoch(model,params, unknown_image_generator, domainnames, train_loader, optimizer, lr_scheduler, step,epoch):
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    warmup_period = 2000
    # tqdm_object = tqdm(train_loader, total=len(train_loader))
    batch  =1
    for img_prev, domain_prev, label_prev, label_one_hot_prev in train_loader:
        img_prev = img_prev.to(device)
        domain_prev = domain_prev.to(device)

        random_prompts = random.sample(prompt_list, 1)
        random_int = random.randint(0, 2)
        label_prev = label_prev.to(device)
        label_one_hot_prev = label_one_hot_prev.to(device)
        

        unknown_posprompt1 = "A photo of an " + random_prompts[0]
        generated_unknown_images1 = unknown_image_generator(batch, unknown_posprompt1, known_classes)
    


        unknown_label_rank = len(train_prev_classnames)
        unknown_label = torch.full((generated_unknown_images1.shape[0],), unknown_label_rank).to(device)
        
        unknown_domain1 = torch.full((generated_unknown_images1.shape[0],), 0).to(device)
        # unknown_domain2 = torch.full((generated_unknown_images2.shape[0],), 1).to(device)
        # unknown_domain3 = torch.full((generated_unknown_images3.shape[0],), 2).to(device)


        # generated_unknown_images = torch.cat((generated_unknown_images1, generated_unknown_images2,generated_unknown_images3), dim=0)
        # unknown_domains = torch.cat((unknown_domain1, unknown_domain2, unknown_domain3), dim=0)
        generated_unknown_images = generated_unknown_images1
        unknown_domains = unknown_domain1
        random_indices = image_filter(generated_unknown_images) 
        selected_images = generated_unknown_images[random_indices]
        selected_labels = unknown_label[random_indices]
        selected_domains = unknown_domains[random_indices]
        # print(len(random_indices))
        
        img = torch.cat((img_prev, selected_images), dim=0)
        img = img.to(device)

        label = torch.cat((label_prev, selected_labels), dim=0)
        label = label.to(device)

        domain = torch.cat((domain_prev, selected_domains), dim=0)
        domain = domain.to(device)
        # with profile(with_flops=True) as prof:

        output,loss_sty,invariant,feat = model(img,attri_embed,mask_embed,label,domain,len(random_indices))
 
        crossentropy_loss =     F.cross_entropy(output, label) + 0.33*loss_sty +(1-F.cosine_similarity(invariant,feat,dim=1)).mean()
    
        loss = crossentropy_loss 

        optimizer.zero_grad()
        loss.backward()
        # print(model.promptlearner.projector[0].linear.weight.grad)
        utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        count = img.size(0)
        loss_meter.update(loss.item(), count)

        acc = compute_accuracy(output, label)[0].item()
        accuracy_meter.update(acc, count)
    return loss_meter, accuracy_meter.avg

unknown_image_generator = GenerateUnknownImages().to(device)

train_classnames = train_prev_classnames + ['unknown']
print(f'length of train_classnames : {len(train_classnames)}')
domains_open = ["image",'2D rendering','grayscale view']
train_model = CustomCLIP(train_classnames, domains_open, clip_model,config)

for param in train_model.parameters():
            param.requires_grad_(False)
for p in train_model.cross_attention.parameters():
    p.requires_grad= True
train_model.projector.requires_grad = True
for p in train_model.promptlearner.parameters():
    p.requires_grad = True
params = [
            {"params": train_model.promptlearner.parameters(),'lr' : config["prompt_lr"]},
            {"params": train_model.projector.parameters(),'lr' : config["projector_lr"]},
            {"params": train_model.cross_attention.parameters(),'lr' : config["cross_attention_lr"]},
        ]
optimizer = torch.optim.AdamW(params,  weight_decay=config["weight_decay"])

warmup_epochs = 1

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=1, factor=0.8
        )
num_epochs = 10
warmup_period = 1
num_steps = len(train_dl) * num_epochs - warmup_period

all_params=[]
for group in optimizer.param_groups:
        all_params += group['params']
scaler = GradScaler() 

test_ds=DataTrain(test_image_path_final,test_label_dom_final,test_label_class_final)
print(len(test_ds))
test_dl=DataLoader(test_ds,batch_size=32, num_workers=4, shuffle=True)
test_img, test_domain, test_label, test_label_one_hot = next(iter(test_dl))

step = "epoch"
best_acc = 0
best_closed_set_acc = 0
best_open_set_acc = 0
best_avg_acc = 0
accuracy_file_path = f"{output_dir}/{domains[-1]}/{target}_{shots}.txt"  
accuracy_dir = os.path.dirname(accuracy_file_path)
if not os.path.exists(accuracy_dir):
    os.makedirs(accuracy_dir)
accuracy_file = open(accuracy_file_path, "w")
torch.autograd.set_detect_anomaly(True)

test_model = CustomCLIP(train_classnames, test_domain_names, clip_model,config).to(device)
train_model = train_model.to(device)
for epoch in range(num_epochs):
    closed_set_features = []    
    closed_set_labels = []  # To store labels of closed-set samples
    open_set_features = []
    print(f"Epoch: {epoch + 1}")
    train_model.train()
    train_loss, train_acc = train_epoch(train_model,all_params, unknown_image_generator, domain_names, train_dl, optimizer, lr_scheduler, step,epoch)
    print(f"epoch {epoch+1} : training accuracy: {train_acc}")

    save_path = f"{output_dir}/{domains[-1]}/{target}_{shots}_temp.pth"

    torch.save(obj=train_model.state_dict(), f=save_path)
    
    test_model.load_state_dict(torch.load(save_path))

    with torch.no_grad():
        test_probs_all = torch.empty(0).to(device)
        test_labels_all = torch.empty(0).to(device)
        test_class_all = torch.empty(0).to(device)
        test_tqdm_object = tqdm(test_dl, total=len(test_dl))

        total_correct_a = 0
        total_samples_a = 0
        total_correct_b = 0
        total_samples_b = 0
        
        for test_img, test_domain, test_label, test_label_one_hot in test_tqdm_object:
            test_img = test_img.to(device)
            test_domain =test_domain.to(device)
            test_label = test_label.to(device)
            test_label_one_hot = test_label_one_hot.to(device)
            
            # with profile(with_flops=True) as prof:
            test_output,_ = test_model(test_img.to(device),attri_embed,mask_embed,test_label)

            predictions = torch.argmax(test_output, dim=1)
            class_a_mask = (test_label <= 47) 
            class_b_mask = (test_label > 47)

            correct_predictions_a = (predictions[class_a_mask] == test_label[class_a_mask]).sum().item()
            correct_predictions_b = (predictions[class_b_mask] == test_label[class_b_mask]).sum().item()
            
            total_correct_a += correct_predictions_a
            total_samples_a += class_a_mask.sum().item()
            
            total_correct_b += correct_predictions_b
            total_samples_b += class_b_mask.sum().item()

        closed_set_accuracy = total_correct_a / total_samples_a if total_samples_a > 0 else 0.0
        closed_set_acc = closed_set_accuracy*100
        open_set_accuracy = total_correct_b / total_samples_b if total_samples_b > 0 else 0.0
        open_set_acc = open_set_accuracy*100

        average_acc = (2*closed_set_acc*open_set_acc)/(closed_set_acc + open_set_acc)

        print(f"Closed Set Accuracy: {closed_set_acc:.2f}%")
        print(f"Open Set Accuracy: {open_set_acc:.2f}%")
        print(f"Harmonic Score: {average_acc:.2f}%")
        
        # Write results
        accuracy_file.write(f"Epoch {epoch+1}\n")
        accuracy_file.write(f"Closed Set Accuracy: {closed_set_acc:.2f}%\n")
        accuracy_file.write(f"Open Set Accuracy: {open_set_acc:.2f}%\n") 
        accuracy_file.write(f"Harmonic Score: {average_acc:.2f}%\n")
        accuracy_file.write("-" * 40 + "\n")
        accuracy_file.flush()

        if average_acc > best_avg_acc:
            best_closed_set_acc = closed_set_acc
            best_open_set_acc = open_set_acc
            best_avg_acc = average_acc
            TEST_MODEL_PATH = Path(f"{output_dir}")
            TEST_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            TEST_MODEL_NAME = f"{target}ours.pth"
            TEST_MODEL_SAVE_PATH = TEST_MODEL_PATH / TEST_MODEL_NAME
            print(f"Saving test_model with best harmonic score to: {TEST_MODEL_SAVE_PATH}")
            torch.save(obj=test_model.state_dict(), f=TEST_MODEL_SAVE_PATH) 
            
            print(f"New best harmonic score: {best_avg_acc:.2f}%")
accuracy_file.write(f"End Closed Score: {best_closed_set_acc:.2f}%\n")
accuracy_file.write(f"End Harmonic Score: {best_avg_acc:.2f}%\n")

print(f"\nTraining completed. Best harmonic score: {best_avg_acc:.2f}%")
print(f"Results saved to: {accuracy_file_path}")
accuracy_file.close()