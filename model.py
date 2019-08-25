import os
import numpy as np
import torch
from PIL import Image
import skimage.draw
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
from engine import train_one_epoch, evaluate
import utils

import json

class_dict = { "short sleeve top" : 1, "long sleeve top":2, "short sleeve outwear":3, "long sleeve outwear" : 4, "vest":5, "sling" : 6, "shorts" : 7, "trousers":8 ,  "skirt" : 9,  "short sleeve dress" : 10, "long sleeve dress" : 11,  "vest dress" : 12 , "sling dress" : 13 }
class DeepFashion(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "train"))))
#         self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "train", self.imgs[idx])
        annos = os.path.join(self.root, "anno")
        img = Image.open(img_path).convert("RGB")
        
        #number of masks
        n_masks = 0
        for item in json.load(open(os.path.join(annos, (self.imgs[idx]).split(".")[-2]+".json"))):
#             cur = a[item]
            if(item[:4]!="item"):
                continue
            n_masks+=1
                
        width,height = img.size
        target = {}
        masks = np.zeros([n_masks,height,width],
                        dtype=np.uint8)
        boxes = []
        labels = []
        i = 0
        a = json.load(open(os.path.join(annos, (self.imgs[idx]).split(".")[-2]+".json")))
        for item in json.load(open(os.path.join(annos, (self.imgs[idx]).split(".")[-2]+".json"))):
            cur = a[item]
            if(item[:4]!="item"):
                continue
            boxes.append(cur["bounding_box"])
            labels.append(cur["category_id"])
            for poly in cur['segmentation']:
                all_x = (poly[0::2])
                all_y = (poly[1::2])
                rr, cc = skimage.draw.polygon(all_y, all_x)
                masks[i,rr,cc] = 1
                break
            i+=1

        num_objs = i

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        ########original##########
#         if self.transforms is not None:
#             img, target = self.transforms(img, target)
        ##########################


        if self.transforms is not None:
            img = self.transforms(img)
#             target = self.transforms(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 14 
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
dataset = DeepFashion(os.path.abspath(""), get_transform(train=True))
dataset_test = DeepFashion(os.path.abspath(""), get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
