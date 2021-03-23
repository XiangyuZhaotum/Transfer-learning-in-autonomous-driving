from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model_list, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    for model in model_list:
        model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")): 

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])  
        targets[:, 2:] *= img_size 

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        img_dim = imgs.shape[2]
        
        with torch.no_grad(): 
            if len(model_list) == 2:  # modularization 1
                encoder, detector = model_list
                h = encoder(imgs)  
                outputs = detector(h, img_dim)  
            elif len(model_list) == 1:  # modularization 2
                encoder = model_list[0]
                fs, fm, fl, outputs = encoder(imgs,img_dim)
            else:  # modularization 3
                encoder, detector_1, detector_2, detector_3 = model_list
                h_detector_1, h_detector_2, h_detector_3 = encoder(imgs)
                outputs_detector_1 = detector_1(h_detector_1, img_dim) 
                outputs_detector_2 = detector_2(h_detector_2, img_dim)
                outputs_detector_3 = detector_3(h_detector_3, img_dim)
                outputs = torch.cat([outputs_detector_1, outputs_detector_2, outputs_detector_3], 1)
            
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres) 

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))] 
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels) 

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default="config/custom_DANN_cluster_fakenight.data", help="path to data config file")
    parser.add_argument("--modularization", choices=[1,2,3], type=int, default=1)
    parser.add_argument("--models_def", nargs='+', help="paths to encoder and detector definition files")  # first encoder, then detectors in ascending order
    parser.add_argument("--pretrained_weights", nargs='*', help="paths to encoder and detector pretrained weights")  # can be left empty
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    check_modularization(opt.modularization, opt.models_def, opt.pretrained_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    test_path = data_config["test_night"]
    class_names = load_classes(data_config["names"])

    # Initialize model
    if opt.modularization == 1:
        encoder = Encoder_Mod1(opt.models_def[0]).to(device)
        detector = Detector_Mod1(opt.models_def[1]).to(device)
        encoder.load_state_dict(opt.pretrained_weights[0])
        detector.load_state_dict(opt.pretrained_weights[1])
        model_list = [encoder, detector]
    elif opt.modularization == 2:
        encoder = Encoder_Mod2(opt.models_def[0]).to(device)
        encoder.load_state_dict(opt.pretrained_weights[0])
        model_list = [encoder]
    else:
        encoder = Encoder_Mod3(opt.models_def[0]).to(device)
        detector_1 = Detector_Mod3(opt.models_def[1]).to(device)
        detector_2 = Detector_Mod3(opt.models_def[2]).to(device)
        detector_3 = Detector_Mod3(opt.models_def[3]).to(device)
        encoder.load_state_dict(opt.pretrained_weights[0])
        detector_1.load_state_dict(opt.pretrained_weights[1])
        detector_2.load_state_dict(opt.pretrained_weights[2])
        detector_3.load_state_dict(opt.pretrained_weights[3])
        model_list = [encoder, detector_1, detector_2, detector_3]

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model_list,
        path=test_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
