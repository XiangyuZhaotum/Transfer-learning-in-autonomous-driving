from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets_DANN import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import copy

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import warnings

import random

from torch.autograd import Function
import numpy as np


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)


class Discriminator_joint(nn.Module):

    def __init__(self, input_nc, ndf=64):
        super(Discriminator_joint, self).__init__()

        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2, inplace=True)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


class Discriminator_alternating(nn.Module):

    def __init__(self, input_nc, ndf=64):
        super(Discriminator_alternating, self).__init__()

        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1 + np.exp(-10. * p)) - 1.
    
def read_weights(files, folder):
    weight_list = []
    for file in files:
        weight_path = folder + file
        with open(weight_path, 'r', encoding='utf-8') as f:
            for line in f:
                weight = float(line)
                weight_list.append(weight)
    weights = torch.Tensor([weight_list]).squeeze()
    return weights

def peak_at_one(w):
        tmp = 1.0 / (2 * torch.abs((w - 1)) + 1e-16)
        return 2 * F.sigmoid(tmp) - 1

def sample_night_city_street():
    global night_set_city_street
    try:
        return night_set_city_street.next()
    except StopIteration:
        night_set_city_street = iter(night_loader_city_street)
        return night_set_city_street.next()

def sample_night_highway():
    global night_set_highway
    try:
        return night_set_highway.next()
    except StopIteration:
        night_set_highway = iter(night_loader_highway)
        return night_set_highway.next()

def sample_night_residential():
    global night_set_residential
    try:
        return night_set_residential.next()
    except StopIteration:
        night_set_residential = iter(night_loader_residential)
        return night_set_residential.next()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default="config/custom_DANN_cluster_fakenight.data", help="path to data config file")
    parser.add_argument("--modularization", choices=[1,2,3], type=int, default=1)
    parser.add_argument("--models_def", nargs='+', help="paths to encoder and detector definition files")  # first encoder, then detectors in ascending order
    parser.add_argument("--pretrained_weights", nargs='*', help="paths to encoder and detector pretrained weights")  # can be left empty
    parser.add_argument("--weighting", choices=["sigmoid", "peak_at_one", None], default="peak_at_one")
    parser.add_argument("--weights_folder", default="./data/custom/weights/train/")
    parser.add_argument("--update_step", choices=["joint", "alternating"], default="alternating")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument('--input_nc', type=int, default=256, help='number of channels of input data') 
    opt = parser.parse_args()
    print(opt)

    check_modularization(opt.modularization, opt.models_def, opt.pretrained_weights)
    config_name = 'config' + str(opt.exp_id)

    os.makedirs("logs-DANN/{}/logs_night".format(config_name), exist_ok=True) 
    os.makedirs("logs-DANN/{}/logs_fakenight".format(config_name), exist_ok=True)  
    os.makedirs("logs-DANN/{}/logs_daytime".format(config_name), exist_ok=True)  
    os.makedirs("checkpoints-DANN/{}".format(config_name), exist_ok=True)
    logger_fakenight = Logger("logs-DANN/{}/logs_fakenight".format(config_name))
    logger_night = Logger("logs-DANN/{}/logs_night".format(config_name))
    logger_daytime = Logger("logs-DANN/{}/logs_daytime".format(config_name))

    if opt.modularization == 1:
        os.makedirs("logs-DANN/{}/logs_D".format(config_name), exist_ok=True) 
        train_writer_D = SummaryWriter('logs-DANN/{}/logs_D'.format(config_name))
    else:
        os.makedirs("logs-DANN/{}/logs_D_1".format(config_name), exist_ok=True) 
        os.makedirs("logs-DANN/{}/logs_D_2".format(config_name), exist_ok=True) 
        os.makedirs("logs-DANN/{}/logs_D_3".format(config_name), exist_ok=True) 
        train_writer_D_1 = SummaryWriter('logs-DANN/{}/logs_D_1'.format(config_name))
        train_writer_D_2 = SummaryWriter('logs-DANN/{}/logs_D_2'.format(config_name))
        train_writer_D_3 = SummaryWriter('logs-DANN/{}/logs_D_3'.format(config_name))

    data_config = parse_data_config(opt.data_config)
    class_names = load_classes(data_config["names"])

    # init the datasets of clusters of scenes
    night_train_city_street = ListDataset(data_config["train_night_city_street"], augment=True, multiscale=opt.multiscale_training)
    night_train_highway = ListDataset(data_config["train_night_highway"], augment=True, multiscale=opt.multiscale_training)
    night_train_residential = ListDataset(data_config["train_night_residential"], augment=True, multiscale=opt.multiscale_training)

    fakenight_train_city_street = ListDataset(data_config["train_fakenight_city_street"], augment=True, multiscale=opt.multiscale_training)
    fakenight_train_highway = ListDataset(data_config["train_fakenight_highway"], augment=True, multiscale=opt.multiscale_training)
    fakenight_train_residential = ListDataset(data_config["train_fakenight_residential"], augment=True, multiscale=opt.multiscale_training)

    batch_size = opt.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Discriminator = Discriminator_alternating if opt.update_step == "alternating" else Discriminator_joint

    # Initialize model
    if opt.modularization == 1:
        encoder = Encoder_Mod1(opt.models_def[0]).to(device)
        detector = Detector_Mod1(opt.models_def[1]).to(device)
        input_nc = parse_model_config(opt.models_def[1])[0]['channels']
        D = Discriminator(input_nc).to(device)
        D.apply(weights_init_normal)
        if opt.pretrained_weights:
            encoder.load_state_dict(opt.pretrained_weights[0])
            detector.load_state_dict(opt.pretrained_weights[1])
        else:
            encoder.apply(weights_init_normal)
            detector.apply(weights_init_normal)

    elif opt.modularization == 2:
        encoder = Encoder_Mod2(opt.models_def[0]).to(device)
        input_nc = (len(class_names)+5)*3
        D_1 = Discriminator(input_nc).to(device)
        D_2 = Discriminator(input_nc).to(device)
        D_3 = Discriminator(input_nc).to(device)
        D_1.apply(weights_init_normal), D_2.apply(weights_init_normal), D_3.apply(weights_init_normal)
        if opt.pretrained_weights:
            encoder.load_state_dict(opt.pretrained_weights[0])
        else:
            encoder.apply(weights_init_normal)
    else:
        encoder = Encoder_Mod3(opt.models_def[0]).to(device)
        detector_1 = Detector_Mod3(opt.models_def[1]).to(device)
        detector_2 = Detector_Mod3(opt.models_def[2]).to(device)
        detector_3 = Detector_Mod3(opt.models_def[3]).to(device)
        input_nc_1 = parse_model_config(opt.models_def[1])[0]['channels']
        input_nc_2 = parse_model_config(opt.models_def[2])[0]['channels']
        input_nc_3 = parse_model_config(opt.models_def[3])[0]['channels']
        D_1 = Discriminator(input_nc_1).to(device)
        D_2 = Discriminator(input_nc_2).to(device)
        D_3 = Discriminator(input_nc_3).to(device)
        D_1.apply(weights_init_normal), D_2.apply(weights_init_normal), D_3.apply(weights_init_normal)
        if opt.pretrained_weights:
            encoder.load_state_dict(opt.pretrained_weights[0])
            detector_1.load_state_dict(opt.pretrained_weights[1])
            detector_2.load_state_dict(opt.pretrained_weights[2])
            detector_3.load_state_dict(opt.pretrained_weights[3])
        else:
            encoder.apply(weights_init_normal)
            detector_1.apply(weights_init_normal)
            detector_2.apply(weights_init_normal)
            detector_3.apply(weights_init_normal)
        
    if opt.weighting == "sigmoid":
        transform_weights = F.sigmoid
    elif opt.weighting == "peak_at_one":
        transform_weights = peak_at_one 

    # Init the dataloader for each cluster in fakenight domain
    fakenight_loader_city_street = torch.utils.data.DataLoader(fakenight_train_city_street, batch_size=opt.batch_size, shuffle=True, pin_memory=True, collate_fn=fakenight_train_city_street.collate_fn)
    fakenight_loader_highway = torch.utils.data.DataLoader(fakenight_train_highway, batch_size=opt.batch_size, shuffle=True, pin_memory=True, collate_fn=fakenight_train_highway.collate_fn)
    fakenight_loader_residential = torch.utils.data.DataLoader(fakenight_train_residential, batch_size=opt.batch_size, shuffle=True, pin_memory=True, collate_fn=fakenight_train_residential.collate_fn)

    # Init the dataloader for each cluster in night domain
    night_loader_city_street = torch.utils.data.DataLoader(night_train_city_street, batch_size=opt.batch_size, shuffle=True, pin_memory=True, collate_fn=night_train_city_street.collate_fn)
    night_loader_highway = torch.utils.data.DataLoader(night_train_highway, batch_size=opt.batch_size, shuffle=True, pin_memory=True, collate_fn=night_train_highway.collate_fn)
    night_loader_residential = torch.utils.data.DataLoader(night_train_residential, batch_size=opt.batch_size, shuffle=True, pin_memory=True, collate_fn=night_train_residential.collate_fn)

    n_batches_city_street = len(night_train_city_street) // batch_size  
    n_batches_highway = len(night_train_highway) // batch_size  
    n_batches_residential = len(night_train_residential) // batch_size  

    bce = nn.BCEWithLogitsLoss()

    # initialize the optimizer for encoder, detector and Discriminator
    lr = 0.0001 if opt.pretrained_weights else 0.001
    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    if opt.modularization == 1:
        detector_opt = torch.optim.Adam(detector.parameters(), lr=lr)
        D_opt = torch.optim.Adam(D.parameters(), lr=lr)
    else:  # modularizations 2 and 3
        if opt.modularization == 3:
            detector_1_opt = torch.optim.Adam(detector_1.parameters(), lr=lr)
            detector_2_opt = torch.optim.Adam(detector_2.parameters(), lr=lr)
            detector_3_opt = torch.optim.Adam(detector_3.parameters(), lr=lr)
        D_1_opt = torch.optim.Adam(D_1.parameters(), lr=lr)
        D_2_opt = torch.optim.Adam(D_2.parameters(), lr=lr)
        D_3_opt = torch.optim.Adam(D_3.parameters(), lr=lr)

    max_epoch = opt.epochs
    step_global = 0
    step_city_street = 0
    step_highway = 0
    step_residential = 0

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    best_mAP = 0
    night_set_city_street = iter(night_loader_city_street)
    night_set_highway = iter(night_loader_highway)
    night_set_residential = iter(night_loader_residential)

    scene_dict = {"city_street": {"loader": fakenight_loader_city_street, "sampler": sample_night_city_street, "step": step_city_street}, 
                  "highway": {"loader": fakenight_loader_highway, "sampler": sample_night_highway, "step": step_highway}, 
                  "residential": {"loader": fakenight_loader_residential, "sampler": sample_night_residential, "step": step_residential}}
    
    for epoch in range(max_epoch):
        lmbd = 0.1 * get_lambda(epoch, max_epoch)
        length_of_dataset = len(fakenight_loader_city_street) + len(fakenight_loader_highway) + len(fakenight_loader_residential)
        if opt.modularization == 1:
            Ld_log = 0
        else opt.modularization == 2:
            Ld_1_log, Ld_2_log, Ld_3_log = 0, 0, 0

        for scene in ["city_street", "highway", "residential"]:
            start_time = time.time()
            for batch_i, (paths, src_images, targets) in enumerate(scene_dict[scene]["loader"]):
                _, tgt_images, labels = scene_dict[scene]["sampler"]()
                
                if src_images.shape[0] == opt.batch_size:                
                    files = [path[-21:-4] + ".txt" for path in paths]
                    weights = None
                    if opt.weighting is not None:
                        weights = read_weights(files, opt.weights_folder).to(device)
                        weights = transform_weights(weights)

                    # to make sure that source image and target image have the same img_size at multi_scale-mode
                    batches_done = len(scene_dict[scene]["loader"]) * epoch + batch_i

                    if tgt_images.shape[0] == batch_size:
                        src, targets, tgt = src_images.to(device), targets.to(device), tgt_images.to(device)
                        img_dim = src.shape[2]

                        x = torch.cat([src, tgt], dim=0)  # concatenate the image from source domain and image from target domain
                        if opt.modularization == 1:
                            h_detector = encoder(x)  # pass the concatenated images to the feature extractor
                        elif opt.modularization == 2:
                            l1_source, l2_source, l3_source, f1_source, f2_source, f3_source, outputs_source = encoder(src, img_dim, targets, weights)  # layers, features, outputs
                            Lc = l1_source + l2_source + l3_source
                            f1_target, f2_target, f3_target, outputs_target = encoder(tgt, img_dim)  # pass the night images to the feature extractors
                        else:  # modularization 3
                            h_detector_1, h_detector_2, h_detector_3 = encoder(x)

                        if opt.update_step == "alternating":
                            if opt.modularization == 1:
                                D_img_out = D(h_detector.detach()) 
                                Ld_source = bce(D_img_out[:batch_size], orch.FloatTensor(D_img_out[:batch_size].data.size()).fill_(0).cuda())  
                                Ld_target = bce(D_img_out[batch_size:], torch.FloatTensor(D_img_out[batch_size:].data.size()).fill_(1).cuda()) 
                                Ld = Ld_source + Ld_target
                            
                                D.zero_grad()
                                Ld.backward()
                                D_opt.step()

                            elif opt.modularization == 2:
                                y_1_source, y_2_source, y_3_source = D_1(f1_source.detach()), D_2(f2_source.detach()), D_3(f3_source.detach())
                                y_1_target, y_2_target, y_3_target = D_1(f1_target.detach()), D_2(f2_target.detach()), D_3(f3_starget.detach())
                                Ld_1 = bce(y_1_source, torch.FloatTensor(y_1_source.data.size()).fill_(0).cuda())+bce(y_1_target, torch.FloatTensor(y_1_target.data.size()).fill_(1).cuda())
                                Ld_2 = bce(y_2_source, torch.FloatTensor(y_2_source.data.size()).fill_(0).cuda())+bce(y_2_target, torch.FloatTensor(y_2_target.data.size()).fill_(1).cuda())
                                Ld_3 = bce(y_3_source, torch.FloatTensor(y_3_source.data.size()).fill_(0).cuda())+bce(y_3_target, torch.FloatTensor(y_3_target.data.size()).fill_(1).cuda())
                                
                                D_1.zero_grad(), D_2.zero_grad(), D_3.zero_grad()
                                Ld_1.backward(), Ld_2.backward(), Ld_3.backward()
                                D_1_opt.step(), D_2_opt.step(), D_3_opt.step()

                            else:  # modularization 3
                                D_img_out_1, D_img_out_2, D_img_out_3 = D_1(h_detector_1.detach()), D_2(h_detector_2.detach()), D_3(h_detector_3.detach())
                                Ld_1_source = bce(D_img_out_1[:batch_size], torch.FloatTensor(D_img_out_1[:batch_size].data.size()).fill_(0).cuda())  # Loss from source domain
                                Ld_1_target = bce(D_img_out_1[batch_size:], torch.FloatTensor(D_img_out_1[batch_size:].data.size()).fill_(1).cuda()) # Loss from target

                                Ld_2_source = bce(D_img_out_2[:batch_size], torch.FloatTensor(D_img_out_2[:batch_size].data.size()).fill_(0).cuda())
                                Ld_2_target = bce(D_img_out_2[batch_size:], torch.FloatTensor(D_img_out_2[batch_size:].data.size()).fill_(1).cuda())

                                Ld_3_source = bce(D_img_out_3[:batch_size], torch.FloatTensor(D_img_out_3[:batch_size].data.size()).fill_(0).cuda())
                                Ld_3_target = bce(D_img_out_3[batch_size:], torch.FloatTensor(D_img_out_3[batch_size:].data.size()).fill_(1).cuda())
                                Ld = Ld_1_source + Ld_1_target + Ld_2_source + Ld_2_target + Ld_3_source + Ld_3_target

                                D_1.zero_grad(), D_2.zero_grad(), D_3.zero_grad()
                                Ld.backward()
                                D_1_opt.step(), D_2_opt.step(), D_3_opt.step()
                                            
                        if opt.modularization == 1:
                            Lc, outputs_detector = detector(h_detector[:batch_size], img_dim, targets, weights) 
                            D_img_out = D(h_detector) 
                            Ld_source = bce(D_img_out[:batch_size], torch.FloatTensor(D_img_out[:batch_size].data.size()).fill_(0).cuda())  
                            Ld_target = bce(D_img_out[batch_size:], torch.FloatTensor(D_img_out[batch_size:].data.size()).fill_(1).cuda()) 
                            Ld = Ld_source + Ld_target

                            Ld_log += Ld
                            yolo_layers = detector.yolo_layers

                        elif opt.modularization == 2:
                            y_1_source, y_2_source, y_3_source = D_1(f1_source), D_2(f2_source), D_3(f3_source)
                            y_1_target, y_2_target, y_3_target = D_1(f1_target), D_2(f2_target), D_3(f3_starget)
                            Ld_1 = bce(y_1_source, torch.FloatTensor(y_1_source.data.size()).fill_(0).cuda())+bce(y_1_target, torch.FloatTensor(y_1_target.data.size()).fill_(1).cuda())
                            Ld_2 = bce(y_2_source, torch.FloatTensor(y_2_source.data.size()).fill_(0).cuda())+bce(y_2_target, torch.FloatTensor(y_2_target.data.size()).fill_(1).cuda())
                            Ld_3 = bce(y_3_source, torch.FloatTensor(y_3_source.data.size()).fill_(0).cuda())+bce(y_3_target, torch.FloatTensor(y_3_target.data.size()).fill_(1).cuda())

                            Ld_1_log += Ld_1
                            Ld_2_log += Ld_2
                            Ld_3_log += Ld_3
                            yolo_layers = encoder.yolo_layers

                        else:  # modularization 3 
                            loss_detector_1, outputs_detector_1 = detector_1(h_detector_1[:batch_size], img_dim, targets, weights)  
                            loss_detector_2, outputs_detector_2 = detector_2(h_detector_2[:batch_size], img_dim, targets, weights) 
                            loss_detector_3, outputs_detector_3 = detector_3(h_detector_3[:batch_size], img_dim, targets, weights)
                            Lc = loss_detector_1 + loss_detector_2 + loss_detector_3 
                            
                            D_img_out_1, D_img_out_2, D_img_out_3 = D_1(h_detector_1), D_2(h_detector_2), D_3(h_detector_3)
                            Ld_1_source = bce(D_img_out_1[:batch_size], torch.FloatTensor(D_img_out_1[:batch_size].data.size()).fill_(0).cuda())  
                            Ld_1_target = bce(D_img_out_1[batch_size:], torch.FloatTensor(D_img_out_1[batch_size:].data.size()).fill_(1).cuda()) 
                            Ld_2_source = bce(D_img_out_2[:batch_size], torch.FloatTensor(D_img_out_2[:batch_size].data.size()).fill_(0).cuda())
                            Ld_2_target = bce(D_img_out_2[batch_size:], torch.FloatTensor(D_img_out_2[batch_size:].data.size()).fill_(1).cuda())
                            Ld_3_source = bce(D_img_out_3[:batch_size], torch.FloatTensor(D_img_out_3[:batch_size].data.size()).fill_(0).cuda())
                            Ld_3_target = bce(D_img_out_3[batch_size:], torch.FloatTensor(D_img_out_3[batch_size:].data.size()).fill_(1).cuda())
                            Ld = Ld_1_source + Ld_1_target + Ld_2_source + Ld_2_target + Ld_3_source + Ld_3_target

                            Ld_1_log += Ld_1_source + Ld_1_target
                            Ld_2_log += Ld_2_source + Ld_2_target
                            Ld_3_log += Ld_3_source + Ld_3_target
                            yolo_layers = [detector_1.yolo_layers[0], detector_2.yolo_layers[0], detector_3.yolo_layers[0]]

                        if opt.update_step == "alternating":
                            if opt.modularization == 1:
                                Ltot = Lc - lmbd * Ld
                                D.zero_grad()
                                encoder.zero_grad()
                                detector.zero_grad()
                                Ltot.backward()
                                detector_opt.step()
                                encoder_opt.step()
                            elif opt.modularization == 2:
                                Ltot = Lc - lmbd * (Ld_1 + Ld_2 + Ld_3) / 3.0
                                encoder.zero_grad()
                                D_1.zero_grad(), D_2.zero_grad(), D_3.zero_grad()
                                Ltot.backward()
                                encoder_opt.step()
                            else:
                                total_loss_source = Lc - lmbd * (Ld_1_source + Ld_2_source + Ld_3_source) / 3.0 
                                total_loss_target = - lmbd * (Ld_1_target + Ld_2_target + Ld_3_target) / 3.0
                                Ltot = total_loss_source + total_loss_target
                                encoder.zero_grad()
                                detector_1.zero_grad(), detector_2.zero_grad(),detector_3.zero_grad()
                                D_1.zero_grad(), D_2.zero_grad(), D_3.zero_grad()
                                Ltot.backward()
                                encoder_opt.step()
                                detector_1_opt.step(), detector_2_opt.step(), detector_3_opt.step()

                        else:  # joint
                            if opt.modularization == 1:
                                Ltot = Lc + lmbd * Ld
                                Ltot.backward()
                                if batches_done % opt.gradient_accumulations:
                                    encoder_opt.step()
                                    detector_opt.step()
                                    D_opt.step()
                                    encoder_opt.zero_grad()
                                    detector_opt.zero_grad()
                                    D_opt.zero_grad()
                            elif opt.modularization == 2:
                                Ltot = Lc + lmbd * (Ld_1 + Ld_2 + Ld_3) / 3.0
                                Ltot.backward()
                                if batches_done % opt.gradient_accumulations:
                                    D_1_opt.step(), D_2_opt.step(), D_3_opt.step()
                                    encoder_opt.step()
                                    encoder_opt.zero_grad()
                                    D_1_opt.zero_grad(), D_2_opt.zero_grad(), D_3_opt.zero_grad()
                            else:  # modularization 3
                                total_loss_source = Lc + lmbd * (Ld_1_source + Ld_2_source + Ld_3_source) / 3.0 
                                total_loss_target = lmbd * (Ld_1_target + Ld_2_target + Ld_3_target) / 3.0
                                Ltot = total_loss_source + total_loss_target
                                Ltot.backward()
                                if batches_done % opt.gradient_accumulations:
                                    encoder_opt.step()
                                    detector_1_opt.step(), detector_2_opt.step(), detector_3_opt.step()
                                    D_1_opt.step(), D_2_opt.step(), D_3_opt.step()
                                    encoder_opt.zero_grad()
                                    detector_1_opt.zero_grad(), detector_2_opt.zero_grad(), detector_3_opt.zero_grad()
                                    D_1_opt.zero_grad(), D_2_opt.zero_grad(), D_3_opt.zero_grad()
                                        
                        scene_dict[scene]["step"] += 1

                        if scene_dict[scene]["step"] % 100 == 0:
                        # --------------------------------
                        #   Log progress every 100 iters
                        # --------------------------------
                            log_str = "\n---- cluster:%s [Epoch %d/%d, Batch %d/%d] ----\n" % (scene, epoch, opt.epochs, batch_i, len(fakenight_loader_city_street)) 

                            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(yolo_layers))]]] 

                            # Log metrics at each YOLO layer
                            for i, metric in enumerate(metrics):
                                formats = {m: "%.6f" for m in metrics}
                                formats["grid_size"] = "%2d"
                                formats["cls_acc"] = "%.2f%%"
                                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in yolo_layers]
                                metric_table += [[metric, *row_metrics]] 

                            log_str += AsciiTable(metric_table).table
                            log_str += f"\nTotal loss: {Lc.item()}"

                            # Determine approximate time left for epoch
                            epoch_batches_left = len(scene_dict[scene]["loader"]) - (batch_i + 1)
                            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))  
                            log_str += f"\n---- ETA for {scene}: {time_left}"
                            print(log_str) 
                            print("Ld: ", Ld_source + Ld_target)

        if opt.modularization == 1:
            model_list = [encoder, detector]
        elif opt.modularization == 2:
            model_list = [encoder]
        else:
            model_list = [encoder, detector_1, detector_2, detector_3]

        if epoch % opt.evaluation_interval == 0:
            
            print("\n---- Evaluating Model on the daytime domain ----")
            precision, recall, AP, f1, ap_class = evaluate(
                model_list,
                path=data_config["valid_daytime"],
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),  #
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger_daytime.list_of_scalars_summary(evaluation_metrics, epoch)  

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

            print("\n---- Evaluating Model on the true night domain ----")
            precision, recall, AP, f1, ap_class = evaluate(
                model_list,
                path=data_config["valid_night"],
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),  
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger_night.list_of_scalars_summary(evaluation_metrics, epoch)  

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            
            print("\n---- Evaluating Model on the fake night domain ----")
            precision, recall, AP, f1, ap_class = evaluate(
                model_list,
                path=data_config["valid_fakenight"],
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),  
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger_fakenight.list_of_scalars_summary(evaluation_metrics, epoch)  

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

            #write the loss of discriminator in to tensorboard

            Ld_log = Ld_log/length_of_dataset
            train_writer_D.add_scalar('Loss_D', Ld_log, epoch)

            for model in model_list:
                model.train()
        
        if epoch % opt.checkpoint_interval == 0:
            torch.save(encoder.state_dict(), "checkpoints-DANN/{}/encoder_{}.pth".format(config_name, epoch))
            if opt.modularization == 1:
                torch.save(detector.state_dict(), "checkpoints-DANN/{}/detector_{}.pth".format(config_name, epoch))
                torch.save(D.state_dict(), "checkpoints-DANN/{}/D_{}.pth".format(config_name, epoch))
            else:
                if opt.modularization == 3:
                    torch.save(detector_1.state_dict(), "checkpoints-DANN/{}/detector1_{}.pth".format(config_name, epoch))
                    torch.save(detector_2.state_dict(), "checkpoints-DANN/{}/detector2_{}.pth".format(config_name, epoch))
                    torch.save(detector_3.state_dict(), "checkpoints-DANN/{}/detector3_{}.pth".format(config_name, epoch))
                torch.save(D_1.state_dict(), "checkpoints-DANN/{}/D1_{}.pth".format(config_name, epoch))
                torch.save(D_2.state_dict(), "checkpoints-DANN/{}/D2_{}.pth".format(config_name, epoch))
                torch.save(D_3.state_dict(), "checkpoints-DANN/{}/D3_{}.pth".format(config_name, epoch))

        if AP.mean() > best_mAP:
            best_epoch = epoch
            best_mAP = AP.mean()
            best_model_weights_encoder = copy.deepcopy(encoder.state_dict())
            if opt.modularization == 1:
                best_model_weights_detector = copy.deepcopy(detector.state_dict())
                best_model_weights_D = copy.deepcopy(D.state_dict())
            else:
                if opt.modularization == 3:
                    best_model_weights_detector_1 = copy.deepcopy(detector_1.state_dict())
                    best_model_weights_detector_2 = copy.deepcopy(detector_2.state_dict())
                    best_model_weights_detector_3 = copy.deepcopy(detector_3.state_dict())
                best_model_weights_D_1 = copy.deepcopy(D_1.state_dict())
                best_model_weights_D_2 = copy.deepcopy(D_2.state_dict())
                best_model_weights_D_3 = copy.deepcopy(D_3.state_dict())

    torch.save(best_model_weights_encoder, "checkpoints-DANN/{}/best_encoder_{}.pth".format(config_name, best_epoch))
    if opt.modularization == 1:
        torch.save(best_model_weights_detector, "checkpoints-DANN/{}/best_detector_{}.pth".format(config_name, best_epoch))
        torch.save(best_model_weights_D.state_dict(), "checkpoints-DANN/{}/best_D_{}.pth".format(config_name, best_epoch))
    else:
        if opt.modularization == 3:
            torch.save(best_model_weights_detector_1, "checkpoints-DANN/{}/best_detector1_{}.pth".format(config_name, best_epoch))
            torch.save(best_model_weights_detector_2, "checkpoints-DANN/{}/best_detector2_{}.pth".format(config_name, best_epoch))
            torch.save(best_model_weights_detector_3, "checkpoints-DANN/{}/best_detector3_{}.pth".format(config_name, best_epoch))
        torch.save(best_model_weights_D_1, "checkpoints-DANN/{}/best_D1_{}.pth".format(config_name, best_epoch))
        torch.save(best_model_weights_D_2, "checkpoints-DANN/{}/best_D2_{}.pth".format(config_name, best_epoch))
        torch.save(best_model_weights_D_3, "checkpoints-DANN/{}/best_D3_{}.pth".format(config_name, best_epoch))
