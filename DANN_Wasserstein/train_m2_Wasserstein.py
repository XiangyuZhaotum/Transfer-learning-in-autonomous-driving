from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import Function

import warnings

import random
import copy
from torch.autograd import Function
import torch.nn.utils.spectral_norm as spectral_norm
import torch.autograd as autograd

def calc_gradient_penalty(netD, source_data, target_data, LAMBDA = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(source_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * source_data + ((1 - alpha) * target_data)

    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=(1, 2, 3)) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)


class Discriminator_large(nn.Module):

    def __init__(self, input_nc, ndf=64):
        super(Discriminator_large, self).__init__()

        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2, inplace=True)

    def forward(self, x):
        #x = grad_reverse(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x

class Discriminator_small(nn.Module):

    def __init__(self, input_nc, ndf=64):
        super(Discriminator_small, self).__init__()

        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2, inplace=True)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x

def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1 + np.exp(-10. * p)) - 1.


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

def read_weights(files):
    weights_folder = "./data/custom/weights/train/"
    weight_list = []
    for file in files:
        weight_path = weights_folder + file
        # print(weight_path)
        with open(weight_path, 'r', encoding='utf-8') as f:
            for line in f:
                weight = float(line)
                weight_list.append(weight)
    weights = torch.Tensor([weight_list]).squeeze()
    return weights

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom_DANN_cluster.data", help="path to data config file")
    #parser.add_argument("--data_config", type=str, default="config/custom_DANN.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str,default=None,help="if specified starts from checkpoint model")
    #parser.add_argument("--pretrained_weights", type=str, default="checkpoints/yolov3_tiny_fake_ckpt_36.pth",help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument('--input_nc', type=int, default=30,help='number of channels of input data')  # #input channels for Discriminator
    parser.add_argument('--n_critic', type=int, default=5, help='train the encoder every n_critic iterations')
    opt = parser.parse_args()
    print(opt)

### Training Code
    os.makedirs("logs_DANN_Discriminator_small/Wasserstein", exist_ok=True)  # 存储DANN训练过程中的small discriminator loss
    os.makedirs("logs_DANN_Discriminator_middle/Wasserstein", exist_ok=True)  # 存储DANN训练过程中的middle discriminator loss
    os.makedirs("logs_DANN_Discriminator_large/Wasserstein", exist_ok=True)  # 存储DANN训练过程中的large discriminator loss
    os.makedirs("logs_DANN_source/Wasserstein", exist_ok=True)  # 存储YOLOv3 关于source domain的训练过程
    os.makedirs("logs_DANN_target/Wasserstein", exist_ok=True)  # 存储YOLOv3 关于target domain的训练过程
    os.makedirs("logs_DANN_others/Wasserstein", exist_ok=True)
    os.makedirs("model_parameters_DANN/Wasserstein", exist_ok=True)

    # initialize the logger
    logger_fake_night = Logger("logs_DANN_source/Wasserstein")
    logger_night = Logger("logs_DANN_target/Wasserstein")
    logger_daytime = Logger("logs_DANN_others/Wasserstein")
    train_writer_small = SummaryWriter('./logs_DANN_Discriminator_small/Wasserstein')
    train_writer_middle = SummaryWriter('./logs_DANN_Discriminator_middle/Wasserstein')
    train_writer_large = SummaryWriter('./logs_DANN_Discriminator_large/Wasserstein')
    train_writer_Wasserstein_metric = SummaryWriter('./Wasserstein_metric')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    one = torch.FloatTensor([1]).squeeze().to(device)
    mone = (one * -1).to(device)

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)
    D_small = Discriminator_large(opt.input_nc).to(device)
    D_small.apply(weights_init_normal)
    D_middle = Discriminator_large(opt.input_nc).to(device)
    D_middle.apply(weights_init_normal)
    D_large = Discriminator_large(opt.input_nc).to(device)
    D_large.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_night_path_city_street = data_config["train_night_city_street"]  # path to the file indicating where the training dataset is
    train_night_path_highway = data_config["train_night_highway"]
    train_night_path_residential= data_config["train_night_residential"]
    valid_night_path = data_config["valid_night"]  # path to the file indicating where the validation dataset is

    train_fake_night_path_city_street = data_config["train_fake_night_city_street"]
    train_fake_night_path_highway = data_config["train_fake_night_highway"]
    train_fake_night_path_residential = data_config["train_fake_night_residential"]
    valid_fake_night_path = data_config["valid_fake_night"]
    
    valid_daytime_path = data_config["valid_daytime"]

    class_names = load_classes(data_config["names"])

     #instantiate the dataset of clusters of scenes
    night_train_city_street = ListDataset(train_night_path_city_street, augment=True, multiscale=opt.multiscale_training)
    night_train_highway = ListDataset(train_night_path_highway, augment=True, multiscale=opt.multiscale_training)
    night_train_residential = ListDataset(train_night_path_residential, augment=True, multiscale=opt.multiscale_training)

    fake_night_train_city_street = ListDataset(train_fake_night_path_city_street, augment=True, multiscale=opt.multiscale_training)
    fake_night_train_highway = ListDataset(train_fake_night_path_highway, augment=True, multiscale=opt.multiscale_training)
    fake_night_train_residential = ListDataset(train_fake_night_path_residential, augment=True, multiscale=opt.multiscale_training)

    batch_size = opt.batch_size

    #Instantiate the dataloader for each cluster in fake_night domain
    fake_night_loader_city_street = torch.utils.data.DataLoader(
        fake_night_train_city_street,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=fake_night_train_city_street.collate_fn,
    )

    fake_night_loader_highway = torch.utils.data.DataLoader(
        fake_night_train_highway,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=fake_night_train_highway.collate_fn,
    )

    fake_night_loader_residential = torch.utils.data.DataLoader(
        fake_night_train_residential,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=fake_night_train_residential.collate_fn,
    )


    #Instantiate the dataloader for each cluster in night domain
    night_loader_city_street = torch.utils.data.DataLoader(
        night_train_city_street,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=night_train_city_street.collate_fn,
    )

    night_loader_highway = torch.utils.data.DataLoader(
        night_train_highway,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=night_train_highway.collate_fn,
    )

    night_loader_residential = torch.utils.data.DataLoader(
        night_train_residential,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=night_train_residential.collate_fn,
    )

    #bce = nn.BCEWithLogitsLoss()

    # initialize the optimizer for yolo model and discriminator
    model_opt = torch.optim.Adam(model.parameters())
    D_small_opt = torch.optim.Adam(params=D_small.parameters())
    D_middle_opt = torch.optim.Adam(params=D_middle.parameters())
    D_large_opt = torch.optim.Adam(params=D_large.parameters())

    max_epoch = opt.epochs
    step_global = 0
    step_city_street = 0
    step_highway = 0
    step_residential = 0
    
    n_batches_city_street = len(night_train_city_street) // batch_size  # 做除法取整数
    n_batches_highway = len(night_train_highway) // batch_size  # 做除法取整数
    n_batches_residential = len(night_train_residential) // batch_size  # 做除法取整数

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
    
    for epoch in range(opt.epochs):
        lamda = 0.1 * get_lambda(epoch,max_epoch)
        length_of_dataset = len(fake_night_loader_city_street) + len(fake_night_loader_highway) + len(fake_night_loader_residential)
        loss_d_small_log = 0
        loss_d_middle_log = 0
        loss_d_large_log = 0
        Wasserstein_metric_log = 0
        # loop over the cluster of the scene "city_street"
        start_time = time.time()
        for batch_i, (paths, src_images, targets) in enumerate(fake_night_loader_city_street):
            batches_done = len(fake_night_loader_city_street) * epoch + batch_i
            
            _, tgt_images, _ = sample_night_city_street()
            if tgt_images.shape[0]==batch_size and src_images.shape[0]==batch_size:
                # Training Discriminator
                files = [path[-21:-4] + ".txt" for path in paths]
                weights = read_weights(files).to(device)
                weights = 1.0/(2*torch.abs((weights-1))+1e-16)
                weights = 2*F.sigmoid(weights)-1
                print(f"weights shape{weights.shape}, weights_0 {weights[0]},weights_device{weights.device}")
                
                src, targets, tgt = src_images.to(device), targets.to(device), tgt_images.to(device)

                for p in D_small.parameters():
                    p.requires_grad = True
                for p in D_middle.parameters():
                    p.requires_grad = True
                for p in D_large.parameters():
                    p.requires_grad = True
                
                D_small_opt.zero_grad()
                D_middle_opt.zero_grad()
                D_large_opt.zero_grad()
                    
                img_dim=src.shape[2]
                layer_loss_small_day, layer_loss_middle_day, layer_loss_large_day, feature_small_day, feature_middle_day, feature_large_day, outputs_day = model(src, img_dim, targets, weights)
                feature_small_night, feature_middle_night, feature_large_night, outputs_night = model(tgt,img_dim)
                
                y_small_day = D_small(feature_small_day)
                y_middle_day = D_middle(feature_middle_day)
                y_large_day = D_large(feature_large_day)
                y_small_night = D_small(feature_small_night)
                y_middle_night = D_middle(feature_middle_night)
                y_large_night = D_large(feature_large_night)
                
                D_source = torch.mean(y_small_day) + torch.mean(y_middle_day) + torch.mean(y_large_day)
                D_target = torch.mean(y_small_night) + torch.mean(y_middle_night) + torch.mean(y_large_night)
                
                gradient_penalty_D_small = calc_gradient_penalty(D_small, feature_small_day, feature_small_night)
                gradient_penalty_D_middle = calc_gradient_penalty(D_middle, feature_middle_day, feature_middle_night)
                gradient_penalty_D_large = calc_gradient_penalty(D_large, feature_large_day, feature_large_night)
                gradient_penalty = gradient_penalty_D_small + gradient_penalty_D_middle + gradient_penalty_D_large
                critic_cost = D_source - D_target +gradient_penalty
                critic_cost.backward()
                # Loss of Discriminators
                Ld_small = torch.mean(y_small_night) - torch.mean(y_small_day)
                Ld_middle = torch.mean(y_middle_night) - torch.mean(y_middle_day)
                Ld_large = torch.mean(y_large_night) - torch.mean(y_large_day)
                Wasserstein_metric = D_target - D_source
                
                Wasserstein_metric_log += Wasserstein_metric
                loss_d_small_log += Ld_small
                loss_d_middle_log += Ld_middle
                loss_d_large_log += Ld_large
               
                D_small_opt.step()
                D_middle_opt.step()
                D_large_opt.step()
                
                if batch_i % opt.n_critic == 0:
                    layer_loss_small_day, layer_loss_middle_day, layer_loss_large_day, feature_small_day, feature_middle_day, feature_large_day, outputs_day = model(src, img_dim, targets, weights)            
                    feature_small_night, feature_middle_night, feature_large_night, outputs_night = model(tgt,img_dim)
                    y_small_day = D_small(feature_small_day)
                    y_middle_day = D_middle(feature_middle_day)
                    y_large_day = D_large(feature_large_day)
                    y_small_night = D_small(feature_small_night)
                    y_middle_night = D_middle(feature_middle_night)
                    y_large_night = D_large(feature_large_night)
                    
                    for p in D_small.parameters():
                        p.requires_grad = False
                    for p in D_middle.parameters():
                        p.requires_grad = False
                    for p in D_large.parameters():
                        p.requires_grad = False
                    
                    model_opt.zero_grad()
                    D_source = torch.mean(y_small_day) + torch.mean(y_middle_day) + torch.mean(y_large_day)
                    D_target = torch.mean(y_small_night) + torch.mean(y_middle_night) + torch.mean(y_large_night)
                    detection_loss = layer_loss_small_day + layer_loss_middle_day + layer_loss_large_day
                    loss_e = detection_loss + lamda * (D_target - D_source)
                    loss_e.backward()
                    model_opt.step()
                    print(
                            "[Epoch %d/%d] [Batch %d/%d] [small loss: %f] [middle loss: %f] [large loss: %f] [Wasserstein_metric: %f]"
                            % (epoch, opt.epochs, batches_done, n_batches_city_street,Ld_small.item(), Ld_middle.item(), Ld_large.item(), Wasserstein_metric))

        # loop over the cluster of the scene "highway"
        start_time = time.time()
        for batch_i, (paths, src_images, targets) in enumerate(fake_night_loader_highway):
            batches_done = len(fake_night_loader_highway) * epoch + batch_i
            
            _, tgt_images, _ = sample_night_highway()
            if tgt_images.shape[0]==batch_size and src_images.shape[0]==batch_size:
                # Training Discriminator
                files = [path[-21:-4] + ".txt" for path in paths]
                weights = read_weights(files).to(device)
                weights = 1.0/(2*torch.abs((weights-1))+1e-16)
                weights = 2*F.sigmoid(weights)-1
                print(f"weights shape{weights.shape}, weights_0 {weights[0]},weights_device{weights.device}")
                
                src, targets, tgt = src_images.to(device), targets.to(device), tgt_images.to(device)

                for p in D_small.parameters():
                    p.requires_grad = True
                for p in D_middle.parameters():
                    p.requires_grad = True
                for p in D_large.parameters():
                    p.requires_grad = True
                
                D_small_opt.zero_grad()
                D_middle_opt.zero_grad()
                D_large_opt.zero_grad()
                    
                img_dim=src.shape[2]
                layer_loss_small_day, layer_loss_middle_day, layer_loss_large_day, feature_small_day, feature_middle_day, feature_large_day, outputs_day = model(src, img_dim, targets, weights)
                feature_small_night, feature_middle_night, feature_large_night, outputs_night = model(tgt,img_dim)
                
                y_small_day = D_small(feature_small_day)
                y_middle_day = D_middle(feature_middle_day)
                y_large_day = D_large(feature_large_day)
                y_small_night = D_small(feature_small_night)
                y_middle_night = D_middle(feature_middle_night)
                y_large_night = D_large(feature_large_night)
                
                D_source = torch.mean(y_small_day) + torch.mean(y_middle_day) + torch.mean(y_large_day)
                D_target = torch.mean(y_small_night) + torch.mean(y_middle_night) + torch.mean(y_large_night)
                
                gradient_penalty_D_small = calc_gradient_penalty(D_small, feature_small_day, feature_small_night)
                gradient_penalty_D_middle = calc_gradient_penalty(D_middle, feature_middle_day, feature_middle_night)
                gradient_penalty_D_large = calc_gradient_penalty(D_large, feature_large_day, feature_large_night)
                gradient_penalty = gradient_penalty_D_small + gradient_penalty_D_middle + gradient_penalty_D_large
                critic_cost = D_source - D_target +gradient_penalty
                critic_cost.backward()
                # Loss of Discriminators
                Ld_small = torch.mean(y_small_night) - torch.mean(y_small_day)
                Ld_middle = torch.mean(y_middle_night) - torch.mean(y_middle_day)
                Ld_large = torch.mean(y_large_night) - torch.mean(y_large_day)
                Wasserstein_metric = D_target - D_source
                
                Wasserstein_metric_log += Wasserstein_metric
                loss_d_small_log += Ld_small
                loss_d_middle_log += Ld_middle
                loss_d_large_log += Ld_large
               
                D_small_opt.step()
                D_middle_opt.step()
                D_large_opt.step()
                
                if batch_i % opt.n_critic == 0:
                    layer_loss_small_day, layer_loss_middle_day, layer_loss_large_day, feature_small_day, feature_middle_day, feature_large_day, outputs_day = model(src, img_dim, targets, weights)            
                    feature_small_night, feature_middle_night, feature_large_night, outputs_night = model(tgt,img_dim)
                    y_small_day = D_small(feature_small_day)
                    y_middle_day = D_middle(feature_middle_day)
                    y_large_day = D_large(feature_large_day)
                    y_small_night = D_small(feature_small_night)
                    y_middle_night = D_middle(feature_middle_night)
                    y_large_night = D_large(feature_large_night)
                    
                    for p in D_small.parameters():
                        p.requires_grad = False
                    for p in D_middle.parameters():
                        p.requires_grad = False
                    for p in D_large.parameters():
                        p.requires_grad = False
                    
                    model_opt.zero_grad()
                    D_source = torch.mean(y_small_day) + torch.mean(y_middle_day) + torch.mean(y_large_day)
                    D_target = torch.mean(y_small_night) + torch.mean(y_middle_night) + torch.mean(y_large_night)
                    detection_loss = layer_loss_small_day + layer_loss_middle_day + layer_loss_large_day
                    loss_e = detection_loss + lamda * (D_target - D_source)
                    loss_e.backward()
                    model_opt.step()
                    
                    print(
                            "[Epoch %d/%d] [Batch %d/%d] [small loss: %f] [middle loss: %f] [large loss: %f] [Wasserstein_metric: %f]"
                            % (epoch, opt.epochs, batches_done, n_batches_highway,Ld_small.item(), Ld_middle.item(), Ld_large.item(), Wasserstein_metric))
                
                
                
        # loop over the cluster of the scene "residential"
        start_time = time.time()
        for batch_i, (paths, src_images, targets) in enumerate(fake_night_loader_residential):
            batches_done = len(fake_night_loader_residential) * epoch + batch_i
            
            _, tgt_images, _ = sample_night_residential()
            if tgt_images.shape[0]==batch_size and src_images.shape[0]==batch_size:
                # Training Discriminator
                files = [path[-21:-4] + ".txt" for path in paths]
                weights = read_weights(files).to(device)
                weights = 1.0/(2*torch.abs((weights-1))+1e-16)
                weights = 2*F.sigmoid(weights)-1
                print(f"weights shape{weights.shape}, weights_0 {weights[0]},weights_device{weights.device}")
                
                src, targets, tgt = src_images.to(device), targets.to(device), tgt_images.to(device)

                for p in D_small.parameters():
                    p.requires_grad = True
                for p in D_middle.parameters():
                    p.requires_grad = True
                for p in D_large.parameters():
                    p.requires_grad = True
                
                D_small_opt.zero_grad()
                D_middle_opt.zero_grad()
                D_large_opt.zero_grad()
                    
                img_dim=src.shape[2]
                layer_loss_small_day, layer_loss_middle_day, layer_loss_large_day, feature_small_day, feature_middle_day, feature_large_day, outputs_day = model(src, img_dim, targets, weights)
                feature_small_night, feature_middle_night, feature_large_night, outputs_night = model(tgt,img_dim)
                
                y_small_day = D_small(feature_small_day)
                y_middle_day = D_middle(feature_middle_day)
                y_large_day = D_large(feature_large_day)
                y_small_night = D_small(feature_small_night)
                y_middle_night = D_middle(feature_middle_night)
                y_large_night = D_large(feature_large_night)
                
                D_source = torch.mean(y_small_day) + torch.mean(y_middle_day) + torch.mean(y_large_day)
                D_target = torch.mean(y_small_night) + torch.mean(y_middle_night) + torch.mean(y_large_night)
                
                gradient_penalty_D_small = calc_gradient_penalty(D_small, feature_small_day, feature_small_night)
                gradient_penalty_D_middle = calc_gradient_penalty(D_middle, feature_middle_day, feature_middle_night)
                gradient_penalty_D_large = calc_gradient_penalty(D_large, feature_large_day, feature_large_night)
                gradient_penalty = gradient_penalty_D_small + gradient_penalty_D_middle + gradient_penalty_D_large
                critic_cost = D_source - D_target +gradient_penalty
                critic_cost.backward()
                # Loss of Discriminators
                Ld_small = torch.mean(y_small_night) - torch.mean(y_small_day)
                Ld_middle = torch.mean(y_middle_night) - torch.mean(y_middle_day)
                Ld_large = torch.mean(y_large_night) - torch.mean(y_large_day)
                Wasserstein_metric = D_target - D_source
                
                Wasserstein_metric_log += Wasserstein_metric
                loss_d_small_log += Ld_small
                loss_d_middle_log += Ld_middle
                loss_d_large_log += Ld_large
               
                D_small_opt.step()
                D_middle_opt.step()
                D_large_opt.step()
                
                if batch_i % opt.n_critic == 0:
                    layer_loss_small_day, layer_loss_middle_day, layer_loss_large_day, feature_small_day, feature_middle_day, feature_large_day, outputs_day = model(src, img_dim, targets, weights)            
                    feature_small_night, feature_middle_night, feature_large_night, outputs_night = model(tgt,img_dim)
                    y_small_day = D_small(feature_small_day)
                    y_middle_day = D_middle(feature_middle_day)
                    y_large_day = D_large(feature_large_day)
                    y_small_night = D_small(feature_small_night)
                    y_middle_night = D_middle(feature_middle_night)
                    y_large_night = D_large(feature_large_night)
                    
                    for p in D_small.parameters():
                        p.requires_grad = False
                    for p in D_middle.parameters():
                        p.requires_grad = False
                    for p in D_large.parameters():
                        p.requires_grad = False
                    
                    model_opt.zero_grad()
                    D_source = torch.mean(y_small_day) + torch.mean(y_middle_day) + torch.mean(y_large_day)
                    D_target = torch.mean(y_small_night) + torch.mean(y_middle_night) + torch.mean(y_large_night)
                    detection_loss = layer_loss_small_day + layer_loss_middle_day + layer_loss_large_day
                    loss_e = detection_loss + lamda * (D_target - D_source)
                    loss_e.backward()
                    model_opt.step()
                
                    print(
                            "[Epoch %d/%d] [Batch %d/%d] [small loss: %f] [middle loss: %f] [large loss: %f] [Wasserstein_metric: %f]"
                            % (epoch, opt.epochs, batches_done, n_batches_residential,Ld_small.item(), Ld_middle.item(), Ld_large.item(), Wasserstein_metric))
                

        if epoch % opt.evaluation_interval == 0:


            print("\n---- Evaluating Model on the daytime ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_daytime_path,
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
            logger_daytime.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
            
            print("\n---- Evaluating Model on the real night ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_night_path,
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
            
            print("\n---- Evaluating Model on the fake night ----")
            # Evaluate the model on the target set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_fake_night_path,
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
            logger_fake_night.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

            loss_d_small_log = loss_d_small_log/(length_of_dataset)
            loss_d_middle_log = loss_d_middle_log/(length_of_dataset)
            loss_d_large_log = loss_d_large_log/(length_of_dataset)
            Wasserstein_metric_log = Wasserstein_metric_log/(length_of_dataset)
            train_writer_small.add_scalar('Loss_D_small', loss_d_small_log.item(), epoch)
            train_writer_middle.add_scalar('Loss_D_middle', loss_d_middle_log.item(), epoch)
            train_writer_large.add_scalar('Loss_D_large', loss_d_large_log.item(), epoch)
            train_writer_Wasserstein_metric.add_scalar('Wasserstein_metric', Wasserstein_metric_log.item(), epoch)

            model.train() # set to training mode after every evaluation
            D_small.train()
            D_middle.train()
            D_large.train()

        if AP.mean() > best_mAP:
            best_epoch = epoch
            best_mAP = AP.mean()
            best_model_weights = copy.deepcopy(model.state_dict())
            best_model_weights_D_small = copy.deepcopy(D_small.state_dict())
            best_model_weights_D_middle = copy.deepcopy(D_middle.state_dict())
            best_model_weights_D_large = copy.deepcopy(D_large.state_dict())

            torch.save(best_model_weights, f"model_parameters_DANN/Wasserstein/model_weighted_cluster.pth")
            torch.save(best_model_weights_D_small, f"model_parameters_DANN/Wasserstein/D_small.pth" )
            torch.save(best_model_weights_D_middle, f"model_parameters_DANN/Wasserstein/D_middle.pth")
            torch.save(best_model_weights_D_large, f"model_parameters_DANN/Wasserstein/D_large.pth")
