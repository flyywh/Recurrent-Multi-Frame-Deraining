#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, copy, pickle
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

### custom lib
import networks
import datasets_multiple
import datasets_multiple_haze
import utils
from pytorch_ssim import *

from matplotlib import pyplot as plt
from torchvision.utils import save_image
from discriminator import *
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fast Blind Video Temporal Consistency")

    ### model options
    parser.add_argument('-multi_model',     type=str,     default="MultiShareTransformNet",  help='MultiShareTransformNet')
    parser.add_argument('-single_model',    type=str,     default="SingleShareTransformNet", help='SingleShareTransformNet')

    parser.add_argument('-nf',              type=int,     default=32,               help='#Channels in conv layer')
    parser.add_argument('-blocks',          type=int,     default=5,                help='#ResBlocks') 
    parser.add_argument('-norm',            type=str,     default='IN',             choices=["BN", "IN", "none"],   help='normalization layer')
    parser.add_argument('-model_name',      type=str,     default='none',           help='path to save model')

    ### dataset options
    parser.add_argument('-datasets_tasks',  type=str,     default='W3_D1_C1_I1',    help='dataset-task pairs list')
    parser.add_argument('-data_dir',        type=str,     default='data_heavy',     help='path to data folder')
    parser.add_argument('-data_haze_dir',   type=str,     default='data_complex',      help='path to data folder')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to lists folder')
    parser.add_argument('-checkpoint_dir',  type=str,     default='checkpoints',    help='path to checkpoint folder')
    parser.add_argument('-crop_size',       type=int,     default=128,               help='patch size')
    parser.add_argument('-geometry_aug',    type=int,     default=1,                help='geometry augmentation (rotation, scaling, flipping)')
    parser.add_argument('-order_aug',       type=int,     default=1,                help='temporal ordering augmentation')
    parser.add_argument('-scale_min',       type=float,   default=0.5,              help='min scaling factor')
    parser.add_argument('-scale_max',       type=float,   default=2.0,              help='max scaling factor')
    parser.add_argument('-sample_frames',   type=int,     default=7,                help='#frames for training')
        
    ### loss optinos
    parser.add_argument('-alpha',           type=float,   default=50.0,             help='alpha for computing visibility mask')
    parser.add_argument('-loss',            type=str,     default="L2",             help="optimizer [Options: SGD, ADAM]")
    parser.add_argument('-w_ST',            type=float,   default=0,                help='weight for short-term temporal loss')
    parser.add_argument('-w_LT',            type=float,   default=0,                help='weight for long-term temporal loss')
    parser.add_argument('-w_VGG',           type=float,   default=0,                help='weight for VGG perceptual loss')
    parser.add_argument('-w_RECT',          type=float,   default=1,                help='weight for reconstruction loss')
    parser.add_argument('-VGGLayers',       type=str,     default="12",              help="VGG layers for perceptual loss, combinations of 1, 2, 3, 4")

    ### training options
    parser.add_argument('-solver',          type=str,     default="ADAM",           choices=["SGD", "ADAIM"],   help="optimizer")
    parser.add_argument('-momentum',        type=float,   default=0.9,              help='momentum for SGD')
    parser.add_argument('-beta1',           type=float,   default=0.9,              help='beta1 for ADAM')
    parser.add_argument('-beta2',           type=float,   default=0.999,            help='beta2 for ADAM')
    parser.add_argument('-weight_decay',    type=float,   default=0,                help='weight decay')
    parser.add_argument('-batch_size',      type=int,     default=2,                help='training batch size')
    parser.add_argument('-train_epoch_size',type=int,     default=1000,             help='train epoch size')
    parser.add_argument('-valid_epoch_size',type=int,     default=100,              help='valid epoch size')
    parser.add_argument('-epoch_max',       type=int,     default=100,              help='max #epochs')

    ### learning rate options
    parser.add_argument('-lr_init',         type=float,   default=1e-4,             help='initial learning Rate')
    parser.add_argument('-lr_offset',       type=int,     default=20,               help='epoch to start learning rate drop [-1 = no drop]')
    parser.add_argument('-lr_step',         type=int,     default=20,               help='step size (epoch) to drop learning rate')
    parser.add_argument('-lr_drop',         type=float,   default=0.5,              help='learning rate drop ratio')
    parser.add_argument('-lr_min_m',        type=float,   default=0.1,              help='minimal learning Rate multiplier (lr >= lr_init * lr_min)')

    ### other options
    parser.add_argument('-seed',            type=int,     default=9487,             help='random seed to use')
    parser.add_argument('-threads',         type=int,     default=8,                help='number of threads for data loader to use')
    parser.add_argument('-suffix',          type=str,     default='',               help='name suffix')
    parser.add_argument('-gpu',             type=int,     default=0,                help='gpu device id')
    parser.add_argument('-cpu',             action='store_true',                    help='use cpu?')
    
    opts = parser.parse_args()

    ### adjust options
    opts.cuda = (opts.cpu != True)
    opts.lr_min = opts.lr_init * opts.lr_min_m
    
    ### default model name
    if opts.model_name == 'none':
        
        opts.model_name = "%s_B%d_nf%d_%s" %(opts.multi_model, opts.blocks, opts.nf, opts.norm)

        opts.model_name = "%s_T%d_%s_pw%d_%sLoss_a%s_wST%s_wHT%s_wVGG%s_L%s_%s_lr%s_off%d_step%d_drop%s_min%s_es%d_bs%d" \
                %(opts.model_name, opts.sample_frames, \
                  opts.datasets_tasks, opts.crop_size, opts.loss, str(opts.alpha), \
                  str(opts.w_ST), str(opts.w_LT), str(opts.w_VGG), opts.VGGLayers, \
                  opts.solver, str(opts.lr_init), opts.lr_offset, opts.lr_step, str(opts.lr_drop), str(opts.lr_min), \
                  opts.train_epoch_size, opts.batch_size)

    ### check VGG layers
    opts.VGGLayers = [int(layer) for layer in list(opts.VGGLayers)]
    opts.VGGLayers.sort()

    if opts.VGGLayers[0] < 1 or opts.VGGLayers[-1] > 4:
        raise Exception("Only support VGG Loss on Layers 1 ~ 4")

    opts.VGGLayers = [layer - 1 for layer in list(opts.VGGLayers)] ## shift index to 0 ~ 3

    if opts.suffix != "":
        opts.model_name += "_%s" %opts.suffix

    opts.size_multiplier = 2 ** 6 ## Inputs to FlowNet need to be divided by 64
    
    print(opts)

    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed(opts.seed)

    ### model saving directory
    opts.model_dir = os.path.join(opts.checkpoint_dir, opts.model_name)
    print("========================================================")
    print("===> Save model to %s" %opts.model_dir)
    print("========================================================")
    if not os.path.isdir(opts.model_dir):
        os.makedirs(opts.model_dir)

    ### initialize model
    print('===> Initializing model from %s...' %opts.single_model)
    multi_model_res    = networks.__dict__[opts.multi_model](opts, nc_in=12, nc_out=3, init_tag=0)
    multi_model_haze   = networks.__dict__[opts.multi_model](opts, nc_in=12, nc_out=3, init_tag=1)
    multi_model_s2 = networks.__dict__[opts.multi_model](opts, nc_in=12, nc_out=3, init_tag=0)
    multi_dis = Discriminator()    

    ### initialize optimizer
    if opts.solver == 'SGD':
        optimizer = optim.SGD(list(multi_model.parameters()), lr=opts.lr_init, momentum=opts.momentum, weight_decay= opts.weight_decay )
    elif opts.solver == 'ADAM':
        optimizer = optim.Adam([ \
                                {'params': multi_model_res.parameters(), 'lr': 0}, \
                                {'params': multi_model_haze.parameters(), 'lr': 1e-4}, \
                                {'params': multi_model_s2.parameters(), 'lr': 1e-4}, \
                               ], lr=opts.lr_init, weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2))

        optimizer_dis = optim.Adam([\
                                {'params': multi_dis.parameters(), 'lr': 1e-4}, \
                                ], lr=opts.lr_init, weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2))
    else:
        raise Exception("Not supported solver (%s)" %opts.solver)

    ### resume latest model
    name_list = glob.glob(os.path.join(opts.model_dir, "model_epoch_*.pth"))
    epoch_st = 0
    if len(name_list) > 0:
        epoch_list = []
        for name in name_list:
            s = re.findall(r'\d+', os.path.basename(name))[0]
            epoch_list.append(int(s))

        epoch_list.sort()
        epoch_st = epoch_list[-1]

    if epoch_st > 0:

        print('=====================================================================')
        print('===> Resuming model from epoch %d' %epoch_st)
        print('=====================================================================')

        ### resume latest model and solver
        multi_model_res, multi_model_haze, multi_model_s2, optimizer = utils.load_model(multi_model_res, multi_model_haze, multi_model_s2, optimizer, opts, epoch_st)

    else:
        ### save epoch 0
        utils.save_model(multi_model_res, multi_model_haze, multi_model_s2, optimizer, opts)

    print(multi_model_res)

    num_params = utils.count_network_parameters(multi_model_res) + utils.count_network_parameters(multi_model_s2) + utils.count_network_parameters(multi_model_haze)

    print('\n=====================================================================')
    print("===> Model has %d parameters" %num_params)
    print('=====================================================================')

    ### initialize loss writer
    loss_dir = os.path.join(opts.model_dir, 'loss')
    loss_writer = SummaryWriter(loss_dir)

    ### convert to GPU
    device = torch.device("cuda" if opts.cuda else "cpu")

    #flow_warping = Resample2d().to(device) 
    multi_model_res = multi_model_res.to(device)
    multi_model_haze = multi_model_haze.to(device)
    multi_model_s2 = multi_model_s2.to(device)
    multi_dis = multi_dis.to(device)

    multi_model_res.eval()
    multi_model_haze.train()
    multi_model_s2.train()
    multi_dis = multi_dis.train()

    ### create dataset
    train_dataset = datasets_multiple.MultiFramesDataset(opts,"rain_removal", "train")
    train_haze_dataset = datasets_multiple_haze.MultiFramesHazeDataset(opts, "rain_removal_haze", "train")

    ### start training
    while multi_model_res.epoch < opts.epoch_max:
        multi_model_res.epoch += 1

        ### re-generate train data loader for every epoch
        data_loader = utils.create_data_loader(train_dataset, opts, "train")
        data_haze_loader = utils.create_data_loader(train_haze_dataset, opts, "train")

        ### update learning rate
        current_lr = utils.learning_rate_decay(opts, multi_model_res.epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        if opts.loss == 'L2':
            criterion = nn.MSELoss(size_average=True)
        elif opts.loss == 'L1':
            criterion = nn.L1Loss(size_average=True)
        else:
            raise Exception("Unsupported criterion %s" %opts.loss)

        criterion_ssim = SSIM()
        criterion_mse = nn.MSELoss(size_average=True)
        criterion_l1 = nn.L1Loss(size_average=True)

        error_last = 1e8
        ts = datetime.now()

        state_dict = torch.load('pretrain/model_epoch_95.pth')

        multi_model_haze.load_state_dict(state_dict['multi_model_haze'])
        multi_model_s2.load_state_dict(state_dict['multi_model_s2'])

        for iteration, batch in enumerate(data_haze_loader, 1):

            total_iter = (multi_model_res.epoch - 1) * opts.train_epoch_size + iteration
            cross_num = 5

            frame_i = []
            frame_h = []
            frame_a = []
            frame_t = []
            frame_g = []

            frame_o = []
            for t in range(opts.sample_frames):
                frame_i.append(batch[t * cross_num].to(device))
                frame_h.append(batch[t * cross_num + 1].to(device))
                frame_a.append(batch[t * cross_num + 2].to(device))
                frame_t.append(batch[t * cross_num + 3].to(device))
                frame_g.append(batch[t * cross_num + 4].to(device))

            data_time = datetime.now() - ts
            ts = datetime.now()

            optimizer.zero_grad()
            optimizer_dis.zero_grad()
            Multi_loss = 0
            Multi_loss_gen = 0
            Dis_loss = 0

            lstm_state = None
            lstm_state_s2 = None
            lstm_state_trans = None
            lstm_state_trans_s2 = None
            lstm_state_alpha = None
            lstm_state_alpha_s2 = None

            for t in range(0, opts.sample_frames-4):

                frame_i1 = frame_i[t]
                frame_i2 = frame_i[t+1]
                frame_i3 = frame_i[t+2]
                frame_i4 = frame_i[t+3]
                frame_i5 = frame_i[t+4]

                frame_h1 = frame_h[t]
                frame_h2 = frame_h[t+1]
                frame_h3 = frame_h[t+2]
                frame_h4 = frame_h[t+3]
                frame_h5 = frame_h[t+4]

                frame_a1 = frame_a[t]
                frame_a2 = frame_a[t+1]
                frame_a3 = frame_a[t+2]
                frame_a4 = frame_a[t+3]
                frame_a5 = frame_a[t+4]

                frame_t1 = frame_t[t]
                frame_t2 = frame_t[t+1]
                frame_t3 = frame_t[t+2]
                frame_t4 = frame_t[t+3]
                frame_t5 = frame_t[t+4]

                frame_g1 = frame_g[t]
                frame_g2 = frame_g[t+1]
                frame_g3 = frame_g[t+2]
                frame_g4 = frame_g[t+3]
                frame_g5 = frame_g[t+4]

                inputs        = torch.cat((frame_i1, frame_i2, frame_i3, frame_i3, frame_i4, frame_i5), dim=1)
                frame_haze3_s1, _, _, _ , lstm_state, _ = multi_model_res(inputs, lstm_state, None)
                frame_trans3_s1, frame_alpha3_s1, _, _, lstm_state_trans, lstm_state_alpha = multi_model_haze(inputs, lstm_state_trans, lstm_state_alpha)
                
                [b, c, h, w] = inputs.shape


                tmp_float = float(np.random.randint(80,100))/100.0
                inputs_ones = torch.ones(b, c, h, w).cuda()*tmp_float
                frame_trans3_ones, frame_alpha3_ones, _, _, _, _ = multi_model_haze(inputs_ones, lstm_state_trans, lstm_state_alpha)

                frame_trans3_label = torch.ones(b, int(c/6), h, w).cuda()
                frame_alpha3_label = torch.ones(b, int(c/6), h, w).cuda()*tmp_float

                frame_trans3_s1 = frame_trans3_s1[:,0:1, :,:]
                frame_trans3_s1 = torch.cat((frame_trans3_s1, frame_trans3_s1, frame_trans3_s1),1)

                frame_trans3_s1[frame_trans3_s1<=0.05] = 0.05
                frame_res3_s1 = (frame_haze3_s1.detach() - frame_alpha3_s1.detach()*(1-frame_trans3_s1.detach()))/frame_trans3_s1.detach()

                if t+2>=4:
                    frame_o1 = frame_o[-2]
                else:
                    frame_o1 = frame_res3_s1.detach()

                if t+2>=3:
                    frame_o2 = frame_o[-1]
                else:
                    frame_o2 = frame_res3_s1.detach()

                inputs_s2     = torch.cat((frame_o1, frame_o2, frame_res3_s1.detach(), frame_i3, frame_i4, frame_i5), dim=1)
                inputs_s2_ones = inputs_ones

                frame_res3_s2, _, _, _, lstm_state_s2, _ = multi_model_s2(inputs_s2, lstm_state_s2, None)
                frame_res3_s2_ones, _, _, _, _, _ = multi_model_s2(inputs_s2_ones, lstm_state_s2, None)
                frame_res3_s2_label = frame_alpha3_label

                frame_o.append(frame_res3_s2.detach())

                lstm_state = utils.repackage_hidden(lstm_state)
                lstm_state_alpha = utils.repackage_hidden(lstm_state_alpha)
                lstm_state_trans = utils.repackage_hidden(lstm_state_trans)
                lstm_state_s2 = utils.repackage_hidden(lstm_state_s2)
              

                Multi_loss += -criterion_ssim(frame_haze3_s1, frame_h3.detach()) - criterion_ssim(frame_trans3_s1, frame_t3.detach()) -criterion_ssim(frame_alpha3_s1, frame_a3.detach()) + \
                              0.1*(criterion_mse(frame_trans3_ones, frame_trans3_label.detach()) + criterion_mse(frame_alpha3_ones, frame_alpha3_label.detach()) + criterion_mse(frame_res3_s2_ones, frame_res3_s2_label.detach()))

                ones_const = torch.ones(opts.batch_size, 1).cuda()
                target_real = (torch.rand(opts.batch_size, 1)*0.5 + 0.7).cuda()
                target_fake = (torch.rand(opts.batch_size, 1)*0.3).cuda()

                if t+2>=3:
                    dis_input = frame_res3_s2
                    dis_gt = frame_g3

                    frame_res3_s_d,_ = frame_res3_s2.min(dim=1, keepdim=True)
                    frame_o2_d,_ = frame_o2.min(dim=1, keepdim=True)
                    [b, c, h, w] = frame_o2_d.shape
                    zero_torch = torch.zeros(b, c, h, w).cuda()

                    Multi_loss_gen += -criterion_ssim(frame_res3_s2, frame_g3.detach()) + 0.001*criterion_mse(multi_dis(dis_input), target_real) + 0.1*criterion_l1(frame_res3_s_d, frame_o2_d.detach())+0.01*criterion_l1(frame_res3_s_d, zero_torch)
                    Dis_loss += criterion_mse(multi_dis(dis_gt), target_real) + criterion_mse(multi_dis(dis_input.detach()), target_fake)
                else:
                    Multi_loss_gen += -criterion_ssim(frame_res3_s2, frame_g3.detach())


            overall_loss =  Multi_loss + Multi_loss_gen + Dis_loss
            overall_loss.backward()

            optimizer_dis.step()
            optimizer.step()

            error_last_inner_epoch = overall_loss.item()

            network_time = datetime.now() - ts

            info = "[GPU %d]: " %(opts.gpu)
            info += "Epoch %d; Batch %d / %d; " %(multi_model_res.epoch, iteration, len(data_loader))
            info += "lr = %s; " %(str(current_lr))

            batch_freq = opts.batch_size / (data_time.total_seconds() + network_time.total_seconds())
            info += "data loading = %.3f sec, network = %.3f sec, batch = %.3f Hz\n" %(data_time.total_seconds(), network_time.total_seconds(), batch_freq)
            
            info += "\tmodel = %s\n" %opts.model_name

            if opts.w_RECT > 0:
                loss_writer.add_scalar('Multi_loss', Multi_loss.item(), total_iter)
                info += "\t\t%25s = %f\n" %("Multi_loss", Multi_loss.item())
                loss_writer.add_scalar('Multi_loss_gen', Multi_loss_gen.item(), total_iter)
                info += "\t\t%25s = %f\n" %("Multi_loss_gen", Multi_loss_gen.item())
                loss_writer.add_scalar('Dis_loss', Dis_loss.item(), total_iter)
                info += "\t\t%25s = %f\n" %("Dis_loss", Dis_loss.item())

            loss_writer.add_scalar('Overall_loss', overall_loss.item(), total_iter)
            info += "\t\t%25s = %f\n" %("Overall_loss", overall_loss.item())

            print(info)
            error_last = error_last_inner_epoch

        utils.save_model(multi_model_res, multi_model_haze, multi_model_s2, optimizer, opts)
