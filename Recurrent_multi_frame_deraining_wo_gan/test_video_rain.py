#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2, time
import numpy as np

### torch lib
import torch
import torch.nn as nn

### custom lib
import networks
import utils
import matplotlib.pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency')

    ### model options
    parser.add_argument('-method',          type=str,     required=True,            help='test model name')
    parser.add_argument('-epoch',           type=int,     required=True,            help='epoch')

    ### dataset options
    parser.add_argument('-dataset',         type=str,     required=True,            help='dataset to test')
    parser.add_argument('-phase',           type=str,     default="test",           choices=["train", "test"])
    parser.add_argument('-data_dir',        type=str,     default='data',           help='path to data folder')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to list folder')
    parser.add_argument('-checkpoint_dir',  type=str,     default='checkpoints',    help='path to checkpoint folder')
    parser.add_argument('-task',            type=str,     required=True,            help='evaluated task')
    parser.add_argument('-redo',            action="store_true",                    help='Re-generate results')

    ### other options
    parser.add_argument('-gpu',             type=int,     default=0,                help='gpu device id')
    
    opts = parser.parse_args()
    opts.cuda = True

    opts.size_multiplier = 2 ** 2 ## Inputs to TransformNet need to be divided by 4

    print(opts)

    if opts.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without -cuda")


    opts_filename = os.path.join(opts.checkpoint_dir, opts.method, "opts.pth")
    print("Load %s" %opts_filename)
    with open(opts_filename, 'rb') as f:
        model_opts = pickle.load(f)

    print('===> Initializing model from %s...' %model_opts.single_model)
    multi_model_res = networks.__dict__[model_opts.multi_model](model_opts, nc_in=12, nc_out=3, init_tag=0)
    multi_model_haze = networks.__dict__[model_opts.multi_model](model_opts, nc_in=12, nc_out=3, init_tag=1)
    multi_model_s2 = networks.__dict__[model_opts.multi_model](model_opts, nc_in=12, nc_out=3, init_tag=0)
    model_filename = os.path.join(opts.checkpoint_dir, opts.method, "model_epoch_%d.pth" %opts.epoch)

    print("Load %s" %model_filename)
    state_dict = torch.load(model_filename)
    multi_model_res.load_state_dict(state_dict['multi_model_res'])
    multi_model_haze.load_state_dict(state_dict['multi_model_haze'])
    multi_model_s2.load_state_dict(state_dict['multi_model_s2'])

    device = torch.device("cuda" if opts.cuda else "cpu")
    multi_model_res = multi_model_res.to(device)
    multi_model_haze = multi_model_haze.to(device)
    multi_model_s2 = multi_model_s2.to(device)

    multi_model_res.train()
    multi_model_haze.train()
    multi_model_s2.train()

    print(count_parameters(multi_model_res)+count_parameters(multi_model_haze)+count_parameters(multi_model_s2))

    multi_model_res.eval()
    multi_model_haze.eval()
    multi_model_s2.eval()

    list_filename = os.path.join(opts.list_dir, "%s_%s.txt" %(opts.dataset, opts.phase))
    with open(list_filename) as f:
        video_list = [line.rstrip() for line in f.readlines()]

    times = []

    for v in range(len(video_list)):

        video = video_list[v]
        print("Test %s on %s-%s video %d/%d: %s" %(opts.task, opts.dataset, opts.phase, v + 1, len(video_list), video))

        input_dir = os.path.join(opts.data_dir, opts.phase, "Rain_Haze", video)
        output_dir = os.path.join(opts.data_dir, opts.phase, "output", opts.method, "epoch_%d" %opts.epoch, opts.task, opts.dataset, video)

        print(input_dir)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        frame_list = glob.glob(os.path.join(input_dir, "*.jpg"))
        output_list = glob.glob(os.path.join(output_dir, "*.png"))

        if len(frame_list) == len(output_list) and not opts.redo:
            print("Output frames exist, skip...")
            continue


        frame_o = []

        lstm_state = None
        lstm_state_s2 = None
        lstm_state_trans = None
        lstm_state_trans_s2 = None
        lstm_state_alpha = None
        lstm_state_alpha_s2 = None

        for t in range(1, len(frame_list)-3):
            frame_i1 = utils.read_img(os.path.join(input_dir, "%d.jpg" %(t))) 
            frame_i2 = utils.read_img(os.path.join(input_dir, "%d.jpg" %(t+1)))
            frame_i3 = utils.read_img(os.path.join(input_dir, "%d.jpg" %(t+2)))
            frame_i4 = utils.read_img(os.path.join(input_dir, "%d.jpg" %(t+3)))
            frame_i5 = utils.read_img(os.path.join(input_dir, "%d.jpg" %(t+4)))

            H_orig = frame_i1.shape[0]
            W_orig = frame_i1.shape[1]

            H_sc = int(math.ceil(float(H_orig) / opts.size_multiplier) * opts.size_multiplier)
            W_sc = int(math.ceil(float(W_orig) / opts.size_multiplier) * opts.size_multiplier)
                
            with torch.no_grad():
                frame_i1 = utils.img2tensor(frame_i1).to(device)
                frame_i2 = utils.img2tensor(frame_i2).to(device)
                frame_i3 = utils.img2tensor(frame_i3).to(device)
                frame_i4 = utils.img2tensor(frame_i4).to(device)
                frame_i5 = utils.img2tensor(frame_i5).to(device)
               
                [b, c, h, w] = frame_i1.shape

                frame_i1_new = torch.zeros(b, c, H_sc, W_sc).cuda()
                frame_i2_new = torch.zeros(b, c, H_sc, W_sc).cuda()
                frame_i3_new = torch.zeros(b, c, H_sc, W_sc).cuda()
                frame_i4_new = torch.zeros(b, c, H_sc, W_sc).cuda()
                frame_i5_new = torch.zeros(b, c, H_sc, W_sc).cuda()

                frame_i1_new[:,:,:h,:w] = frame_i1
                frame_i2_new[:,:,:h,:w] = frame_i2
                frame_i3_new[:,:,:h,:w] = frame_i3
                frame_i4_new[:,:,:h,:w] = frame_i4
                frame_i5_new[:,:,:h,:w] = frame_i5

                for s in range(h, H_sc):
                    frame_i1_new[:,:,s,:] = frame_i1_new[:,:,h-1,:]
                    frame_i2_new[:,:,s,:] = frame_i2_new[:,:,h-1,:]
                    frame_i3_new[:,:,s,:] = frame_i3_new[:,:,h-1,:]
                    frame_i4_new[:,:,s,:] = frame_i4_new[:,:,h-1,:]
                    frame_i5_new[:,:,s,:] = frame_i5_new[:,:,h-1,:]

                for s in range(w, W_sc):
                    frame_i1_new[:,:,:,s] = frame_i1_new[:,:,:,w-1]
                    frame_i2_new[:,:,:,s] = frame_i2_new[:,:,:,w-1]
                    frame_i3_new[:,:,:,s] = frame_i3_new[:,:,:,w-1]
                    frame_i4_new[:,:,:,s] = frame_i4_new[:,:,:,w-1]
                    frame_i5_new[:,:,:,s] = frame_i5_new[:,:,:,w-1]

                frame_i1 = frame_i1_new
                frame_i2 = frame_i2_new
                frame_i3 = frame_i3_new
                frame_i4 = frame_i4_new
                frame_i5 = frame_i5_new

                inputs = torch.cat((frame_i1, frame_i2, frame_i3, frame_i3, frame_i4, frame_i5), 1) 

                frame_haze3_s1, _, _, _ , lstm_state, _ = multi_model_res(inputs, lstm_state, None)
                frame_trans3_s1, frame_alpha3_s1, _, _, lstm_state_trans, lstm_state_alpha = multi_model_haze(inputs, lstm_state_trans, lstm_state_alpha)

                frame_trans3_s1 = frame_trans3_s1[:,0:1, :,:]
                frame_trans3_s1 = torch.cat((frame_trans3_s1, frame_trans3_s1, frame_trans3_s1),1)

                frame_trans3_s1[frame_trans3_s1<=0.05] = 0.05
                frame_res3_s1 = (frame_haze3_s1.detach() - frame_alpha3_s1.detach()*(1-frame_trans3_s1.detach()))/frame_trans3_s1.detach()

                if t+2>=5:
                    frame_o1 = frame_o[-2]
                else:
                    frame_o1 = frame_res3_s1.detach()

                if t+2>=4:
                    frame_o2 = frame_o[-1]
                else:
                    frame_o2 = frame_res3_s1.detach()

                inputs_s2     = torch.cat((frame_o1, frame_o2, frame_res3_s1.detach(), frame_i3, frame_i4, frame_i5), dim=1)

                frame_res3_s2, _, _, _, lstm_state_s2, _ = multi_model_s2(inputs_s2, lstm_state_s2, None)


                frame_o.append(frame_res3_s2.detach())

                lstm_state = utils.repackage_hidden(lstm_state)
                lstm_state_trans = utils.repackage_hidden(lstm_state_trans)
                lstm_state_alpha = utils.repackage_hidden(lstm_state_alpha)

                lstm_state_s2 = utils.repackage_hidden(lstm_state_s2)

            frame_res3_s1 = utils.tensor2img(frame_res3_s1[:,:,:h,:w])
            frame_res3_s2 = utils.tensor2img(frame_res3_s2[:,:,:h,:w])
            frame_i3 = utils.tensor2img(frame_i3[:,:,:h,:w])

            output_filename = os.path.join(output_dir, "%d.png" %(t+2))
            utils.save_img(frame_res3_s2, output_filename)
            output_filename = os.path.join(output_dir, "%d_s1.png" %(t+2))
            utils.save_img(frame_res3_s1, output_filename)
#            output_filename = os.path.join(output_dir, "%d_input.png" %(t+2))
#            utils.save_img(frame_i3, output_filename)

    if len(times) > 0:
        time_avg = sum(times) / len(times)
        print("Average time = %f seconds (Total %d frames)" %(time_avg, len(times)))
