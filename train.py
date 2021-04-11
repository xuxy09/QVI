import models
import datas

import argparse
import torch
import torchvision
import torchvision.transforms as TF
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from math import log10
import numpy as np
import datetime
from utils.config import Config
from tensorboardX import SummaryWriter
import sys


# prepare perceptual loss
vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22]).cuda()
vgg16_conv_4_3 = nn.DataParallel(vgg16_conv_4_3.cuda())


# loss function
def lossfn(output, I1, I2, IT):
    It_warp = output

    recnLoss = F.l1_loss(It_warp, IT)
    prcpLoss = F.mse_loss(vgg16_conv_4_3(It_warp), vgg16_conv_4_3(IT))

    loss = 204 * recnLoss + 0.005 * prcpLoss

    return loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# loading configures
parser = argparse.ArgumentParser()
parser.add_argument('config')
args = parser.parse_args()

# load input config
config = Config.from_file(args.config)

# preparing transform & datasets
normalize1 = TF.Normalize(config.mean, [1.0, 1.0, 1.0])
normalize2 = TF.Normalize([0, 0, 0], config.std)
trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

revmean = [-x for x in config.mean]
revstd = [1.0/x for x in config.std]
revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
revNormalize = TF.Compose([revnormalize1, revnormalize2])

revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])

trainset = getattr(datas, config.trainset)(config.trainset_root, trans, config.train_size, config.train_crop_size, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size, shuffle=True, num_workers=32)

validationset = getattr(datas, config.validationset)(config.validationset_root, trans, config.validation_size, config.validation_crop_size, train=False)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=8)

print(validationset)


# model
model = getattr(models, config.model)(config.pwc_path).cuda()
model = nn.DataParallel(model)

# optimizer
params = list(model.module.refinenet.parameters()) + list(model.module.masknet.parameters())
optimizer = optim.Adam(params, lr=config.init_learning_rate)

# scheduler to decrease learning rate by a factor of 10 at milestones.
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.3)
recorder = SummaryWriter(config.record_dir)

print('Everything prepared. Ready to train...')

to_img = TF.ToPILImage()

def validate():
    retImg = []
    # For details see training.
    psnr = 0
    psnrs = [0 , 0, 0, 0, 0, 0, 0]
    tloss = 0
    tlosses = [0, 0, 0, 0, 0, 0, 0]
    flag = True
    retImg = []

    with torch.no_grad():

        for validationIndex, validationData in enumerate(validationloader, 0):
            frame0, frame1, frameT1, frameT2, frameT3, frameT4, frameT5, frameT6, frameT7, frame2, frame3 = validationData

            ITs = [frameT1.cuda(), frameT2.cuda(), frameT3.cuda(), frameT4.cuda(), frameT5.cuda(), frameT6.cuda(), frameT7.cuda()]

            I0 = frame0.cuda()
            I1 = frame1.cuda()
            I2 = frame2.cuda()
            I3 = frame3.cuda()

            It_warps = []
            Ms = []

            for tt in range(7):
                IT = ITs[tt]

                output = model(I0, I1, I2, I3, tt/8.0 + 0.125)
                It_warp = output
             
                It_warps.append(It_warp)

                loss = lossfn(output, I1, I2, IT)
                tlosses[tt] += loss.item()

            # record psnrs 
                MSE_val = F.mse_loss(It_warp, IT)
                psnrs[tt] += (10 * log10(1 / MSE_val.item()))
	    # record interpolated frames 
            img_grid = []
            img_grid.append(revNormalize(frame1[0]))
            for tt in range(7):
                img_grid.append(revNormalize(It_warps[tt].cpu()[0]))
            img_grid.append(revNormalize(frame2[0]))

            retImg.append(torchvision.utils.make_grid(img_grid, nrow=10, padding=10))

        for tt in range(7):
            psnrs[tt] /= len(validationloader)
            tlosses[tt] /= len(validationloader)

    return psnrs, tlosses, retImg

def train():

    if config.train_continue:
        dict1 = torch.load(config.checkpoint)
        model.load_state_dict(dict1['model_state_dict'])
    else:
        dict1 = {'loss': [], 'valLoss': [], 'valPSNR': [], 'epoch': -1}

    if not os.path.exists(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)

    start = time.time()
    cLoss   = dict1['loss']
    valLoss = dict1['valLoss']
    valPSNR = dict1['valPSNR']
    checkpoint_counter = 0


    for epoch in range(dict1['epoch'] + 1, config.epochs):

        print("Epoch: ", epoch)

        # Append and reset
        cLoss.append([])
        valLoss.append([])
        valPSNR.append([])
        iLoss = 0

        trainFrameIndex = 3
        for trainIndex, (trainData, t) in enumerate(trainloader, 0):
            print(trainIndex, len(trainloader))
            # Get the input and the target from the training set
            frame0, frame1, frameT, frame2, frame3 = trainData


            I0 = frame0.cuda()
            I1 = frame1.cuda()
            I2 = frame2.cuda()
            I3 = frame3.cuda()
            IT = frameT.cuda()
            t = t.view(t.size(0,), 1, 1, 1).float().cuda()

            optimizer.zero_grad()
            output = model(I0, I1, I2, I3, t)
            loss = lossfn(output, I1, I2, IT)
            loss.backward()
            optimizer.step()

            iLoss += loss.item()

            # Validation and progress every `config.progress_iter` iterations
            if ((trainIndex % config.progress_iter) == config.progress_iter - 1):
                end = time.time()

                psnrs, vLosses, valImgs = validate()

                psnr = np.mean(psnrs)
                vLoss = np.mean(vLosses)

                valPSNR[epoch].append(np.mean(psnrs))
                valLoss[epoch].append(np.mean(vLosses))

                #Tensorboard
                itr = trainIndex + epoch * (len(trainloader))

                recorder.add_scalars('Loss', {'trainLoss': iLoss/config.progress_iter, 'validationLoss': vLoss}, itr)
                    # recorder.add_scalar('PSNR' + , psnr, itr)

                vtdict = {}
                psnrdict = {}
                for tt in range(7):
                    vtdict['validationLoss' + str(tt + 1)] = vLosses[tt]
                    psnrdict['PSNR' + str(tt + 1)] = psnrs[tt]

                recorder.add_scalars('Losst', vtdict, itr)
                recorder.add_scalars('PSNRt', psnrdict, itr)

                #for vi, valImg in enumerate(valImgs):
                #    recorder.add_image('Validation' + str(vi), valImg , itr)

                endVal = time.time()

                print(" Loss: %0.6f  Iterations: %4d/%4d  TrainExecTime: %0.1f  ValLoss:%0.6f  ValPSNR: %0.4f  ValEvalTime: %0.2f LearningRate: %f" % (iLoss / config.progress_iter, trainIndex, len(trainloader), end - start, vLoss, psnr, endVal - end, get_lr(optimizer)))
                sys.stdout.flush()

                cLoss[epoch].append(iLoss/config.progress_iter)
                iLoss = 0
                start = time.time()

        # Create checkpoint after every `config.checkpoint_epoch` epochs
        if ((epoch % config.checkpoint_epoch) == config.checkpoint_epoch - 1):
            dict1 = {
                    'Detail':"Quadratic video interpolation.",
                    'epoch':epoch,
                    'timestamp':datetime.datetime.now(),
                    'trainBatchSz':config.train_batch_size,
                    'validationBatchSz':1,
                    'learningRate':get_lr(optimizer),
                    'loss':cLoss,
                    'valLoss':valLoss,
                    'valPSNR':valPSNR,
                    'model_state_dict': model.state_dict(),
                    }
            torch.save(dict1, config.checkpoint_dir + "/model" + str(checkpoint_counter) + ".ckpt")
            checkpoint_counter += 1

        # Increment scheduler count
        scheduler.step()

train()
