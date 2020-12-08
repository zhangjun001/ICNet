import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
from torch.autograd import Variable
from Models import ModelFlow_stride,SpatialTransform,antifoldloss,mse_loss,smoothloss
from Functions import Dataset,generate_grid
import torch.utils.data as Data
parser = ArgumentParser()
parser.add_argument("--lr", type=float, 
                    dest="lr", default=5e-4,help="learning rate") 
parser.add_argument("--iteration", type=int, 
                    dest="iteration", default=20001,
                    help="number of total iterations")
parser.add_argument("--inverse", type=float, 
                    dest="inverse", default=0.05,
                    help="Inverse consistent：suggested range 0.001 to 0.1")
parser.add_argument("--antifold", type=float, 
                    dest="antifold", default=100000,
                    help="Anti-fold loss: suggested range 100000 to 1000000")
parser.add_argument("--smooth", type=float, 
                    dest="smooth", default=0.5,
                    help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=4000, 
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8, 
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath", default='../Dataset', 
                    help="data path for training images")
opt = parser.parse_args()

lr = opt.lr
iteration = opt.iteration
start_channel = opt.start_channel
inverse = opt.inverse
antifold = opt.antifold
n_checkpoint = opt.checkpoint
smooth = opt.smooth
datapath = opt.datapath

def train():
    model =ModelFlow_stride(2,3,start_channel).cuda()
    loss_similarity =mse_loss
    loss_inverse = mse_loss
    loss_antifold = antifoldloss
    loss_smooth = smoothloss
    transform = SpatialTransform().cuda()
    for param in transform.parameters():
        param.requires_grad = False
        param.volatile=True
    names = glob.glob(datapath + '/*.gz')
    grid = generate_grid(imgshape)
    grid = Variable(torch.from_numpy(np.reshape(grid, (1,) + grid.shape))).cuda().float()

    print(grid.type())
    optimizer = torch.optim.Adam(model.parameters(),lr=lr) 
    model_dir = '../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((5,iteration))
    training_generator = Data.DataLoader(Dataset(names,iteration,True), batch_size=1,
                        shuffle=False, num_workers=2)
    step=0
    for  X,Y in training_generator:

        X = X.cuda().float()
        Y = Y.cuda().float()
        F_xy = model(X,Y)
        F_yx = model(Y,X)
    
        X_Y = transform(X,F_xy.permute(0,2,3,4,1)*range_flow,grid)
        Y_X = transform(Y,F_yx.permute(0,2,3,4,1)*range_flow,grid)
        # Note that, the generation of inverse flow depends on the definition of transform. 
        # The generation strategies are sligtly different for the backward warpping and forward warpping
        F_xy_ = transform(-F_yx,F_xy.permute(0,2,3,4,1)*range_flow,grid)
        F_yx_ = transform(-F_xy,F_yx.permute(0,2,3,4,1)*range_flow,grid)
        loss1 = loss_similarity(Y,X_Y) + loss_similarity(X,Y_X)
        loss2 = loss_inverse(F_xy*range_flow,F_xy_*range_flow) + loss_inverse(F_yx*range_flow,F_yx_*range_flow)
        
        
        loss3 =  loss_antifold(F_xy*range_flow) + loss_antifold(F_yx*range_flow)
        loss4 =  loss_smooth(F_xy*range_flow) + loss_smooth(F_yx*range_flow)
        loss = loss1+inverse*loss2 + antifold*loss3 + smooth*loss4
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        lossall[:,step] = np.array([loss.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item()])
        sys.stdout.write("\r" + 'step "{0}" -> training loss "{1:.4f}" - sim "{2:.4f}" - inv "{3:.4f}" \
            - ant "{4:.4f}" -smo "{5:.4f}" '.format(step, loss.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item()))
        sys.stdout.flush()
        if(step % n_checkpoint == 0):
            modelname = model_dir + '/' + str(step) + '.pth'
            torch.save(model.state_dict(), modelname)
        step+=1
    np.save(model_dir+'/loss.npy',lossall)
    
    
if __name__ == '__main__':
    imgshape = (144, 192, 160)
    range_flow = 7
    train()