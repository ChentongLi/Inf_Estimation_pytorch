import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from GenerPeric import generators
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Hybird_net(nn.Module):

    def __init__(self,in_d,out_d,hidden):

        super().__init__()
        self.L1 = nn.Linear(in_d,hidden)
        self.L2 = nn.Linear(hidden,hidden)
        self.L3 = nn.Linear(hidden,out_d)
        self.relu = nn.ReLU()
    
    def Snake(self,x):
        a = 0.1
        return torch.sin(a*x).pow(2)/a

    def forward(self,input):
        out = self.Snake(self.L1(input))
        out = self.Snake(self.L2(out))
        return  self.L3(out)

# def odeModel(xlist,betat):
#     x,y=xlist
#     g = 0.1
#     return torch.tensor([-betat*x*y/(x+y) + g*y + 1-x,betat*x*y/(x+y) - g*y])

loss_func = nn.MSELoss()
relu = nn.ReLU()
# def rk4loss(NNT,target,Yint):
#     N = target.size()[0]-1
#     dt = 0.1
#     f = odeModel
#     Y = Yint
#     Ylist = torch.zeros(2,N)
#     Ylist[:,0] = Yint
#     for i in range(N):
#         nnt = NNT[i]
#         k1 = f(Y,nnt)
#         k2 = f(Y+k1*dt/2,nnt)
#         k3 = f(Y+k2*dt/2,nnt)
#         k4 = f(Y+k3*dt,nnt)
#         Y = Y+ dt/6 *(k1+2*k2+2*k3+k4)
#         Ylist[:,i+1] = Y

#      return lossfunc(Ylist[0,:],target)


def FEulerLoss(NNT,target,Yint,dt,M):
    N = target.size()[0]-1
    S,I = Yint
    Ylist = torch.zeros(N+1,1)
    Ylist[0] = I
    # print(Yint)
    g,d = 0.1,0.05
    for i in range(N):
        nnt = NNT[i]*0.2
        Nm = len(str(int(nnt.abs())))-2
        Nm = np.power(10,Nm) if Nm > 0 else 1
        dmt = dt/Nm
        for j in range(Nm):
            lam = (1+g*I)/(nnt*I/(S+I)+d)
            S = lam + (S-lam)*torch.exp(-(nnt*I/(S+I)+d)*dmt)
            I = I * torch.exp((nnt*S/(S+I) - (g+d))*dmt)
        Ylist[i+1] = I
        # if I<0:
        #     print('i is:',i)
        #     print('The nan causing nnt is:',NNT.t())
        #     print('The record Ylist is:',Ylist.t())
        #     sys.exit(1) 

    return loss_func(Ylist.float(),target.float()) + 300 * relu((g+d)-NNT.mean()*0.2)+ 100 * relu(-NNT.min()),Ylist

def train():
    epochs = 10000
    lr = 0.01
    model = Hybird_net(1,1,16)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer,  milestones = [200,1500], gamma = 0.3, last_epoch=-1)
    # scheduler = lr_scheduler.StepLR(optimizer, 200, 0.3)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    T,dt,k,M,target = generators()
    Yint = [M-target[0],target[0]]
    target = torch.from_numpy(target).unsqueeze(1)
    tlist = Variable(torch.arange(0,k*dt,dt).unsqueeze(1))
    loss_record = np.zeros(epochs)
    # model.load_state_dict(torch.load('net_params.pkl')) # given the initial value
    # f1,f2=1,1
    for epoch in range(epochs):
        optimizer.zero_grad()
        NNT = model(tlist)
        loss,ylist = FEulerLoss(NNT,target,Yint,dt,M)
        # loss = loss_func(model(tlist).float(),target.float())
        loss.backward()
        optimizer.step()
        scheduler.step()
        # if epoch > 200:
        #     scheduler.step(loss)
        print('epoch:',epoch,'loss:',loss.item())
        loss_record[epoch] = loss.item()   
     
        if epoch > 3599 and epoch % 100 == 0:
            torch.save(model.state_dict(), 'model_params.pkl')

    #     if epoch % 5 == 0:
    #         plt.cla()
    #         plt.scatter(tlist.data.numpy(),target.data.numpy())
    #         plt.plot(tlist.data.numpy(),prediction.data.numpy(),'r-',lw=5)

    torch.save(model.state_dict(), 'model_params.pkl')
    plt.figure()
    plt.scatter(tlist.data.numpy(),target.data.numpy(),label='The samples')
    plt.plot(tlist.data.numpy(),ylist.data.numpy(),'r-',lw=2,label='The fitted result')
    # plt.plot(tlist.data.numpy(),NNT.data.numpy(),'b-',lw=2)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('The number of infected person')
    plt.figure()
    plt.plot(np.arange(epochs),loss_record,'b-',lw=2)
    plt.yscale('log')
    plt.grid(True, which="both",ls='--',linewidth = 0.3)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

# def plotfig():

#     T,dt,k,M,target = generators()
#     target = torch.from_numpy(target).unsqueeze(1)
#     tlist = Variable(torch.arange(0,k*dt,dt).unsqueeze(1))
#     model = Hybird_net(1,1,16)
#     model.load_state_dict(torch.load('model_paramsP2.pkl'))
#     Yint = [M-target[0],target[0]]
#     NNT = model(tlist)
#     _,ylist = FEulerLoss(NNT,target,Yint,dt,M)
#     erros = ylist.data.numpy()-target.data.numpy()
#     erros = np.absolute(erros.ravel())
#     # print(erros)
#     plt.figure()
#     plt.plot(tlist.data.numpy().ravel(),erros,'b-')
#     plt.xlabel('Time')
#     plt.ylabel('The erros of the fitted equations')
#     plt.xlim(0,548)
#     plt.show()


if __name__ == '__main__':
   train()
   # plotfig()