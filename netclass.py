import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from generation import generators
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Hybird_net(nn.Module):

    def __init__(self,in_d,out_d,hidden):

        super(Hybird_net,self).__init__()
        # self.beta = nn.Parameter(torch.ones(1)*0.2)
        # self.gamma = nn.Parameter(torch.ones(1)*0.2)
        self.beta = 0.18
        self.gamma = 0.1
        self.L1 = nn.Linear(in_d,hidden)
        self.L2 = nn.Linear(hidden,hidden)
        self.L3 = nn.Linear(hidden,out_d)
        self.sigmoid = nn.Sigmoid()
        self.phi = nn.Tanh()
        self.relu = nn.ReLU()
        self.loss = nn.MSELoss()

    def forward(self,input):
        out = self.phi(self.L1(input))
        out = self.phi(self.L2(out))
        return self.L3(out)

    def odeModel(self,xlist):
        x,y = xlist
        b,g = self.beta, self.gamma
        out = self(x.unsqueeze(0))*y
        return torch.cat(( -b*out,
                      b*out-g*y ),0)

    def rk4loss(self,T,target):
        dt = 0.1
        N = int(T/dt)
        f = self.odeModel
        Yint = torch.tensor([98.,2.])
        Y = Yint
        Ylist = torch.zeros(2,N)
        Ylist[:,0] = Yint
        for i in range(N-1):
            k1 = f(Y)
            k2 = f(Y+k1*dt/2)
            k3 = f(Y+k2*dt/2)
            k4 = f(Y+k3*dt)
            Y = Y+ dt/6 *(k1+2*k2+2*k3+k4)
            Ylist[:,i+1] = Y

        return self.loss(Ylist[1,:].float(),target.float())  + self.relu(self.gamma - self.beta*self(torch.tensor([98.]) ))*100, Ylist[1,:]

epochs = 10000
lr = 0.01
model = Hybird_net(1,1,16)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
T,target = generators()
target = torch.from_numpy(target)
# scheduler = lr_scheduler.StepLR(optimizer, 250, 0.3) # [250,1500]

scheduler = lr_scheduler.MultiStepLR(optimizer,  milestones = [250,3000], gamma = 0.4, last_epoch=-1)
loss_record = np.zeros(epochs)
for epoch in range(epochs):
    optimizer.zero_grad()
    loss,ylist = model.rk4loss(T,target)
    loss.backward()
    optimizer.step()
    scheduler.step()
    print('epoch:',epoch,'loss:',loss.item())
    loss_record[epoch] = loss.item()
    if epoch > 1399 and epoch % 100 == 0:
        torch.save(model.state_dict(), 'model_paramsS.pkl')

np.savetxt("loss.txt", loss_record)
# model.load_state_dict(torch.load('model_paramsS.pkl'))
# plt.figure()
# plt.plot(np.arange(epochs),loss_record,'b-',lw=2)
# plt.xlabel('Iterations')
# plt.ylabel('Loss')

plt.figure()
tlist = np.arange(0,T,0.1)
plt.scatter(tlist,target.data.numpy(),label='The samples')
plt.plot(tlist,ylist.data.numpy(),'r-',lw=2,label='The fitted result')
plt.legend()
plt.xlabel('Time')
plt.ylabel('The number of infected person')
# erros = target.data.numpy() - ylist.data.numpy()
# erros = np.absolute(erros.ravel())
# plt.plot(tlist,erros,'b-')
# plt.xlabel('Time')
# plt.ylabel('The erros of the fitted eqution')
# plt.show()
# plt.figure()
# xlist = np.arange(0,20,0.1)
# inputs = torch.from_numpy(xlist).unsqueeze(1)
# plt.plot(xlist,model(inputs).data.numpy())
# plt.scatter(xlist,xlist/(xlist+20),'g')
# plt.show()
