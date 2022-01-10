import numpy as np
# import matplotlib.pyplot as plt 
def generators():
    np.random.seed(1234)
    t=0
    beta=0.18
    gamma=0.1
    S=98
    I=2
    dt=0.1
    T = 100
    N=int(T/dt)
    It = np.zeros(N)
    Tt = np.arange(0,T,0.1)
    for i in range(N):
        betat = beta*dt
        S=S*np.exp(-betat*I/(20+S))
        I=I*np.exp(betat*S/(20+S)-gamma*dt)
        t=t+dt
        It[i]=I
    for i in range(N):
        It[i]+=0.1*np.random.randn()
    # plt.plot(Tt,It)
    # plt.show()
    return T,It

# generators()