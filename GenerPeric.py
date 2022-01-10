import numpy as np
# import matplotlib.pyplot as plt

def generators():
    np.random.seed(1234)
    beta,gamma,d = 0.5,0.1,0.05
    S0,I0 = 19.5,0.5
    S,I =S0,I0
    dt = 0.1 # 0.05
    T = 2000.
    N = int(T/dt)
    It = np.zeros(N)
    for i in range(N):
        # Bt = np.sqrt(dt)*np.random.randn()*0.1
        # betat = beta*(dt+Bt)
        betat = beta*dt
        # peric = np.sin(i*dt/11) + 1 
        peric = (-np.cos(i*dt/11)-np.sin(i*dt/11)-2.5*np.cos(2*i*dt/11)+0.5*np.sin(2*i*dt/11))/4. + 1
        S = S - betat*peric*S*I/(S+I) + (gamma*I+1-d*S)*dt 
        I = I + betat*peric*S*I/(S+I) - (gamma+d)*I*dt
        It[i] = I
    for i in range(N):
        It[i] = It[i] + np.random.randn()*0.1
    # Tt = np.arange(0,T,dt)
    # plt.plot(Tt,It)
    # plt.xlim([1500,2000])
    # plt.show()
    k=int(550/dt)
    return T,dt,k,S+I,It[-(k+1):-1]

# generators()