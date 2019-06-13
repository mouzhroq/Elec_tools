import numpy as np 
import matplotlib.pyplot as plt 
import ph_tools as pt

#Ejercicio 2
v = np.arange(-40, 121, 20)
alfa = 0.01*(10-v)/(np.exp((10-v)/10.)-1)
beta = 0.125*np.exp(-v/80.)
dt = 0.1
Q = np.array([])
for i in range(len(v)):
    Q = np.append(Q, np.array([[1-4*alfa[i]*dt, beta[i]*dt, 0, 0, 0],[4*alfa[i]*dt, 1-beta[i]*dt-3*alfa[i]*dt, 2*beta[i]*dt, 0, 0],[0, 3*alfa[i]*dt, 1-2*beta[i]*dt-2*alfa[i]*dt, 3*beta[i]*dt, 0],[0, 0, 2*alfa[i]*dt, 1- 3*beta[i]*dt-alfa[i]*dt, 4*beta[i]*dt],[0, 0, 0, alfa[i]*dt, 1-4*beta[i]*dt]]))

Q = np.reshape(Q, (9,5,5))
p = np.matrix([1, 0, 0, 0, 0]).transpose()
tm = 50
N = tm / dt
t = np.arange(0, tm+dt, dt)
p0 = pt.mul(Q[0, :, :], p, N)
p1 = pt.mul(Q[1, :, :], p, N)
p2 = pt.mul(Q[2, :, :], p, N)
p3 = pt.mul(Q[3, :, :], p, N)
p4 = pt.mul(Q[4, :, :], p, N)
p5 = pt.mul(Q[5, :, :], p, N)
p6 = pt.mul(Q[6, :, :], p, N)
p7 = pt.mul(Q[7, :, :], p, N)
p8 = pt.mul(Q[8, :, :], p, N)

fig, ax = plt.subplots()

ax.plot(t, p0[4,:].transpose(), label='-40 mV')
ax.plot(t, p1[4,:].transpose(), label='-20 mV')
ax.plot(t, p2[4,:].transpose(), label='0 mV')
ax.plot(t, p3[4,:].transpose(), label='20 mV')
ax.plot(t, p4[4,:].transpose(), label='40 mV')
ax.plot(t, p5[4,:].transpose(), label='60 mV')
ax.plot(t, p6[4,:].transpose(), label='80 mV')
ax.plot(t, p7[4,:].transpose(), label='100 mV')
ax.plot(t, p8[4,:].transpose(), label='120 mV')
ax.set(title='Probabilidad del estado O', xlabel='t (ms)', ylabel='Probabilidad')
ax.legend(loc='upper right')
#plt.show()
fig.savefig('Prob.png')

#Ejercicio 3
dt = 0.01
am = 0.1*(25-v)/(np.exp((25-v)/10.)-1)
bm = 4*np.exp(-v/18.)
ah = 0.07*np.exp(-v/20.)
bh = 1/(np.exp((30-v)/10.)+1)
Q1 = np.array([])
for i in range(len(v)):
    Q1 = np.append(Q1, np.array([[1-3*am[i]*dt-ah[i]*dt,bm[i]*dt,0,bh[i]*dt,0,0,0,0],[3*am[i]*dt,1-bm[i]*dt-ah[i]*dt-2*am[i]*dt,2*bm[i]*dt,0,bh[i]*dt,0,0,0],[0,2*am[i]*dt,1-2*bm[i]*dt-ah[i]*dt-am[i]*dt,0,0,bh[i]*dt,0,3*bm[i]*dt],[ah[i]*dt,0,0,1-bh[i]*dt-3*am[i]*dt,bm[i]*dt,0,0,0],[0,ah[i]*dt,0,3*am[i]*dt,1-bh[i]*dt-bm[i]*dt-2*am[i]*dt,2*bm[i]*dt,0,0],[0,0,ah[i]*dt,0,2*am[i]*dt,1-2*bm[i]*dt-bh[i]*dt-am[i]*dt,3*bm[i]*dt,0],[0,0,0,0,0,am[i]*dt,1-3*bm[i]*dt-bh[i]*dt,ah[i]*dt],[0,0,am[i]*dt,0,0,0,bh[i]*dt,1-3*bm[i]*dt-ah[i]*dt]]))

Q1 = np.reshape(Q1, (9,8,8))
p1 = np.matrix([1, 0, 0, 0, 0, 0, 0, 0]).transpose()

p01 = pt.mul(Q1[0, :, :], p1, N)
p11 = pt.mul(Q1[1, :, :], p1, N)
p21 = pt.mul(Q1[2, :, :], p1, N)
p31 = pt.mul(Q1[3, :, :], p1, N)
p41 = pt.mul(Q1[4, :, :], p1, N)
p51 = pt.mul(Q1[5, :, :], p1, N)
p61 = pt.mul(Q1[6, :, :], p1, N)
p71 = pt.mul(Q1[7, :, :], p1, N)
p81 = pt.mul(Q1[8, :, :], p1, N)

fig, ax = plt.subplots()

ax.plot(t, p01[7,:].transpose(), label='-40 mV')
ax.plot(t, p11[7,:].transpose(), label='-20 mV')
ax.plot(t, p21[7,:].transpose(), label='0 mV')
ax.plot(t, p31[7,:].transpose(), label='20 mV')
ax.plot(t, p41[7,:].transpose(), label='40 mV')
ax.plot(t, p51[7,:].transpose(), label='60 mV')
ax.plot(t, p61[7,:].transpose(), label='80 mV')
ax.plot(t, p71[7,:].transpose(), label='100 mV')
ax.plot(t, p81[7,:].transpose(), label='120 mV')
ax.set(title='Probabilidad del estado O', xlabel='t (ms)', ylabel='Probabilidad')
ax.legend(loc='upper right')
#plt.show()
fig.savefig('Prob1.png')

#Ejercicio 4
dt1 = 0.25
tm=500
N = tm / dt1
t2 = np.arange(0, tm+dt1, dt1)
Q2 = np.array([[0.975, 0.025],[0.025, 0.975]])
Q21 = np.array([[0.875, 0.025],[0.125, 0.975]])
Q22 = np.array([[0.95, 0.25],[0.05, 0.75]])
p2 = np.matrix([1, 0]).transpose()

p20 = pt.mul(Q2, p2, N)
p21 = pt.mul(Q21, p2, N)
p22 = pt.mul(Q22, p2, N)

fig, ax = plt.subplots()

ax.plot(t2, p20[0,:].transpose(), label='Cerrado')
ax.plot(t2, p20[1,:].transpose(), label='Abierto')
ax.legend(loc='upper right')
ax.set(title='Probabilidad K+=0.1 & K-=0.1', xlabel='t (ms)', ylabel='Probabilidad')
fig.savefig('Probma1.png')

fig, ax = plt.subplots()

ax.plot(t2, p21[0,:].transpose(), label='Cerrado')
ax.plot(t2, p21[1,:].transpose(), label='Abierto')
ax.legend(loc='upper right')
ax.set(title='Probabilidad K+=0.5 & K-=0.1', xlabel='t (ms)', ylabel='Probabilidad')
fig.savefig('Probma2.png')

fig, ax = plt.subplots()

ax.plot(t2, p22[0,:].transpose(), label='Cerrado')
ax.plot(t2, p22[1,:].transpose(), label='Abierto')
ax.legend(loc='upper right')
ax.set(title='Probabilidad K+=0.2 & K-=1', xlabel='t (ms)', ylabel='Probabilidad')
fig.savefig('Probma3.png')
plt.show()