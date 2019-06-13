import numpy as np 
import matplotlib.pyplot as plt 
from numpy.linalg import matrix_power

v = np.arange(-40, 121, 20)
dt = 0.01
tm = 50
N = tm / dt
t = np.arange(0, tm+dt, dt)
am = 0.1*(25-v)/(np.exp((25-v)/10.)-1)
bm = 4*np.exp(-v/18.)
ah = 0.07*np.exp(-v/20.)
bh = 1/(np.exp((30-v)/10.)+1)
Q1 = np.array([])
for i in range(len(v)):
    Q1 = np.append(Q1, np.array([[1-3*am[i]*dt-ah[i]*dt,bm[i]*dt,0,bh[i]*dt,0,0,0,0],[3*am[i]*dt,1-bm[i]*dt-ah[i]*dt-2*am[i]*dt,2*bm[i]*dt,0,bh[i]*dt,0,0,0],[0,2*am[i]*dt,1-2*bm[i]*dt-ah[i]*dt-am[i]*dt,0,0,bh[i]*dt,0,3*bm[i]*dt],[ah[i]*dt,0,0,1-bh[i]*dt-3*am[i]*dt,bm[i]*dt,0,0,0],[0,ah[i]*dt,0,3*am[i]*dt,1-bh[i]*dt-bm[i]*dt-2*am[i]*dt,2*bm[i]*dt,0,0],[0,0,ah[i]*dt,0,2*am[i]*dt,1-2*bm[i]*dt-bh[i]*dt-am[i]*dt,3*bm[i]*dt,0],[0,0,0,0,0,am[i]*dt,1-3*bm[i]*dt-bh[i]*dt,ah[i]*dt],[0,0,am[i]*dt,0,0,0,bh[i]*dt,1-3*bm[i]*dt-ah[i]*dt]]))

Q1 = np.reshape(Q1, (9,8,8))
p1 = np.matrix([1, 0, 0, 0, 0, 0, 0, 0]).transpose()
p = p1

for i in range (1, int(N)+1):
        p = np.append(p, matrix_power(Q1[0,:,:], i) * p1, axis=1)

fig, ax = plt.subplots()

ax.plot(t, p[7,:].transpose(), label='-40 mV')
ax.set(title='Probabilidad del estado O', xlabel='t (ms)', ylabel='Probabilidad')
ax.legend(loc='upper right')
plt.show()
#fig.savefig('Prob1.png')