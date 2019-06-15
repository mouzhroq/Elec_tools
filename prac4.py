import numpy as np 
import matplotlib.pyplot as plt 
import ph_tools as pt

#Ejercicio 1
dt = 0.01
tm = 2
N = int(tm / dt)
t = np.arange(0, tm+dt, dt)
Q1 = np.array([[0.999, 0.001], [0.001, 0.999]])
Q2 = np.array([[0.995, 0.001], [0.005, 0.999]])
Q3 = np.array([[0.999, 0.005], [0.001, 0.995]])
p0 = np.matrix([1, 0]).transpose()

p1 = pt.mul(Q1, p0, N)
p2 = pt.mul(Q2, p0, N)
p3 = pt.mul(Q2, p0, N)

s1 = pt.simc(p1, 1, N)
s2 = pt.simc(p2, 1, N)
s3 = pt.simc(p3, 1, N)

fig, (ax0, ax1, ax2) = plt.subplots(3, sharex=True)

ax0.plot(t, s1)
ax0.set(title='k+=0.1 & k-=0.1')
ax1.plot(t, s2)
ax1.set(title='k+=0.5 & k-=0.1')
ax2.plot(t, s3)
ax2.set(title='k+=0.1 & k-=0.5', xlabel='t (ms)')

#fig.savefig('Simulacion5.png')
#plt.show()

#Ejercicio 2
v = np.arange(-40, 81, 40)
a = (0.01*(10 - v)) / (np.exp((10 - v)/10.) - 1)
b = 0.125*np.exp(-v/80.)
Q11 = np.array([])
for i in range(len(v)):
    Q11 = np.append(Q11, np.array([[1-4*a[i]*dt, b[i]*dt, 0, 0, 0],[4*a[i]*dt, 1-b[i]*dt-3*a[i]*dt, 2*b[i]*dt, 0, 0],[0, 3*a[i]*dt, 1-2*b[i]*dt-2*a[i]*dt, 3*b[i]*dt, 0],[0, 0, 2*a[i]*dt, 1-3*b[i]*dt-a[i]*dt, 4*b[i]*dt],[0, 0, 0, a[i]*dt, 1-4*b[i]*dt]]))

Q11 = np.reshape(Q11, (4,5,5))
p01 = np.matrix([1, 0, 0, 0, 0]).transpose()

p11 =pt.mul(Q11[0,:,:], p01, N)
p12 =pt.mul(Q11[1,:,:], p01, N)
p13 =pt.mul(Q11[2,:,:], p01, N)
p14 =pt.mul(Q11[3,:,:], p01, N)

s11 = pt.simc(p11, 4, N)
s12 = pt.simc(p12, 4, N)
s13 = pt.simc(p13, 4, N)
s14 = pt.simc(p14, 4, N)

fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, sharex=True, sharey=True)

ax0.plot(t, s11)
ax0.set(title='Simulacion de Monte Carlo canal K+ V=-40')
ax1.plot(t, s12)
ax1.set(title='V=0')
ax2.plot(t, s13)
ax2.set(title='V=40')
ax3.plot(t, s14)
ax3.set(title='V=80', xlabel='t (ms)')

#fig.savefig('Simulacion11.png')
#plt.show()

#Ejercicio 3
vt = v + 60
e1 = 1 + np.exp(-0.1*(v + 45))
Q12 = np.array([])
for i in range(len(v)):
    q01 = 0.03 * np.exp(-0.05 * vt[i])
    q10 = 13.06 * np.exp(0.03 * vt[i])
    q02 = 0.001
    q20 = 8.94 / e1[i]
    q12 = 89.4 / e1[i]
    q21 = 0.001
    Q12 = np.append(Q12, np.array([[1-q01*dt-q02*dt, q10*dt, q20*dt], [q01*dt, 1-q10*dt-q12*dt, q21*dt], [q02*dt, q12*dt, 1-q20*dt-q21*dt]]))

Q12 = np.reshape(Q12, (4,3,3))
p02 = np.matrix([1, 0, 0]).transpose()

p012 = pt.mul(Q12[0,:,:], p02, N)
p022 = pt.mul(Q12[1,:,:], p02, N)
p032 = pt.mul(Q12[2,:,:], p02, N)
p042 = pt.mul(Q12[3,:,:], p02, N)

s012 = pt.simc(p012, 2, N) 
s022 = pt.simc(p022, 2, N)
s032 = pt.simc(p032, 2, N)
s042 = pt.simc(p042, 2, N)

fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, sharex=True, sharey=True)

ax0.plot(t, s012)
ax0.set(title='Simulacion de Monte Carlo canal Ca2+ V=-40')
ax1.plot(t, s022)
ax1.set(title='V=0')
ax2.plot(t, s032)
ax2.set(title='V=40')
ax3.plot(t, s042)
ax3.set(title='V=80', xlabel='t (ms)')

fig.savefig('Simulacion21.png')
plt.show()