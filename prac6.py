import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import ph_tools as pt 

Ik = np.loadtxt('Data_FV_IK_Noisy.txt')
v = np.arange(-60, 51, 10)

fig, ax = plt.subplots()
ax.plot(Ik[:,0], Ik[:,1:13])
ax.set(title='Corriente de potasio', xlabel='t (ms)', ylabel='IK (pA/pF)')
fig.savefig('Corriente_k.png')
plt.show(block=False)

vol = pt.protoc(v, Ik[:,0], 200, 1200, -70)
fig, ax = plt.subplots()
for i in range (len(v)):
    ax.plot(Ik[:,0], vol[i,:])
ax.set(title='Protocolo de estimulacion', xlabel='t (ms)', ylabel='Voltaje (mV)')
fig.savefig('protocolo.png')
plt.show(block=False)

cor = np.array([])
for x in range (1, Ik.shape[1]):
    cor = np.append(cor, np.mean(Ik[600:800,x]))

fig, ax = plt.subplots()
ax.plot(v, cor)
ax.set(title='Curva I-V', xlabel='Voltaje (mV)', ylabel='Corriente (pA)')
fig.savefig('Corriente_voltaje.png')

vk = -75
gv = np.array([])
for i in range (len(v)):
    if (cor[i]/(v[i]-vk)) < 0:
        gv = np.append(gv, 0)
    else:
        gv = np.append(gv, cor[i]/(v[i]-vk))


gvn = gv / np.max(gv)
fig, ax = plt.subplots()
ax.plot(v, gvn)
ax.set(title='Curva de activacion m infinito', xlabel='Voltaje (mV)', ylabel='m infinito')
plt.show(block=False)
fig.savefig('m_inf.png')

def func (t, A, tau, n):
    return A*((1-np.exp(-t/tau))**n)

popt, pcov = curve_fit(func, Ik[150:300,0], Ik[150:300,7])
tau = popt[1]
print "La constante de tiempo tau es: %.3f" %tau
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(Ik[150:300,0], Ik[150:300,7])
ax.plot(Ik[150:300,0], func(Ik[150:300,0], *popt))
ax.set(title='Ajuste de la funcion', xlabel='t (ms)', ylabel='Corriente (pA)')
plt.show(block=False)
fig.savefig('Ajuste.png')

alpha = gvn / tau
beta = (1- gvn) / tau

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, constrained_layout=True)
ax0.plot(v, alpha)
ax0.set(title='Alfa', ylabel='Alfa')
ax1.plot(v, beta)
ax1.set(title='Beta', xlabel='Voltaje (mV)', ylabel='Beta')
plt.show(block=False)
fig.savefig('A_B.png')

dt = 0.01
tm = 100
N = int(tm / dt)
t = np.arange(0, tm+dt, dt)
Q1 = np.array([[1-alpha[1]*dt, beta[1]*dt],[alpha[1]*dt,1-beta[1]*dt]])
Q2 = np.array([[1-alpha[6]*dt, beta[6]*dt],[alpha[6]*dt,1-beta[6]*dt]])
Q3 = np.array([[1-alpha[11]*dt, beta[11]*dt],[alpha[11]*dt,1-beta[11]*dt]])
p0 = np.matrix([1, 0]).transpose()

p1 = pt.mul(Q1, p0, N)
p2 = pt.mul(Q2, p0, N)
p3 = pt.mul(Q3, p0, N)

s1 = pt.simc(p1, 1, N)
s2 = pt.simc(p2, 1, N)
s3 = pt.simc(p3, 1, N)

fig, (ax0, ax1, ax2) = plt.subplots(3, sharex=True, constrained_layout=True)
ax0.plot(t[0:200], s1[0:200])
ax0.set(title='-50 mV')
ax1.plot(t[0:200], s2[0:200])
ax1.set(title='0 mV')
ax2.plot(t[0:200], s3[0:200])
ax2.set(title='50 mV', xlabel='t (ms)')
fig.suptitle('Simulacion de Monte Carlo')
plt.show(block=False)
fig.savefig('Simp6.png')

fig, ax = plt.subplots()
ax.hist(s2); ax.set(title='Histograma de amplitudes',ylabel='# de puntos', xlabel='Amplitud')
plt.show(block=False)
fig.savefig('Amplitudesp6.png')

ti, tic, bins, binsc = pt.tiempo(s2, t, 0.01)
fig, (ax0, ax1) = plt.subplots(nrows=2, constrained_layout=True)
ax0.hist(ti, bins); ax0.set(title='Histograma de tiempo abierto',ylabel='# de puntos', xlabel='t (ms)')
ax1.hist(tic, binsc); ax1.set(title='Histograma de tiempo cerrado',ylabel='# de puntos', xlabel='t (ms)')
plt.show()
fig.savefig('Tiemposp6.png')
