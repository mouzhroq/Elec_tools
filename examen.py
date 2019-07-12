import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import ph_tools as pt 

Ik = np.loadtxt('FV_IKDR_6exp.txt')

v = np.array([70])
pro = pt.protoc(v, Ik[:,0], 500, 3501, -70)
fig, ax = plt.subplots()
ax.plot(Ik[:,0], pro[0,:]); ax.set(title='Protocolo de estimulacion', xlabel='t (ms)', ylabel='Voltaje (mV)')
fig.savefig('exaf1.png')
#plt.show()

fig, ax = plt.subplots()
ax.plot(Ik[:,0], Ik[:, 1:7]); ax.set(title='Corrientes', xlabel='t (ms)', ylabel='Corriente (nA)')
fig.savefig('exaf2.png')
#plt.show()

fig, ax = plt.subplots()
ax.plot(np.mean(Ik[500:3501,1::], axis=1), np.var(Ik[500:3501,1::], axis=1))
ax.set(title='Media de corriente vs varianza', xlabel='Corriente media', ylabel='Varianza')
fig.savefig('exaf3.png')
#plt.show()

def func (I, i, N):
    return I*i-(I**2 / N)

popt, pcov = curve_fit(func, np.mean(Ik[500:3501,1::], axis=1), np.var(Ik[500:3501,1::], axis=1))

fig, ax = plt.subplots()
ax.plot(np.mean(Ik[500:3501,1::], axis=1), np.var(Ik[500:3501,1::], axis=1))
ax.plot(np.mean(Ik[500:3501,1::], axis=1), func(np.mean(Ik[500:3501,1::], axis=1), *popt))
ax.set(title='Ajuste de la funcion', xlabel='Corriente media', ylabel='Varianza')
fig.savefig('exaf4.png')
#plt.show()

print ("Los valores del ajuste son: i = %.6f nA & N = %.2f canales." %(popt[0], popt[1]))

cu = popt[0] / 160
print ("La conductancia unitaria es: %.6f uS." %cu)

m_inf = 1/(1+np.exp(75.5/-16.7))
tau = 2 
alfa = m_inf / tau
beta = (1-m_inf) / tau

print("""
La m infinito es: %.6f.
La alfa es: %.6f.
La beta es: %.6f.
""" %(m_inf, alfa, beta) 
)

dt = 0.01
tm = 50
N = int(tm / dt)
t = np.arange(0, tm+dt , dt)
nc = int(np.around(popt[1]))
Q = np.array([[1-alfa*dt, beta*dt],[alfa*dt, 1-beta*dt]])
p0 = np.matrix([1, 0]).transpose()

pr = pt.mul(Q, p0, N)

sim = np.array([])
for i in range (nc):
    sim = np.append(sim, pt.simc(pr, 1, N))

sim = np.reshape(sim, (nc, N+1))
sim = sim * popt[0]
Iks = np.sum(sim, axis=0)

fig, (ax0, ax1)= plt.subplots(nrows=2, constrained_layout=True)
ax0.plot(Ik[500:3501,0], Ik[500:3501, 1:7]); ax0.set(title='Corrientes registradas', xlabel='t (ms)', ylabel='Corriente (nA)')
ax1.plot(t, Iks); ax1.set(title='Corriente simulada', xlabel='t (ms)', ylabel='Corriente (nA)')
fig.savefig('exaf5.png')
plt.show()

