import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy as sp
import ph_tools as pt

cik = 124
cout = np.loadtxt('pot.txt') 
vk = pt.eq_po(cik, cout, 1, 37)

print (vk)

fig, ax = plt.subplots()

ax.plot(cout, vk*1000)
ax.set(title="[K+]o vs Vk", xlabel = "[K+]o (mM)", ylabel="VK(mV)" )

#plt.show()
#fig.savefig('Potencialk.png')

cina = 14
vNa = pt.eq_po(cina, cout, 1, 37)

print (vNa)

fig, ax = plt.subplots()

ax.plot(cout, vNa*1000)
ax.set(title="[Na+]o vs VNa", xlabel = "[Na+]o (mM)", ylabel="VNa(mV)" )

#plt.show()
#fig.savefig('PotencialNa.png')

ica = np.loadtxt('ICa.txt')
ica2 = np.loadtxt('ICa2.txt')

fig, (ax0, ax1) = plt.subplots(nrows=2, constrained_layout=True)

ax0.plot(ica[:,0],ica[:,1])
ax1.plot(ica2[:,0],ica2[:,1])

ax0.set(title='Corriente 1', xlabel='mV', ylabel='pA'); ax0.grid()
ax1.set(title='Corriente 2', xlabel='mV', ylabel='pA'); ax1.grid()

#fig.savefig('CorrientesCa.png')
#plt.show()

mod = LinearRegression().fit(ica[:,0].reshape((-1, 1)),ica[:,1])
mod1 = LinearRegression().fit(ica2[:,0].reshape((-1, 1)),ica2[:,1])
nv = np.arange(-100, 100, 0.1)
ext = mod.predict(nv.reshape((-1, 1)))
ext2 = mod1.predict(nv.reshape((-1, 1)))

fig, (ax0, ax1) = plt.subplots(nrows=2, constrained_layout=True)

ax0.plot(nv, ext)
ax1.plot(nv, ext2)

ax0.set(title='Corriente 1 extrapolada', xlabel='mV', ylabel='pA'); ax0.grid()
ax1.set(title='Corriente 2 extrapolada', xlabel='mV', ylabel='pA'); ax1.grid()

#fig.savefig('CorrientesCaExt.png')
#plt.show()

p1 = pt.pend(nv, ext)
p2 = pt.pend(nv, ext2)
print """
Conductancia 1: %.4f
Conductancia 2: %.4f
 """ % (p1, p2)

pe = pt.eq_po(100E-6, 5.24E-3, 2, 23)/1E-3
print "%.2f" % pe

