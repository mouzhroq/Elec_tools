import numpy as np 
import matplotlib.pyplot as plt 
import ph_tools as pt
from sklearn.linear_model import LinearRegression

v0 = np.loadtxt('single_-60mv.txt')
v1 = np.loadtxt('single_-40mv.txt')
v2 = np.loadtxt('single_-20mv.txt')
v3 = np.loadtxt('single_0mv.txt')
v4 = np.loadtxt('single_20mv.txt')
v5 = np.loadtxt('single_40mv.txt')
v6 = np.loadtxt('single_60mv.txt')

fig,(ax0, ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(nrows=7,sharex=True ,sharey=True,constrained_layout=True)
ax0.plot(v0[0:10000,0],v0[0:10000,1]);ax0.set(ylabel='I[pA] -60mV')
ax1.plot(v1[0:10000,0],v1[0:10000,1]);ax1.set(ylabel='I[pA] -40mV')
ax2.plot(v2[0:10000,0],v2[0:10000,1]);ax2.set(ylabel='I[pA] -20mV')
ax3.plot(v3[0:10000,0],v3[0:10000,1]);ax3.set(ylabel='I[pA] 0mV')
ax4.plot(v4[0:10000,0],v4[0:10000,1]);ax4.set(ylabel='I[pA] 20mV')
ax5.plot(v5[0:10000,0],v5[0:10000,1]);ax5.set(ylabel='I[pA] 40mV')
ax6.plot(v6[0:10000,0],v6[0:10000,1]);ax6.set(xlabel='Tiempo[ms]'); ax6.set(ylabel='I[pA] 60mV')
fig.set_size_inches(18.5, 10.5)
fig.suptitle('Corrientes de potasio',fontsize=20)
fig.savefig('IK.png')

fig,(ax0, ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(nrows=7,sharex=True ,sharey=True,constrained_layout=True)
ax0.plot(v0[0:100000,0],v0[0:100000,2]);ax0.set(ylabel='INa -60mV')
ax1.plot(v1[0:100000,0],v1[0:100000,2]);ax1.set(ylabel='INa -40mV')
ax2.plot(v2[0:100000,0],v2[0:100000,2]);ax2.set(ylabel='INa -20mV')
ax3.plot(v3[0:100000,0],v3[0:100000,2]);ax3.set(ylabel='INa 0mV')
ax4.plot(v4[0:100000,0],v4[0:100000,2]);ax4.set(ylabel='INa 20mV')
ax5.plot(v5[0:100000,0],v5[0:100000,2]);ax5.set(ylabel='INa 40mV')
ax6.plot(v6[0:100000,0],v6[0:100000,2]);ax6.set(ylabel='INa 60mV', xlabel='Tiempo[ms]')
fig.suptitle('Corrientes de sodio (Na+)',fontsize=20)
fig.set_size_inches(18.5, 10.5)
fig.savefig('INa.png')
#plt.show()

Ik = np.array([max(v0[:,1]),max(v1[:,1]),max(v2[:,1]),max(v3[:,1]),max(v4[:,1]),max(v5[:,1]),max(v6[:,1])])
INa = np.array([min(v0[:,2]),min(v1[:,2]),min(v2[:,2]),min(v3[:,2]),min(v4[:,2]),min(v5[:,2]),max(v6[:,2])])
V = np.linspace(-60,60,7)

fig,(ax0,ax1) = plt.subplots(nrows=2,sharex=True ,constrained_layout=True)
ax0.plot(V,Ik);ax0.set(ylabel='IK');ax0.grid()
ax1.plot(V,INa);ax1.set(xlabel='Voltaje [mV]',ylabel='INa');ax1.grid()
fig.suptitle('Curvas de I-V',fontsize=20)
fig.savefig('I_V.png')
#plt.show()

mod = LinearRegression().fit(V.reshape((-1, 1)),Ik)
mod1 = LinearRegression().fit(V.reshape((-1, 1)),INa)
nv = np.arange(-100, 100, 1)
ext = mod.predict(nv.reshape((-1, 1)))
ext1 = mod1.predict(nv.reshape((-1, 1)))

fig,(ax0,ax1) = plt.subplots(nrows=2,sharex=True ,constrained_layout=True)
ax0.plot(nv,ext);ax0.set(ylabel='IK');ax0.grid()
ax1.plot(nv,ext1);ax1.set(xlabel='Voltaje [mV]',ylabel='INa');ax1.grid()
fig.suptitle('Curvas de I-V extrapoladas',fontsize=20)
fig.savefig('I_V_ext.png')
#plt.show()

cu = pt.pend(nv, ext)
cu1 = pt.pend(nv, ext1)

print "Conductancia unitaria del K+= %.3f" %cu
print "Conductancia unitaria del Na+= %.3f" %cu1

fig,(ax0,ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(nrows=7,sharey=True,sharex=True,constrained_layout=True)
ax0.hist(v0[:,1], 10); ax0.set(ylabel='-60 mV')
ax1.hist(v1[:,1], 20); ax1.set(ylabel='-40 mV')
ax2.hist(v2[:,1], 30); ax2.set(ylabel='-20 mV')
ax3.hist(v3[:,1], 40); ax3.set(ylabel='0 mV')
ax4.hist(v4[:,1], 50); ax4.set(ylabel='20 mV')
ax5.hist(v5[:,1], 60); ax5.set(ylabel='40 mV')
ax6.hist(v6[:,1], 70); ax6.set(xlabel='pA', ylabel='60 mV')
fig.set_size_inches(15,10)
fig.suptitle('Histogramas de amplitudes IK',fontsize=20)
fig.savefig('hist_k.png')
#plt.show()

fig,(ax0,ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(nrows=7,sharey=True,sharex=True,constrained_layout=True)
ax0.hist(v0[:,2], 68); ax0.set(ylabel='-60 mV')
ax1.hist(v1[:,2], 58); ax1.set(ylabel='-40 mV')
ax2.hist(v2[:,2], 48); ax2.set(ylabel='-20 mV')
ax3.hist(v3[:,2], 38); ax3.set(ylabel='0 mV')
ax4.hist(v4[:,2], 28); ax4.set(ylabel='20 mV')
ax5.hist(v5[:,2], 18); ax5.set(ylabel='40 mV')
ax6.hist(v6[:,2], 8); ax6.set(xlabel='pA', ylabel='60 mV')
fig.set_size_inches(15,10)
fig.suptitle('Histogramas de amplitudes INa',fontsize=20)
fig.savefig('hist_na.png')
#plt.show()

ti0, tic0, bi0, bic0 = pt.tiempo(v0[:,1], v0[:,0], 0.5)
ti1, tic1, bi1, bic1 = pt.tiempo(v1[:,1], v1[:,0], 0.5)
ti2, tic2, bi2, bic2 = pt.tiempo(v2[:,1], v2[:,0], 0.5)
ti3, tic3, bi3, bic3 = pt.tiempo(v3[:,1], v3[:,0], 0.5)
ti4, tic4, bi4, bic4 = pt.tiempo(v4[:,1], v4[:,0], 0.5)
ti5, tic5, bi5, bic5 = pt.tiempo(v5[:,1], v5[:,0], 0.5)
ti6, tic6, bi6, bic6 = pt.tiempo(v6[:,1], v6[:,0], 0.5)

fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=7, sharex= True,constrained_layout=True)
ax0.hist(ti0, bi0)
ax1.hist(ti1, bi1)
ax2.hist(ti2, bi2)
ax3.hist(ti3, bi3)
ax4.hist(ti4, bi4)
ax5.hist(ti5, bi5)
ax6.hist(ti6, bi6)
#lt.show()

fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=7, sharex= True,constrained_layout=True)
ax0.hist(tic0, bic0)
ax1.hist(tic1, bic1)
ax2.hist(tic2, bic2)
ax3.hist(tic3, bic3)
ax4.hist(tic4, bic4)
ax5.hist(tic5, bic5)
ax6.hist(tic6, bic6)
plt.show()


