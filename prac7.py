import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import ph_tools as pt 

c1 = np.loadtxt('Corriente_1.txt')
c1[:,0] = c1[:,0] / 1E-3
c1[:,1::] = c1[:,1::] / 1E-12
c2 = np.loadtxt('Corriente_2.txt')
c2[:,0] = c2[:,0] / 1E-3
c2[:,1::] = c2[:,1::] / 1E-12
c3 = np.loadtxt('Corriente_3.txt')
c3[:,0] = c3[:,0] / 1E-3
c3[:,1::] = c3[:,1::] / 1E-12
iso = np.array([0, 1.25, 3.5, 3.75, 4.9])
N = iso / c1[1,0]


fig, (ax0, ax1, ax2)=plt.subplots(nrows=3,sharex=True, constrained_layout=True)
ax0.plot(c1[:,0],c1[:, 1:21]); ax0.set(title = 'Corriente 1', ylabel = 'Corriente (pA)')
ax1.plot(c2[:,0],c2[:, 1:21]); ax1.set(title = 'Corriente 2', ylabel = 'Corriente (pA)')
ax2.plot(c3[:,0],c3[:, 1:21]); ax2.set(title = 'Corriente 3', ylabel = 'Corriente (pA)', xlabel = 't (ms)')
fig.suptitle('Trazos de corrientes')
#plt.show(block=False)
fig.savefig('Trazosp7.png')

fig, (ax0, ax1, ax2)=plt.subplots(nrows=3,sharex=True, constrained_layout=True)
ax0.plot(c1[int(N[0]), 1::]); ax0.set(title='Corriente 1', ylabel='Corriente (pA)')
ax1.plot(c2[int(N[0]), 1::]); ax1.set(title='Corriente 2', ylabel='Corriente (pA)')
ax2.plot(c3[int(N[0]), 1::]); ax2.set(title='Corriente 3', ylabel='Corriente (pA)', xlabel='Numero de trazo')
fig.suptitle('Grafica isocrona tiempo 0 ms')
#plt.show(block=False)
fig.savefig('iso1p7.png')


fig, (ax0, ax1, ax2)=plt.subplots(nrows=3,sharex=True, constrained_layout=True)
ax0.plot(c1[int(N[1]), 1::]); ax0.set(title='Corriente 1', ylabel='Corriente (pA)')
ax1.plot(c2[int(N[1]), 1::]); ax1.set(title='Corriente 2', ylabel='Corriente (pA)')
ax2.plot(c3[int(N[1]), 1::]); ax2.set(title='Corriente 3', ylabel='Corriente (pA)', xlabel='Numero de trazo')
fig.suptitle('Grafica isocrona tiempo 1.25 ms')
#plt.show(block=False)
fig.savefig('iso2p7.png')

fig, (ax0, ax1, ax2)=plt.subplots(nrows=3,sharex=True, constrained_layout=True)
ax0.plot(c1[int(N[2]), 1::]); ax0.set(title='Corriente 1', ylabel='Corriente (pA)')
ax1.plot(c2[int(N[2]), 1::]); ax1.set(title='Corriente 2', ylabel='Corriente (pA)')
ax2.plot(c3[int(N[2]), 1::]); ax2.set(title='Corriente 3', ylabel='Corriente (pA)', xlabel='Numero de trazo')
fig.suptitle('Grafica isocrona tiempo 2.5 ms')
#plt.show(block=False)
fig.savefig('iso3p7.png')

fig, (ax0, ax1, ax2)=plt.subplots(nrows=3,sharex=True, constrained_layout=True)
ax0.plot(c1[int(N[3]), 1::]); ax0.set(title='Corriente 1', ylabel='Corriente (pA)')
ax1.plot(c2[int(N[3]), 1::]); ax1.set(title='Corriente 2', ylabel='Corriente (pA)')
ax2.plot(c3[int(N[3]), 1::]); ax2.set(title='Corriente 3', ylabel='Corriente (pA)', xlabel='Numero de trazo')
fig.suptitle('Grafica isocrona tiempo 3.75 ms')
#plt.show(block=False)
fig.savefig('iso4p7.png')

fig, (ax0, ax1, ax2)=plt.subplots(nrows=3,sharex=True, constrained_layout=True)
ax0.plot(c1[int(N[4]), 1::]); ax0.set(title='Corriente 1', ylabel='Corriente (pA)')
ax1.plot(c2[int(N[4]), 1::]); ax1.set(title='Corriente 2', ylabel='Corriente (pA)')
ax2.plot(c3[int(N[4]), 1::]); ax2.set(title='Corriente 3', ylabel='Corriente (pA)', xlabel='Numero de trazo')
fig.suptitle('Grafica isocrona tiempo 4.9 ms')
#plt.show(block=False)
fig.savefig('iso5p7.png')

fig, ax0 = plt.subplots(constrained_layout=True)
ax0.plot(c1[:,0], np.mean(c1[:,1::], axis=1))
ax0.set(title='Corriente 1', xlabel='t (ms)', ylabel='Media')
ax01 = ax0.twinx()
ax01.plot(c1[:,0], np.var(c1[:,1::], axis=1))
ax01.set(ylabel='Varianza')
#plt.show()
fig.savefig('mv1pc7.png')

fig, ax1 = plt.subplots(constrained_layout=True)
ax1.plot(c2[:,0], np.mean(c2[:,1::], axis=1))
ax1.set(title='Corriente 2', xlabel='t (ms)', ylabel='Media')
ax11 = ax1.twinx()
ax11.plot(c2[:,0], np.var(c2[:,1::], axis=1))
ax11.set(ylabel='Varianza')
#plt.show()
fig.savefig('mv2pc7.png')

fig, ax2 = plt.subplots(constrained_layout=True)
ax2.plot(c3[:,0], np.mean(c3[:,1::], axis=1))
ax2.set(title='Corriente 3', xlabel='t (ms)', ylabel='Media')
ax21 = ax2.twinx()
ax21.plot(c3[:,0], np.var(c3[:,1::], axis=1))
ax21.set(ylabel='Varianza')
#plt.show()
fig.savefig('mv3pc7.png')

fig, (ax0, ax1, ax2)=plt.subplots(nrows=3, constrained_layout=True)
ax0.plot(np.mean(c1[:,1::], axis=1), np.var(c1[:,1::], axis=1))
ax0.set(title='Corriente 1', xlabel='Corriente promedio', ylabel='Varianza')
ax1.plot(np.mean(c2[:,1::], axis=1), np.var(c2[:,1::], axis=1))
ax1.set(title='Corriente 2', xlabel='Corriente promedio', ylabel='Varianza')
ax2.plot(np.mean(c3[:,1::], axis=1), np.var(c3[:,1::], axis=1))
ax2.set(title='Corriente 3', xlabel='Corriente promedio', ylabel='Varianza')
fig.suptitle('Relacion varianza - corriente promedio')
fig.savefig('vvsmprac7.png')
#plt.show()

def func (I, i, N):
    return I*i-(I**2 / N)

popt, pcov = curve_fit(func, np.mean(c1[:,1::], axis=1), np.var(c1[:,1::], axis=1))
print "Los valores del ajuste son: i = %.2f pA & N = %.2f canales." %(popt[0], popt[1])

fig, ax0 = plt.subplots(constrained_layout=True)
ax0.plot(np.mean(c1[:,1::], axis=1), np.var(c1[:,1::], axis=1))
ax0.plot(np.mean(c1[:,1::], axis=1), func(np.mean(c1[:,1::], axis=1), *popt))
ax0.set(title='Ajuste en corriente 1', xlabel='Corriente promedio', ylabel='Varianza')
fig.savefig('ajuste1p7.png')
#plt.show()

popt1, pcov1 = curve_fit(func, np.mean(c2[:,1::], axis=1), np.var(c2[:,1::], axis=1))
print "Los valores del ajuste son: i = %.2f pA & N = %.2f canales." %(popt1[0], popt1[1])

fig, ax1 = plt.subplots(constrained_layout=True)
ax1.plot(np.mean(c2[:,1::], axis=1), np.var(c2[:,1::], axis=1))
ax1.plot(np.mean(c2[:,1::], axis=1), func(np.mean(c2[:,1::], axis=1), *popt1))
ax1.set(title='Ajuste en corriente 2', xlabel='Corriente promedio', ylabel='Varianza')
fig.savefig('ajuste2p7.png')
#plt.show()

popt2, pcov2 = curve_fit(func, np.mean(c3[:,1::], axis=1), np.var(c3[:,1::], axis=1))
print "Los valores del ajuste son: i = %.2f pA & N = %.2f canales." %(popt2[0], popt2[1])

fig, ax2 = plt.subplots(constrained_layout=True)
ax2.plot(np.mean(c3[:,1::], axis=1), np.var(c3[:,1::], axis=1))
ax2.plot(np.mean(c3[:,1::], axis=1), func(np.mean(c3[:,1::], axis=1), *popt2))
ax2.set(title='Ajuste en corriente 3', xlabel='Corriente promedio', ylabel='Varianza')
fig.savefig('ajuste3p7.png')
#plt.show()

p = np.array([np.max(np.mean(c1[:,1::], axis=1))/(popt[0]*popt[1]), np.max(np.mean(c2[:,1::], axis=1))/(popt1[0]*popt1[1]), np.max(np.mean(c3[:,1::], axis=1))/(popt2[0]*popt2[1])])
print "La probabilidad maxima de la corriente 1 es: %.3f" %p[0]
print "La probabilidad maxima de la corriente 2 es: %.3f" %p[1]
print "La probabilidad maxima de la corriente 3 es: %.3f" %p[2]
