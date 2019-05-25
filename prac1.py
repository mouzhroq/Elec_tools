import numpy as np
import matplotlib.pyplot as plt
import ph_tools as pt

K = pt.eq_po(400., 20, 1, 23)
Ca = pt.eq_po(0.4E-4, 10, 2, 23)
Cl= pt.eq_po(150., 560, -1, 23)
Na = pt.eq_po(50., 440, 1, 23)

print "K: %.5f, Ca: %.5f, Cl: %.5f, Na: %.5f" % (K, Ca, Cl, Na)

prkna = pt.res_po(400., 20, 50, 440, 150, 560, 0.4E-4, 10, 90, 1, 0, 0, 37)

print "Potencial de reposo: %.5f" % prkna

prknacl = pt.res_po(400., 20, 50, 440, 150, 560, 0.4E-4, 10, 90, 1, 1./5, 0, 37)

print "Potencial de reposo: %.5f" % prknacl

p1 = pt.res_po(140, 2.25, 10.4, 109, 1.5, 77.5, 5E-4, 2.1, 1, 0, 1, 0, 37)/1E-3
p2 = pt.res_po(140, 2.25, 10.4, 109, 1.5, 77.5, 5E-4, 2.1, 1, 10, 1, 0, 37)/1E-3
p3 = pt.res_po(140, 2.25, 10.4, 109, 1.5, 77.5, 5E-4, 2.1, 1, 10, 10, 0, 37)/1E-3

t = np.arange (0, 5 , 0.01)
v = np.array([]); v = np.append(v, [p1]*100) ; v = np.append(v, [p2]*300); v =np.append(v, [p3]*100)

fig, ax = plt.subplots()

ax.plot(t, v)

ax.set(xlabel = "Tiempo (s)", ylabel = "Voltaje (mV)", title = "Respuesta de la celula")

fig.savefig("Respuesta.png")
plt.show()

txt = np.loadtxt('IV.txt')

fig, (ax0, ax1, ax2) = plt.subplots(nrows = 3, sharex = True, constrained_layout=True)

ax0.plot(txt[:,0], txt[:,1])
ax1.plot(txt[:,0], txt[:,2])
ax2.plot(txt[:,0], txt[:,3])

ax0.set(ylabel = "Corriente (pA)", title = "Corriente 1"); ax0.grid()
ax1.set(ylabel = "Corriente (pA)", title = "Corriente 2"); ax1.grid()
ax2.set(xlabel = "Voltaje (mV)", ylabel = "Corriente (pA)", title = "Corriente 3"); ax2.grid()

fig.savefig("Corrientes.png")
plt.show()

print "Conductancia 1: %.3f" % pt.pend (txt[10,1], txt[0,1], txt[10,0], txt[0,0])
print "Conductancia 2: %.3f" % pt.pend (txt[10,2], txt[0,2], txt[10,0], txt[0,0])
print "Conductancia 3: %.3f" % pt.pend (txt[10,3], txt[0,3], txt[10,0], txt[0,0])

