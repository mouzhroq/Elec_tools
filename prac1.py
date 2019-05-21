import numpy as np
import matplotlib.pyplot as plt
import ph_tools as pt

K = pt.eq_po(200., 2, 1, 37)
Ca = pt.eq_po(0.001, 1, 2, 37)
Cl= pt.eq_po(25., 250, -1, 37)
Na = pt.eq_po(40., 400, 1, 37)

print "K: %.5f, Ca: %.5f, Cl: %.5f, Na: %.5f" % (K, Ca, Cl, Na)

prkna = pt.res_po(200., 2, 40, 400, 25, 250, 0, 0, 100, 1, 0, 0, 37)

print "Potencial de reposo: %.5f" % prkna

prknacl = pt.res_po(200., 2, 40, 400, 25, 250, 0, 0, 100, 1, 10, 0, 37)

print "Potencial de reposo: %.5f" % prknacl

p1 = pt.res_po(150, 150, 10, 90, 50, 250, 1E-4, 5, 1, 0, 1, 0, 37)
p2 = pt.res_po(150, 150, 10, 90, 50, 250, 1E-4, 5, 1, 10, 1, 0, 37)
p3 = pt.res_po(150, 150, 10, 90, 50, 250, 1E-4, 5, 1, 10, 10, 0, 37)

t = np.arange(7)
v = np.array([p1, p1, p2, p2, p2, p3, p3])

plt.plot(t, v)
plt.show()