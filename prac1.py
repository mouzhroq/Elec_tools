import ph_tools as pt

K = pt.eq_po(200., 2, 1, 37)
Ca = pt.eq_po(0.001, 1, 2, 37)
Cl= pt.eq_po(25., 250, -1, 37)
Na = pt.eq_po(40., 400, 1, 37)

print "K: %.5f, Ca: %.5f, Cl: %.5f, Na: %.5f" % (K, Ca, Cl, Na)

