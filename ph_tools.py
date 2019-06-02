# Import libraries
import numpy as np


R = 8.3149
F = 96485

# Calculo del potencial de equilibrio
def eq_po (con_in, con_ex, z, t):
    T = t + 273.15
    return (((R*T)/(z*F))*np.log(con_ex/con_in))


#Calculo del potencial de reposo
def res_po (k_in, k_out, na_in, na_out, cl_in, cl_out, ca_in, ca_out, pk, pna, pcl, pca, t):
    T = t + 273.15
    return (R*T/F)*np.log(((pk * k_out) + (pna * na_out) + (pcl * cl_in) + (pca * ca_out))/( (pk * k_in) + (pna * na_in) + (pcl * cl_out) + (pca * ca_in)))

#Calculo de la pendiente
def pend (x, y):
    return (y[len(y)-1] - y[0]) / (x[len(x)-1] - x[0])


