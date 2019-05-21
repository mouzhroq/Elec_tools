# Import libraries
import numpy as np

def eq_po (con_in, con_ex, z, t):
    R = 8.3149
    T = t + 273.15
    F = 96485
    return (((R*T)/(z*F))*np.log(con_ex/con_in))

def res_po ()