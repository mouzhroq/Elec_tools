import numpy as np
import matplotlib.pyplot as plt
import ph_tools as pt

cin = 124
cout = np.loadtxt('pot.txt') 
vk = np.array([])

for i in range (len(cout)):
    vk = np.append(vk, pt.eq_po(cin, cout[i], 1, 37))

print (vk)

fig, ax = plt.subplots()

ax.plot(cout, vk*1000)
ax.set(title="[K+]o vs Vk", xlabel = "[K+]o (mM)", ylabel="VK(mV)" )

plt.show()
fig.savefig('Potencialk.png')

cin = 14
vNa = np.array([])

for i in range (len(cout)):
    vNa = np.append(vNa, pt.eq_po(cin, cout[i], 1, 37))

print (vNa)

fig, ax = plt.subplots()

ax.plot(cout, vNa*1000)
ax.set(title="[Na+]o vs VNa", xlabel = "[Na+]o (mM)", ylabel="VNa(mV)" )

plt.show()
fig.savefig('PotencialNa.png')
