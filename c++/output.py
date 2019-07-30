import numpy as np
import matplotlib.pyplot as plt
c=[2779.44,3228.45,3737.39,4786.74,5291.93,4561.15,3505.79,2961.47,]
m=[0.84023,0.808489,0.765175,0.681664,0.548283,0.390608,0.273885,0.21006,]
t=[2.14419,2.1799,2.21561,2.25133,2.28704,2.32276,2.35847,2.39419,]
x=[4097.81,8131.7,1.60098e+06,980313,843110,516710,278219,170047,]
plt.plot(t,m,label="Magnetization")
plt.plot(t,x,label="Succetibility")
plt.plot(t,c,label="Heat Capacity")
plt.legend()
plt.show()
