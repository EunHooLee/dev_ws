import numpy as np

x = 10
mu = 0
sigma = 1.0
dt = 0.01

y = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))
print("likelihood : {}".format(y))

for i in range(0,x):
    y1 = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(i-mu)**2/(2*sigma**2))
    prob =  y1 * dt
    i += dt

print("probability: {}".format(y1))


G = ['A','b','C']
for i,_ in enumerate(G):
    print(_)