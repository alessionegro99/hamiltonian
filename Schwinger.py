import numpy as np 
import matplotlib.pyplot as plt

n_sites = 4
n_states = 2**n_sites

states = []

for i in range(n_states):
    binary = format(i, '04b')
    foo = [int(bit) for bit in binary]    
    states.append(foo)

print(states)

coeffs = [0]*n_states

## mass term sum_{n=0}^{N-1}(-1)^{n} psi_dag(n) psi(n)
coeffs = np.ones(n_states)
print(states == [0,0,0,0])

