import numpy as np 
import matplotlib.pyplot as plt

n_sites = 4
n_states = 2**n_sites

x, mu, eps = 0.6, 0.1, 0.0

sigma_x = np.array([[0, 1],
                    [1, 0]], dtype=complex)

sigma_y = np.array([[0, -1j],
                    [1j, 0]], dtype=complex)

sigma_z = np.array([[1, 0],
                    [0, -1]], dtype=complex)

sigma_p  = 0.5 * (sigma_x + 1j * sigma_y)
sigma_m = 0.5 * (sigma_x - 1j * sigma_y)

I = np.eye(2, dtype=complex)

zero = np.array([[1], [0]], dtype=complex)
one = np.array([[0], [1]], dtype=complex)

basis_states = np.eye(n_states, dtype=complex)
    
## hopping term
hopping = np.zeros((n_states, n_states), dtype=complex)

for n in range(0, n_sites-1):
    A = np.eye(2**(n), dtype=complex)
    C = np.eye(2**(n_sites-n-2), dtype=complex)
    ##B = np.kron(I,sigma_p) @ np.kron(sigma_m,I)
    B = np.kron(sigma_p, sigma_m)
    hopping += np.kron(A, np.kron(B, C))
    ##B = np.kron(I,sigma_m) @ np.kron(sigma_p,I)
    B = np.kron(sigma_m, sigma_p)
    hopping += np.kron(A, np.kron(B, C))
    
hopping *= x

## mass term

mass = np.zeros((n_states, n_states), dtype=complex)

for n in range(0, n_sites):
    A = np.eye(2**(n), dtype=complex)
    C = np.eye(2**(n_sites-n-1), dtype=complex)
    B = sigma_z
    mass += (-1)**(n+1)*np.kron(A, np.kron(B, C))
    
mass *= 0.5*mu

## electric term

el = np.zeros((n_states, n_states), dtype=complex)

for n in range(0, n_sites-1):
    tmp = eps*np.eye(2**n_sites, dtype=complex)
    for m in range(n+1):
        A = np.eye(2**m, dtype=complex)
        C = np.eye(2**(n_sites-m-1), dtype=complex)
        B = sigma_z
        tmp += 0.5*(np.kron(A, np.kron(B, C))+(-1)**(m+1)*np.eye(2**n_sites, dtype=complex))
    
    el += tmp@tmp

H = hopping + el + mass

eigv = np.linalg.eigvals(H)

print(max(eigv) - min(eigv))
print(4.211+2.844)

plt.imshow(np.real(H), cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title("Heatmap of Matrix")
plt.xlabel("Column Index")
plt.ylabel("Row Index")
plt.show()

