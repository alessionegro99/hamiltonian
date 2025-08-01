import numpy as np
from scipy.linalg import eigh

def main(g):

    #  ^ y
    #  |
    #  B----U2-->---C
    #  |            |
    #  ^            ^
    #  |            |
    #  U3           U1
    #  |            |
    #  A----U0->----D--> x
    #

    # l=1 truncation
    Z3_vals = [-1, 0, 1]
    mod3 = lambda x: (x+3) % 3

    # create full link basis (3**4=81)

    basis_states = []
    for l0 in Z3_vals:
        for l1 in Z3_vals:
            for l2 in Z3_vals:
                for l3 in Z3_vals:
                    basis_states.append([l0, l1, l2, l3])  # links: [top, right, bottom, left]


    # Gauss's law at 4 sites, no charges (divergence ≡ 0 mod 3)

    def gauss_law(state):
        l0, l1, l2, l3 = state

        div_A = mod3(l0 + l3) # site A = (0,0)
        div_B = mod3(-l0 + l1) # site B = (1,0)
        div_C = mod3(-l1 - l2) # site C = (1,1)
        div_D = mod3(l2 - l3) # site D = (1,0)

        return (div_A == 0 and div_B == 0 and div_C == 0 and div_D == 0)

    # filter physical states
    physical_states = [s for s in basis_states if gauss_law(s)]
    phys_dim = len(physical_states)
    print(f"Physical Hilbert space size (OBC): {phys_dim}")
    for i in range(phys_dim):
        state = physical_states[i]
        print(f"Basis state {i}: |{state[0]} {state[1]} {state[2]} {state[3]}>")

    # index lookup table
    def state_to_index(state):
        return physical_states.index(state)

    H = np.zeros((phys_dim, phys_dim), dtype=complex)
    
    for i, state in enumerate(physical_states):
        l0, l1, l2, l3 = state

        # Electric term (diagonal)
        electric_energy = sum(l**2 for l in state)  # l ∈ {-1, 0, +1}, so l^2 \in {0,1}
        H[i, i] += (g / 2.) * electric_energy
        
        # mapping {-1, 0, 1} -> {0, 1, 2}
        tmp = [
            (l0 + 1) ,
            (l1 + 1) ,
            (l2 + 1) ,
            (l3 + 1) ,
        ]
        # magnetic plaquette term U_0U_1U_2^\dagU_3^\dag
        new_state = [
            mod3(tmp[0] + 1) - 1,
            mod3(tmp[1] + 1) - 1,
            mod3(tmp[2] - 1) - 1,
            mod3(tmp[3] - 1) - 1,
        ]
                
        if gauss_law(new_state) and new_state in physical_states:
            j = state_to_index(new_state)
            H[i, j] += -1./(2*g)
            H[j, i] += -1./(2*g)  # Hermitian conjugate
            
    print(H)
    
    eigvals, eigvecs = eigh(H)
    print("\nEigenvalues (sorted):")
    for i, val in enumerate(np.round(eigvals.real, 6)):
        print(f"{i}: {val}")

    print("\nGround state energy:", eigvals[0].real)

if __name__ == "__main__":
    
    g = 1.0 # bare coupling
    
    main(g)