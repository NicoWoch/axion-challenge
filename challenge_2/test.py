import numpy as np

from quantum_simulator import QuantumState, Gate


def main():
    # -------- Bramka Hadamard --------

    state_0 = QuantumState.zero
    
    print('\n-- Hadamard Gate --\n')
    print('|0> =', state_0)
    print('H |0> =', state_0 >> Gate.H)
    print('H H |0> =', state_0 >> Gate.H >> Gate.H)
    
    
    # -------- Stan Bella --------
    
    psi_0 = state_0 | state_0
    psi_1 = psi_0 >> (Gate.H | Gate.I)  # bramka H działa na bit pierwszy, bit drugi bez zmian
    psi_2 = psi_1 >> (Gate.CNOT)  # pierwszy bit kontrolny, drugi target
    
    state_entangled = psi_2
    
    print('\n-- Entangled state of 2 qubits --\n')
    print('Bell State = ', state_entangled)
    print('Measurement of qubit 0 -->', state_entangled.measure(0), '\n')
    print('Current State = ', state_entangled)
    print('Measurement of qubits 0 and 1 -->', state_entangled.measure(0, 1), '\n')
    
    
    # -------- Test Bramki Ry(pi/4) --------
        
    print('\n-- Gate Ry(pi/4) on state |0> --\n')
    print(state_0 >> Gate.Ry(np.pi / 4))
    
    
    # -------- Phase Kickback --------
    
    plus_state = QuantumState.zero >> Gate.H
    minus_state = QuantumState.one >> Gate.H
    
    psi_0 = plus_state | minus_state
    psi_1 = psi_0 >> Gate.CNOT
    
    print('\n-- Phase kickback using CNOT on |+-> state --\n')
    print('|+-> =', psi_0)
    print('CNOT |+-> =', psi_1)
    


if __name__ == '__main__':
    main()
