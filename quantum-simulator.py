import random
from types import UnionType
from typing import Any, Callable, Iterator

import numpy as np
from numpy.typing import NDArray


class QuantumState:
    def __init__(self, initial_state: np.typing.ArrayLike) -> None:
        self._state: NDArray[np.complex256] = np.asarray(initial_state, dtype=np.complex256)
        self.qubits_count = int(round(np.log2(self._state.shape[0])))
        
        assert len(self._state.shape) == 2 and self._state.shape[1] == 1, 'Quantum state should be a vector'
        assert self._state.shape[0] == 2 ** self.qubits_count, 'Quantum state should have power of two rows counts'
    
    @classmethod
    def zero(cls) -> QuantumState:
        return QuantumState([[1], [0]])
    
    @classmethod
    def one(cls) -> QuantumState:
        return QuantumState([[0], [1]])
    
    def _iterstate(self) -> Iterator[tuple[int, tuple[bool, ...]]]:
        for i in range(len(self._state)):
            yield i, tuple(((i >> qubit) & 1) == 1 for qubit in range(self.qubits_count))
    
    def measure(self, *qubits: int) -> dict[int, int]:
        collapses: dict[int, int] = {}
        
        for qubit in qubits:
            propability_of_one = 0
            
            for i, q in self._iterstate():
                if q[qubit]:
                    propability_of_one += np.abs(self._state[i]) ** 2
                    
            if random.random() < propability_of_one:
                collapses[qubit] = 1
            else:
                collapses[qubit] = 0
                
        for i, q in self._iterstate():
            if any(qubit in collapses and collapses[qubit] != q[qubit] for qubit in range(self.qubits_count)):
                self._state[i, 0] = 0
            
        self.normalize()

        return collapses
    
    def normalize(self) -> None:
        length = np.sqrt(np.sum(self._state ** 2))
        self._state /= length
    
    def join(self, other: QuantumState) -> QuantumState:
        return QuantumState(np.kron(self._state, other._state))
    
    __or__ = join
    
    def apply(self, gate: Gate | Callable[[], Gate]) -> QuantumState:
        if callable(gate):
            gate = gate()
        
        assert gate.size == self._state.shape[0]
        
        return QuantumState(gate.matrix @ self._state)
    
    __rshift__ = apply

    def __repr__(self) -> str:
        return f'<QuantumState {self._state}>'
    
    def __str__(self) -> str:
        qubits = int(np.ceil(np.log2(self._state.shape[0])))
        out = f'QuantumState(qubits = {qubits}):\n'
        
        for i, qubits in self._iterstate():
            ket = '|' + ''.join(
                ('1' if qubit else '0') for qubit in qubits
            ) + '>'
            
            amplitude = np.round(self._state[i, 0], 6)
            
            if amplitude.imag == 0:
                amplitude = amplitude.real
            
            out += f'amplitude {ket} = {amplitude}\n'

        
        return out


class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, _, owner):
        return self.func(owner)
    

class Gate:
    def __init__(self, matrix: Any) -> None:
        self.matrix = np.asarray(matrix, dtype=np.complex256)
        
        assert len(self.matrix.shape) == 2, 'A gate needs to be a two dimensional matrix'
        assert self.matrix.shape[0] == self.matrix.shape[1], 'A gate needs to be a square matrix'
        
    @property
    def size(self) -> int:
        return self.matrix.shape[0]
    
    def join(self, other: Gate) -> Gate:
        return Gate(np.kron(self.matrix, other.matrix))
    
    __or__ = join

    INV_SQRT_2 = 1 / np.sqrt(2)
    
    @classproperty
    def I(cls) -> Gate:
        return Gate([[1, 0], [0, 1]])
    
    @classproperty
    def X(cls) -> Gate:
        return Gate([[0, 1], [1, 0]])
    
    @classproperty
    def Y(cls) -> Gate:
        return Gate([[0, -1j], [1j, 0]])
        
    @classproperty
    def Z(cls) -> Gate:
        return Gate([[1, 0], [0, -1]])
    
    @classproperty
    def H(cls) -> Gate:
        return Gate(cls.INV_SQRT_2 * np.array([
            [1, 1],
            [1, -1]
        ]))
        
    @classmethod
    def Ry(cls, theta: float) -> Gate:
        return Gate([
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)]
        ])
        
    @classproperty
    def CNOT(cls) -> Gate:
        return Gate([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])


def main():
    state_0 = QuantumState.zero()
    
    print(state_0)
    print(state_0 >> Gate.H)
    print(state_0 >> Gate.H >> Gate.H)
    
    print(state_0 >> Gate.Ry(0.15 * np.pi) >> Gate.Ry(0.85 * np.pi) >> Gate.Ry(-np.pi))
    
    psi_0 = state_0.join(state_0)
    psi_1 = psi_0.apply(Gate.H | Gate.I)
    psi_2 = psi_1.apply(Gate.CNOT)
    
    state_entangled = psi_2
    
    print('\n-- Entangled state of 2 qubits --')
    print(state_entangled)
    print(state_entangled.measure(0))
    print(state_entangled)
    print(state_entangled.measure(0, 1))
    print(state_entangled)
    
    plus_state = QuantumState.zero() >> Gate.H
    minus_state = QuantumState.one() >> Gate.H
    
    psi_0 = plus_state | minus_state
    psi_1 = psi_0 >> Gate.CNOT
    
    print('\n-- Phase kickback using CNOT on (plus, minus) state --')
    print(psi_0)
    print(psi_1)
    


if __name__ == '__main__':
    main()
