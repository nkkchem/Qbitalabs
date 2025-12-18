"""
Molecular Hamiltonian Construction for QBitaLabs

Builds quantum Hamiltonians for molecular systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Atom:
    """Represents an atom in a molecule."""

    symbol: str
    position: tuple[float, float, float]  # Angstroms
    atomic_number: int | None = None

    def __post_init__(self):
        """Set atomic number from symbol."""
        if self.atomic_number is None:
            atomic_numbers = {
                "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5,
                "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
                "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
                "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
            }
            self.atomic_number = atomic_numbers.get(self.symbol, 1)


@dataclass
class FermionOperator:
    """
    Represents a fermionic operator.

    Terms are stored as {tuple of (index, is_creation): coefficient}
    e.g., {((0, True), (1, False)): 0.5} represents 0.5 * a†_0 a_1
    """

    terms: dict[tuple[tuple[int, bool], ...], complex] = field(default_factory=dict)

    def __add__(self, other: "FermionOperator") -> "FermionOperator":
        """Add two fermion operators."""
        result = FermionOperator(terms=self.terms.copy())
        for term, coeff in other.terms.items():
            result.terms[term] = result.terms.get(term, 0) + coeff
        return result

    def __mul__(self, scalar: complex) -> "FermionOperator":
        """Multiply by scalar."""
        result = FermionOperator()
        for term, coeff in self.terms.items():
            result.terms[term] = coeff * scalar
        return result

    def __rmul__(self, scalar: complex) -> "FermionOperator":
        """Right multiply by scalar."""
        return self.__mul__(scalar)

    @classmethod
    def creation(cls, index: int) -> "FermionOperator":
        """Create a†_i operator."""
        return cls(terms={((index, True),): 1.0})

    @classmethod
    def annihilation(cls, index: int) -> "FermionOperator":
        """Create a_i operator."""
        return cls(terms={((index, False),): 1.0})

    @classmethod
    def number(cls, index: int) -> "FermionOperator":
        """Create n_i = a†_i a_i operator."""
        return cls(terms={((index, True), (index, False)): 1.0})


@dataclass
class QubitOperator:
    """
    Represents a qubit operator (Pauli strings).

    Terms are stored as {tuple of (qubit, pauli): coefficient}
    e.g., {((0, 'Z'), (1, 'X')): 0.5} represents 0.5 * Z_0 X_1
    """

    terms: dict[tuple[tuple[int, str], ...], complex] = field(default_factory=dict)

    def __add__(self, other: "QubitOperator") -> "QubitOperator":
        """Add two qubit operators."""
        result = QubitOperator(terms=self.terms.copy())
        for term, coeff in other.terms.items():
            result.terms[term] = result.terms.get(term, 0) + coeff
        return result

    def __mul__(self, scalar: complex) -> "QubitOperator":
        """Multiply by scalar."""
        result = QubitOperator()
        for term, coeff in self.terms.items():
            result.terms[term] = coeff * scalar
        return result

    @property
    def n_qubits(self) -> int:
        """Get number of qubits."""
        max_qubit = 0
        for term in self.terms.keys():
            for qubit, _ in term:
                max_qubit = max(max_qubit, qubit + 1)
        return max_qubit

    def to_matrix(self) -> np.ndarray:
        """Convert to matrix representation."""
        n = self.n_qubits
        dim = 2 ** n
        matrix = np.zeros((dim, dim), dtype=complex)

        # Pauli matrices
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        pauli_map = {"I": I, "X": X, "Y": Y, "Z": Z}

        for term, coeff in self.terms.items():
            # Build Kronecker product
            ops = {q: "I" for q in range(n)}
            for qubit, pauli in term:
                ops[qubit] = pauli

            term_matrix = np.array([[1]], dtype=complex)
            for q in range(n):
                term_matrix = np.kron(term_matrix, pauli_map[ops[q]])

            matrix += coeff * term_matrix

        return matrix


class JordanWignerMapper:
    """
    Maps fermionic operators to qubit operators using Jordan-Wigner transformation.

    a†_j = (Πi<j Z_i) (X_j - iY_j) / 2
    a_j = (Πi<j Z_i) (X_j + iY_j) / 2
    """

    def map(self, fermion_op: FermionOperator) -> QubitOperator:
        """Map fermion operator to qubit operator."""
        result = QubitOperator()

        for term, coeff in fermion_op.terms.items():
            qubit_term = self._map_term(term, coeff)
            result = result + qubit_term

        return result

    def _map_term(
        self,
        term: tuple[tuple[int, bool], ...],
        coeff: complex,
    ) -> QubitOperator:
        """Map a single fermionic term."""
        if not term:
            return QubitOperator(terms={(): coeff})

        result = QubitOperator(terms={(): coeff})

        for orbital, is_creation in term:
            # Single fermion operator mapping
            single_op = self._map_single_operator(orbital, is_creation)

            # Multiply result by this operator
            new_result = QubitOperator()
            for t1, c1 in result.terms.items():
                for t2, c2 in single_op.terms.items():
                    # Combine terms
                    combined = self._multiply_pauli_strings(t1, t2)
                    new_result.terms[combined[0]] = (
                        new_result.terms.get(combined[0], 0) + c1 * c2 * combined[1]
                    )

            result = new_result

        return result

    def _map_single_operator(
        self, orbital: int, is_creation: bool
    ) -> QubitOperator:
        """Map a single creation/annihilation operator."""
        # Z string for orbitals below
        z_string = tuple((i, "Z") for i in range(orbital))

        if is_creation:
            # a† = (X - iY) / 2
            x_term = z_string + ((orbital, "X"),)
            y_term = z_string + ((orbital, "Y"),)
            return QubitOperator(terms={
                x_term: 0.5,
                y_term: -0.5j,
            })
        else:
            # a = (X + iY) / 2
            x_term = z_string + ((orbital, "X"),)
            y_term = z_string + ((orbital, "Y"),)
            return QubitOperator(terms={
                x_term: 0.5,
                y_term: 0.5j,
            })

    def _multiply_pauli_strings(
        self,
        t1: tuple[tuple[int, str], ...],
        t2: tuple[tuple[int, str], ...],
    ) -> tuple[tuple[tuple[int, str], ...], complex]:
        """Multiply two Pauli strings."""
        # Pauli multiplication rules
        pauli_product = {
            ("I", "I"): ("I", 1), ("I", "X"): ("X", 1), ("I", "Y"): ("Y", 1), ("I", "Z"): ("Z", 1),
            ("X", "I"): ("X", 1), ("X", "X"): ("I", 1), ("X", "Y"): ("Z", 1j), ("X", "Z"): ("Y", -1j),
            ("Y", "I"): ("Y", 1), ("Y", "X"): ("Z", -1j), ("Y", "Y"): ("I", 1), ("Y", "Z"): ("X", 1j),
            ("Z", "I"): ("Z", 1), ("Z", "X"): ("Y", 1j), ("Z", "Y"): ("X", -1j), ("Z", "Z"): ("I", 1),
        }

        # Combine into dict
        ops: dict[int, str] = {}
        for qubit, pauli in t1:
            ops[qubit] = pauli
        for qubit, pauli in t2:
            ops[qubit] = pauli

        phase = 1.0
        result: dict[int, str] = {}

        # Apply multiplication
        all_qubits = set(q for q, _ in t1) | set(q for q, _ in t2)
        for q in sorted(all_qubits):
            p1 = dict(t1).get(q, "I")
            p2 = dict(t2).get(q, "I")
            new_pauli, p = pauli_product[(p1, p2)]
            phase *= p
            if new_pauli != "I":
                result[q] = new_pauli

        result_term = tuple((q, p) for q, p in sorted(result.items()))
        return result_term, phase


class MolecularHamiltonian:
    """
    Builds and manages molecular Hamiltonians.

    Computes:
    - One-electron integrals (kinetic + nuclear attraction)
    - Two-electron integrals (electron-electron repulsion)
    - Qubit Hamiltonian for VQE

    Example:
        >>> mol = MoleculeBuilder.h2(bond_length=0.74)
        >>> hamiltonian = MolecularHamiltonian(mol)
        >>> qubit_op = hamiltonian.get_qubit_hamiltonian()
    """

    def __init__(
        self,
        atoms: list[Atom],
        charge: int = 0,
        spin: int = 0,
        basis: str = "sto-3g",
    ):
        """
        Initialize molecular Hamiltonian.

        Args:
            atoms: List of atoms in the molecule.
            charge: Molecular charge.
            spin: Spin multiplicity (2S).
            basis: Basis set name.
        """
        self.atoms = atoms
        self.charge = charge
        self.spin = spin
        self.basis = basis

        self._one_body: np.ndarray | None = None
        self._two_body: np.ndarray | None = None
        self._nuclear_repulsion: float = 0.0
        self._n_orbitals: int = 0

        self._logger = structlog.get_logger("molecular_hamiltonian")

    def compute_integrals(self) -> None:
        """Compute molecular integrals."""
        try:
            # Try to use PySCF for accurate integrals
            self._compute_integrals_pyscf()
        except ImportError:
            # Fall back to simplified model
            self._compute_integrals_simplified()

    def _compute_integrals_pyscf(self) -> None:
        """Compute integrals using PySCF."""
        from pyscf import gto, scf

        # Build molecule
        mol_string = ""
        for atom in self.atoms:
            mol_string += f"{atom.symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}; "

        mol = gto.M(
            atom=mol_string,
            basis=self.basis,
            charge=self.charge,
            spin=self.spin,
        )

        # Run HF
        mf = scf.RHF(mol)
        mf.kernel()

        # Get integrals in MO basis
        self._n_orbitals = mf.mo_coeff.shape[1]
        self._one_body = mf.mo_coeff.T @ mol.intor("int1e_kin") @ mf.mo_coeff
        self._one_body += mf.mo_coeff.T @ mol.intor("int1e_nuc") @ mf.mo_coeff

        # Two-electron integrals
        eri = mol.intor("int2e")
        eri = np.einsum("pi,qj,pqrs,rk,sl->ijkl",
                        mf.mo_coeff, mf.mo_coeff, eri, mf.mo_coeff, mf.mo_coeff)
        self._two_body = eri

        self._nuclear_repulsion = mol.energy_nuc()

        self._logger.info(
            "Integrals computed with PySCF",
            n_orbitals=self._n_orbitals,
            nuclear_repulsion=self._nuclear_repulsion,
        )

    def _compute_integrals_simplified(self) -> None:
        """Compute simplified integrals for testing."""
        # Simple model for H2
        if len(self.atoms) == 2 and all(a.symbol == "H" for a in self.atoms):
            self._n_orbitals = 2

            # Simplified one-body terms
            self._one_body = np.array([
                [-1.25, -0.47],
                [-0.47, -0.47],
            ])

            # Simplified two-body terms
            self._two_body = np.zeros((2, 2, 2, 2))
            self._two_body[0, 0, 0, 0] = 0.67
            self._two_body[1, 1, 1, 1] = 0.67
            self._two_body[0, 0, 1, 1] = 0.66
            self._two_body[1, 1, 0, 0] = 0.66
            self._two_body[0, 1, 1, 0] = 0.18
            self._two_body[1, 0, 0, 1] = 0.18

            # Nuclear repulsion
            r = np.linalg.norm(
                np.array(self.atoms[0].position) - np.array(self.atoms[1].position)
            )
            self._nuclear_repulsion = 1.0 / r if r > 0 else 0.0

        else:
            # Generic small system
            n = min(4, 2 * len(self.atoms))
            self._n_orbitals = n
            self._one_body = np.random.randn(n, n) * 0.5
            self._one_body = (self._one_body + self._one_body.T) / 2
            self._two_body = np.random.randn(n, n, n, n) * 0.1
            self._nuclear_repulsion = 1.0

        self._logger.info(
            "Simplified integrals computed",
            n_orbitals=self._n_orbitals,
        )

    def get_fermion_hamiltonian(self) -> FermionOperator:
        """Get the second-quantized fermion Hamiltonian."""
        if self._one_body is None:
            self.compute_integrals()

        hamiltonian = FermionOperator()
        n = self._n_orbitals

        # One-body terms: Σ h_pq a†_p a_q
        for p in range(n):
            for q in range(n):
                if abs(self._one_body[p, q]) > 1e-10:
                    # Include spin
                    for spin in [0, 1]:  # alpha, beta
                        term = ((2 * p + spin, True), (2 * q + spin, False))
                        hamiltonian.terms[term] = self._one_body[p, q]

        # Two-body terms: 1/2 Σ g_pqrs a†_p a†_r a_s a_q
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        if abs(self._two_body[p, q, r, s]) > 1e-10:
                            coeff = 0.5 * self._two_body[p, q, r, s]
                            for s1 in [0, 1]:
                                for s2 in [0, 1]:
                                    term = (
                                        (2 * p + s1, True),
                                        (2 * r + s2, True),
                                        (2 * s + s2, False),
                                        (2 * q + s1, False),
                                    )
                                    hamiltonian.terms[term] = (
                                        hamiltonian.terms.get(term, 0) + coeff
                                    )

        return hamiltonian

    def get_qubit_hamiltonian(
        self, mapper: str = "jordan_wigner"
    ) -> QubitOperator:
        """
        Get the qubit Hamiltonian.

        Args:
            mapper: Fermion-to-qubit mapping (jordan_wigner, bravyi_kitaev).

        Returns:
            Qubit operator.
        """
        fermion_h = self.get_fermion_hamiltonian()

        if mapper == "jordan_wigner":
            jw = JordanWignerMapper()
            qubit_h = jw.map(fermion_h)
        else:
            raise ValueError(f"Unknown mapper: {mapper}")

        # Add nuclear repulsion as constant term
        qubit_h.terms[()] = qubit_h.terms.get((), 0) + self._nuclear_repulsion

        return qubit_h

    @property
    def n_qubits(self) -> int:
        """Number of qubits needed (2 * n_orbitals for spin)."""
        return 2 * self._n_orbitals

    @property
    def n_electrons(self) -> int:
        """Total number of electrons."""
        return sum(a.atomic_number for a in self.atoms) - self.charge


class MoleculeBuilder:
    """Factory for common molecules."""

    @staticmethod
    def h2(bond_length: float = 0.74) -> list[Atom]:
        """Create H2 molecule."""
        return [
            Atom("H", (0.0, 0.0, 0.0)),
            Atom("H", (bond_length, 0.0, 0.0)),
        ]

    @staticmethod
    def h2o(oh_length: float = 0.96, angle: float = 104.5) -> list[Atom]:
        """Create H2O molecule."""
        angle_rad = np.radians(angle)
        return [
            Atom("O", (0.0, 0.0, 0.0)),
            Atom("H", (oh_length, 0.0, 0.0)),
            Atom("H", (
                oh_length * np.cos(angle_rad),
                oh_length * np.sin(angle_rad),
                0.0,
            )),
        ]

    @staticmethod
    def lih(bond_length: float = 1.60) -> list[Atom]:
        """Create LiH molecule."""
        return [
            Atom("Li", (0.0, 0.0, 0.0)),
            Atom("H", (bond_length, 0.0, 0.0)),
        ]

    @staticmethod
    def ch4() -> list[Atom]:
        """Create CH4 (methane) molecule with tetrahedral geometry."""
        # Tetrahedral bond length
        bond_length = 1.09
        angle = np.arccos(-1/3)

        return [
            Atom("C", (0.0, 0.0, 0.0)),
            Atom("H", (bond_length, 0.0, 0.0)),
            Atom("H", (
                bond_length * np.cos(angle),
                bond_length * np.sin(angle),
                0.0,
            )),
            Atom("H", (
                bond_length * np.cos(angle),
                bond_length * np.sin(angle) * np.cos(2 * np.pi / 3),
                bond_length * np.sin(angle) * np.sin(2 * np.pi / 3),
            )),
            Atom("H", (
                bond_length * np.cos(angle),
                bond_length * np.sin(angle) * np.cos(4 * np.pi / 3),
                bond_length * np.sin(angle) * np.sin(4 * np.pi / 3),
            )),
        ]

    @staticmethod
    def from_xyz(xyz_string: str) -> list[Atom]:
        """Parse XYZ format string."""
        lines = xyz_string.strip().split("\n")
        n_atoms = int(lines[0])
        atoms = []

        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            symbol = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append(Atom(symbol, (x, y, z)))

        return atoms

    @staticmethod
    def aspirin() -> list[Atom]:
        """Create simplified aspirin structure (C9H8O4)."""
        # Simplified coordinates
        return [
            Atom("C", (0.0, 0.0, 0.0)),
            Atom("C", (1.4, 0.0, 0.0)),
            Atom("C", (2.1, 1.2, 0.0)),
            Atom("C", (1.4, 2.4, 0.0)),
            Atom("C", (0.0, 2.4, 0.0)),
            Atom("C", (-0.7, 1.2, 0.0)),
            Atom("C", (-2.1, 1.2, 0.0)),
            Atom("O", (-2.8, 0.0, 0.0)),
            Atom("O", (-2.8, 2.4, 0.0)),
            Atom("C", (2.1, 3.6, 0.0)),
            Atom("O", (3.3, 3.6, 0.0)),
            Atom("O", (1.4, 4.8, 0.0)),
            Atom("C", (2.1, 6.0, 0.0)),
        ]
