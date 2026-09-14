"""Initial MPS for the split-system Anderson-Holstein DQME.

The physical initial state is exactly

    system phonon ground state x empty spin-up x empty spin-down
    x dissipaton vacuum
    x empty bra spin-down x empty bra spin-up x bra phonon ground state.

Consequently, the reduced density operator has rho[0, 0] = 1 in the ordered
basis and all other elements zero.

Although the exact Schmidt rank of this product state is one, the tensor shapes
are allocated with larger virtual-bond capacities.  This is intentional: the
one-site TDVP implementation in this project should not be initialized with
all virtual dimensions equal to one, or its variational manifold cannot grow
to the requested working rank.
"""

from typing import Iterable, List, Sequence

import numpy as np

import params as pa


def _capped_product(values: Iterable[int], cap: int) -> int:
    """Multiply positive integers, stopping once ``cap`` is reached."""
    result = 1
    for value in values:
        result *= int(value)
        if result >= cap:
            return int(cap)
    return int(result)


def allocated_bond_dimensions(
    physical_dimensions: Sequence[int] = None,
    max_rank: int = None,
) -> np.ndarray:
    """Return admissible MPS bond capacities for every cut.

    The returned array has length ``nsite + 1`` and boundary values one.  At a
    cut after site i, the capacity is bounded by the Hilbert dimensions on both
    sides as well as ``max_rank``.
    """
    dims = np.asarray(
        pa.nb if physical_dimensions is None else physical_dimensions,
        dtype=np.int64,
    )
    rank_cap = int(pa.nrtt if max_rank is None else max_rank)

    if dims.ndim != 1 or dims.size == 0:
        raise ValueError("physical_dimensions must be a non-empty 1-D sequence")
    if np.any(dims <= 0):
        raise ValueError("all physical dimensions must be positive")
    if rank_cap <= 0:
        raise ValueError("max_rank must be positive")

    nsite = dims.size
    bonds = np.ones(nsite + 1, dtype=np.int64)
    for cut in range(1, nsite):
        left_capacity = _capped_product(dims[:cut], rank_cap)
        right_capacity = _capped_product(dims[cut:], rank_cap)
        bonds[cut] = min(rank_cap, left_capacity, right_capacity)
    return bonds


def initial_physical_indices() -> np.ndarray:
    """Return the occupied local basis index at every MPS site.

    Index zero means the system-phonon ground state, an empty system fermion,
    or the vacuum of a dissipaton mode.  Keeping this function explicit makes
    later tests with occupied electronic or vibrational initial states easy.
    """
    return np.zeros(pa.nsite_total, dtype=np.int64)


def init_rho(max_rank: int = None):
    """Build the exact vacuum/ground-state RDT with padded TDVP bond capacity."""
    dims = np.asarray(pa.nb, dtype=np.int64)
    if dims.size != pa.nsite_total:
        raise ValueError(
            f"params.nb has {dims.size} sites, expected {pa.nsite_total}"
        )

    physical_indices = initial_physical_indices()
    if np.any(physical_indices < 0) or np.any(physical_indices >= dims):
        raise ValueError("an initial physical index lies outside its local basis")

    bonds = allocated_bond_dimensions(dims, max_rank=max_rank)
    rhoall = pa.Density()
    rhoall.nb = dims.copy()

    # Only virtual channel zero carries the initial product-state amplitude.
    # The remaining allocated channels are exactly zero and are available to
    # the projector-splitting TDVP gauge construction.
    for site, (left_rank, physical_dim, right_rank, physical_index) in enumerate(
        zip(bonds[:-1], dims, bonds[1:], physical_indices)
    ):
        node = np.zeros(
            (int(left_rank), int(physical_dim), int(right_rank)),
            dtype=np.complex128,
        )
        node[0, int(physical_index), 0] = 1.0
        rhoall.nodes.append(node)

    validate_initial_rho(rhoall, expected_bonds=bonds)
    print("initial RDT: phonon ground state x fermion vacuum x dissipaton vacuum")
    print("number of MPS sites =", len(rhoall.nodes))
    print("allocated bond dimensions =", bonds)
    print("exact initial Schmidt rank = 1")
    return rhoall


def _product_amplitude(rhoall, physical_indices: Sequence[int]) -> complex:
    """Contract one physical configuration without external tensor packages."""
    amplitude = np.array([1.0 + 0.0j])
    for node, physical_index in zip(rhoall.nodes, physical_indices):
        amplitude = amplitude @ node[:, int(physical_index), :]
    if amplitude.shape != (1,):
        raise AssertionError(f"final boundary has unexpected shape {amplitude.shape}")
    return complex(amplitude[0])


def validate_initial_rho(rhoall, expected_bonds=None) -> None:
    """Validate shapes, vacuum amplitude, and absence of spurious components."""
    if len(rhoall.nodes) != pa.nsite_total:
        raise AssertionError(
            f"initial MPS has {len(rhoall.nodes)} nodes, expected {pa.nsite_total}"
        )
    if not np.array_equal(rhoall.nb, pa.nb):
        raise AssertionError("rhoall.nb is inconsistent with params.nb")

    if expected_bonds is None:
        expected_bonds = allocated_bond_dimensions(pa.nb)
    expected_bonds = np.asarray(expected_bonds, dtype=np.int64)

    nonzero_count = 0
    squared_norm_of_entries = 0.0
    for site, node in enumerate(rhoall.nodes):
        expected_shape = (
            int(expected_bonds[site]),
            int(pa.nb[site]),
            int(expected_bonds[site + 1]),
        )
        if node.shape != expected_shape:
            raise AssertionError(
                f"site {site}: got shape {node.shape}, expected {expected_shape}"
            )
        nonzero_count += int(np.count_nonzero(node))
        squared_norm_of_entries += float(np.vdot(node, node).real)

    # There is exactly one nonzero entry per site.  This is stronger than just
    # checking the final contracted amplitude and catches accidental branches.
    if nonzero_count != pa.nsite_total:
        raise AssertionError(
            f"expected exactly {pa.nsite_total} nonzero tensor entries, "
            f"found {nonzero_count}"
        )
    if not np.isclose(squared_norm_of_entries, float(pa.nsite_total)):
        raise AssertionError("initial tensor entries do not all have unit amplitude")

    vacuum = initial_physical_indices()
    vacuum_amplitude = _product_amplitude(rhoall, vacuum)
    if not np.isclose(vacuum_amplitude, 1.0 + 0.0j):
        raise AssertionError(
            f"vacuum product amplitude is {vacuum_amplitude}, expected 1"
        )

    # Flip every site once and verify that no one-site excitation was inserted.
    for site, dim in enumerate(pa.nb):
        if dim <= 1:
            continue
        excited = vacuum.copy()
        excited[site] = 1
        amplitude = _product_amplitude(rhoall, excited)
        if not np.isclose(amplitude, 0.0 + 0.0j):
            raise AssertionError(
                f"spurious initial amplitude {amplitude} at excited site {site}"
            )


def dense_initial_system_rdo() -> np.ndarray:
    """Small dense reference RDO in the program-b ordered system basis.

    The basis index zero is |n_up=0, n_down=0, n_ph=0>.  This function is only
    a benchmark reference; the production initial state remains factorized.
    """
    rdo = np.zeros((pa.ndvr_dense, pa.ndvr_dense), dtype=np.complex128)
    rdo[0, 0] = 1.0
    return rdo


if __name__ == "__main__":
    rho = init_rho()
    reference = dense_initial_system_rdo()
    assert reference[0, 0] == 1.0
    assert np.count_nonzero(reference) == 1
    print("dense RDO check: rho[0, 0] = 1 and all other elements are zero")
