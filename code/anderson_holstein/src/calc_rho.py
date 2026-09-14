"""Reduced density operator, observables, and currents for split MPS-DQME.

The zero-dissipaton RDO is obtained by projecting every fermionic and bosonic
dissipaton site onto local index zero while retaining the six system indices.
Spin-resolved lead currents are grouped exclusively through the mode metadata in
``params.fermion_modes``; no hard-coded pole ranges are used.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

import params as pa


CurrentKey = Tuple[int, int, int]  # (lead, spin, orbital)


@dataclass
class CurrentResult:
    """Spin-, lead-, and orbital-resolved particle currents."""

    resolved: Dict[CurrentKey, complex]

    def by_lead_spin(self) -> np.ndarray:
        values = np.zeros((pa.nalphaf, pa.nspinf), dtype=np.complex128)
        for (lead, spin, _orbital), value in self.resolved.items():
            values[lead, spin] += value
        return values

    def by_lead(self) -> np.ndarray:
        return np.sum(self.by_lead_spin(), axis=1)

    def by_spin(self) -> np.ndarray:
        return np.sum(self.by_lead_spin(), axis=0)

    def total(self) -> complex:
        return complex(np.sum(self.by_lead_spin()))


_SYSTEM_SITES_IN_CHAIN_ORDER = (
    int(pa.site_system_boson_ket),
    *map(int, pa.site_system_fermion_ket),
    *map(int, pa.site_system_fermion_bra),
    int(pa.site_system_boson_bra),
)
_SYSTEM_SITE_SET = set(_SYSTEM_SITES_IN_CHAIN_ORDER)


def _validate_mps(rin) -> None:
    if len(rin.nodes) != pa.nsite_total:
        raise ValueError(
            f"MPS has {len(rin.nodes)} sites, expected {pa.nsite_total}"
        )
    for site, node in enumerate(rin.nodes):
        if node.ndim != 3:
            raise ValueError(f"MPS node {site} is not rank three: {node.shape}")
        if node.shape[1] != pa.nb[site]:
            raise ValueError(
                f"site {site}: physical dimension {node.shape[1]}, "
                f"expected {pa.nb[site]}"
            )
        if site == 0 and node.shape[0] != 1:
            raise ValueError("left MPS boundary dimension is not one")
        if site == pa.nsite_total - 1 and node.shape[2] != 1:
            raise ValueError("right MPS boundary dimension is not one")
        if site + 1 < pa.nsite_total:
            if node.shape[2] != rin.nodes[site + 1].shape[0]:
                raise ValueError(
                    f"bond mismatch between sites {site} and {site + 1}: "
                    f"{node.shape[2]} != {rin.nodes[site + 1].shape[0]}"
                )


def _project_system_tensor(
    rin,
    dissipaton_occupations: Optional[Mapping[int, int]] = None,
) -> np.ndarray:
    """Project dissipatons and retain all system physical indices.

    ``dissipaton_occupations`` maps absolute MPS site numbers to selected local
    indices.  Unspecified dissipaton sites are projected onto zero.
    """
    _validate_mps(rin)
    selected = {} if dissipaton_occupations is None else dict(dissipaton_occupations)
    illegal = set(selected) & _SYSTEM_SITE_SET
    if illegal:
        raise ValueError(f"system sites cannot be projected as dissipatons: {illegal}")

    tensor = np.array([1.0 + 0.0j])
    for site, node in enumerate(rin.nodes):
        if site in _SYSTEM_SITE_SET:
            # (..., left bond) x (left bond, physical, right bond)
            tensor = np.tensordot(tensor, node, axes=([-1], [0]))
        else:
            occupation = int(selected.get(site, 0))
            if occupation < 0 or occupation >= node.shape[1]:
                raise ValueError(
                    f"site {site}: occupation {occupation} outside "
                    f"0..{node.shape[1] - 1}"
                )
            tensor = np.tensordot(
                tensor, node[:, occupation, :], axes=([-1], [0])
            )

    if tensor.shape[-1] != 1:
        raise AssertionError(f"final MPS boundary has dimension {tensor.shape[-1]}")
    return np.squeeze(tensor, axis=-1)


def _system_tensor_to_rdo(system_tensor: np.ndarray, basis_order: str) -> np.ndarray:
    """Convert chain axes to a conventional dense system RDO.

    Chain axes are
      (phonon ket, up ket, down ket, down bra, up bra, phonon bra).

    The default program-b-compatible dense basis is
      (up, down, phonon)
    for both the row and column indices.
    """
    expected = (
        pa.nboson,
        2,
        2,
        2,
        2,
        pa.nboson,
    )
    if system_tensor.shape != expected:
        raise ValueError(
            f"retained system tensor has shape {system_tensor.shape}, "
            f"expected {expected}"
        )

    if basis_order == "fermion_then_boson":
        # (upK, downK, phK, upB, downB, phB)
        ordered = np.transpose(system_tensor, (1, 2, 0, 4, 3, 5))
    elif basis_order == "chain":
        # Logical order (ph, up, down) on both matrix axes.
        ordered = np.transpose(system_tensor, (0, 1, 2, 5, 4, 3))
    else:
        raise ValueError(
            "basis_order must be 'fermion_then_boson' or 'chain', "
            f"got {basis_order!r}"
        )

    return np.reshape(
        ordered,
        (pa.ndvr_dense, pa.ndvr_dense),
        order="C",
    )


def calc_rho(rin, basis_order: str = "fermion_then_boson") -> np.ndarray:
    """Return the physical zero-dissipaton reduced density operator."""
    tensor = _project_system_tensor(rin)
    return _system_tensor_to_rdo(tensor, basis_order)


def calc_first_tier_rdo(
    rin,
    dissipaton_site: int,
    basis_order: str = "fermion_then_boson",
) -> np.ndarray:
    """Return the RDO coefficient with one selected dissipaton occupied."""
    dissipaton_site = int(dissipaton_site)
    if dissipaton_site in _SYSTEM_SITE_SET:
        raise ValueError(f"site {dissipaton_site} is a system site")
    tensor = _project_system_tensor(rin, {dissipaton_site: 1})
    return _system_tensor_to_rdo(tensor, basis_order)


def _dense_system_fermion_operators(spin: int):
    """Bare d and d^dagger in the (up, down, phonon) dense basis."""
    if spin == pa.SPIN_UP:
        d_fermion = np.kron(pa.sigma_plus, pa.Id)
    elif spin == pa.SPIN_DOWN:
        d_fermion = np.kron(pa.sigma_z, pa.sigma_plus)
    else:
        raise ValueError(f"invalid spin index {spin}")
    d = np.kron(d_fermion, pa.I_boson).astype(np.complex128)
    return d, d.conj().T


def _system_fermion_parity() -> np.ndarray:
    """Return P=(-1)^(n_up+n_down), with identity on the system phonon."""
    parity_f = np.kron(pa.sigma_z, pa.sigma_z)
    return np.kron(parity_f, pa.I_boson).astype(np.complex128)


def _fermion_mode_pairs():
    ket_modes = {
        (m.spin, m.orbital, m.lead, m.pole): m
        for m in pa.fermion_modes
        if m.side == "ket"
    }
    bra_modes = {
        (m.spin, m.orbital, m.lead, m.pole): m
        for m in pa.fermion_modes
        if m.side == "bra"
    }
    if set(ket_modes) != set(bra_modes):
        raise ValueError("fermionic ket/bra dissipaton labels do not match")
    for label in sorted(ket_modes):
        yield label, ket_modes[label], bra_modes[label]


def calc_spin_resolved_currents(rin) -> CurrentResult:
    r"""Calculate currents grouped by ``(lead, spin, orbital)`` metadata.

    The split-chain first tiers carry the boundary-Jordan-Wigner convention.
    Their physical dense-RDO representatives are

      rho_m(physical) = P rho_m(raw),
      rho_n(physical) = rho_n(raw) P,

    with P=(-1)^(n_up+n_down).  This parity dressing was fixed by comparison
    with the lead-resolved Liouvillian current and is required once the system
    acquires electronic population.

    The DQME first-tier identification is

      F^- rho_tot -> zeta^- rho_m,
      F^+ rho_tot -> zeta^+ rho_n.

    With I = -i(d^dagger F^- - F^+ d), each pole contributes

      i [zeta^+ Tr(rho_n d) - zeta^- Tr(d^dagger rho_m)].

    This expression follows the displayed current operator and the zeta terms
    that couple the physical zero-dissipaton RDO to the m/n first tiers.
    """
    resolved: Dict[CurrentKey, complex] = {
        (lead, spin, orbital): 0.0 + 0.0j
        for lead in range(pa.nalphaf)
        for spin in range(pa.nspinf)
        for orbital in range(pa.nvarf)
    }

    parity = _system_fermion_parity()

    for label, mode_m, mode_n in _fermion_mode_pairs():
        spin, orbital, lead, _pole = label
        rho_m = parity @ calc_first_tier_rdo(rin, mode_m.site)
        rho_n = calc_first_tier_rdo(rin, mode_n.site) @ parity
        d, ddagger = _dense_system_fermion_operators(spin)

        zeta_minus = pa.zeta_fermion[mode_m.chain_index]
        zeta_plus = pa.zeta_fermion[mode_n.chain_index]
        contribution = 1.0j * (
            zeta_plus * np.trace(rho_n @ d)
            - zeta_minus * np.trace(ddagger @ rho_m)
        )
        resolved[(lead, spin, orbital)] += contribution

    return CurrentResult(resolved=resolved)


def calc_observables_from_rho(rho: np.ndarray) -> Dict[str, complex]:
    """Return system observables from an already reconstructed dense RDO."""
    rho = np.asarray(rho, dtype=np.complex128)
    expected_shape = (pa.ndvr_dense, pa.ndvr_dense)
    if rho.shape != expected_shape:
        raise ValueError(f"rho has shape {rho.shape}, expected {expected_shape}")
    d_up, ddag_up = _dense_system_fermion_operators(pa.SPIN_UP)
    d_down, ddag_down = _dense_system_fermion_operators(pa.SPIN_DOWN)

    identity_f = np.eye(pa.nfermion, dtype=np.complex128)
    a = np.kron(identity_f, pa.boson_a)
    ad = np.kron(identity_f, pa.boson_ad)
    x = (ad + a) / np.sqrt(2.0)
    p = 1.0j * (ad - a) / np.sqrt(2.0)
    phonon_number = ad @ a
    symmetrized_xp = 0.5 * (x @ p + p @ x)

    operators = {
        "n_up": ddag_up @ d_up,
        "n_down": ddag_down @ d_down,
        "x": x,
        "p": p,
        "x2": x @ x,
        "p2": p @ p,
        "phonon_number": phonon_number,
        "symmetrized_xp": symmetrized_xp,
    }
    values = {name: np.trace(op @ rho) for name, op in operators.items()}
    values["trace"] = np.trace(rho)
    return values


def calc_observables(rin) -> Dict[str, complex]:
    """Reconstruct the RDO and return explicitly named system observables."""
    return calc_observables_from_rho(calc_rho(rin))


def calc_current(rin):
    """Backward-compatible tuple used by the current program-b prop.py.

    Returns observables followed by (L-up, L-down, R-up, R-down).  The seventh
    value remains phonon number for numerical compatibility with the old code,
    whose ``xp_px_ave`` label was incorrect.
    """
    obs = calc_observables(rin)
    currents = calc_spin_resolved_currents(rin).by_lead_spin()
    if pa.nalphaf != 2 or pa.nspinf != 2:
        raise ValueError(
            "legacy calc_current tuple requires exactly two leads and two spins; "
            "use calc_spin_resolved_currents for the general result"
        )
    return (
        obs["n_up"],
        obs["n_down"],
        obs["x"],
        obs["p"],
        obs["x2"],
        obs["p2"],
        obs["phonon_number"],
        obs["trace"],
        currents[pa.LEAD_LEFT, pa.SPIN_UP],
        currents[pa.LEAD_LEFT, pa.SPIN_DOWN],
        currents[pa.LEAD_RIGHT, pa.SPIN_UP],
        currents[pa.LEAD_RIGHT, pa.SPIN_DOWN],
    )


def _rank_one_product_mps(physical_indices: Sequence[int]):
    """Small exact product MPS used only by this module's self-tests."""
    if len(physical_indices) != pa.nsite_total:
        raise ValueError("wrong number of physical indices")
    rho = pa.Density()
    rho.nb = pa.nb.copy()
    for site, physical_index in enumerate(physical_indices):
        node = np.zeros((1, int(pa.nb[site]), 1), dtype=np.complex128)
        node[0, int(physical_index), 0] = 1.0
        rho.nodes.append(node)
    return rho


def validate_calc_rho() -> None:
    vacuum = np.zeros(pa.nsite_total, dtype=np.int64)
    rho_mps = _rank_one_product_mps(vacuum)
    rho = calc_rho(rho_mps)
    reference = np.zeros_like(rho)
    reference[0, 0] = 1.0
    if not np.array_equal(rho, reference):
        raise AssertionError("vacuum RDO is not diag(1,0,...,0)")

    obs = calc_observables(rho_mps)
    if not np.isclose(obs["trace"], 1.0):
        raise AssertionError(f"initial trace is {obs['trace']}, expected 1")
    for name in ("n_up", "n_down", "phonon_number", "x", "p"):
        if not np.isclose(obs[name], 0.0):
            raise AssertionError(f"initial {name} is {obs[name]}, expected 0")

    currents = calc_spin_resolved_currents(rho_mps).by_lead_spin()
    if not np.allclose(currents, 0.0):
        raise AssertionError(f"initial currents are nonzero: {currents}")


if __name__ == "__main__":
    validate_calc_rho()
    print("calc_rho.py validation passed")
    print("vacuum RDO: rho[0,0] = 1; all other entries = 0")
    print("vacuum lead-spin currents = 0")
