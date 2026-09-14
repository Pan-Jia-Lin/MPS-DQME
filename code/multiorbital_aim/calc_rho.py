"""Reduced density operator, observables, and currents for split MPS-DQME.

The zero-dissipaton RDO is obtained by projecting every fermionic dissipaton
site onto local index zero while retaining all split system spin-orbitals.
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
    *map(int, pa.site_system_fermion_ket),
    *map(int, pa.site_system_fermion_bra),
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


# Pure-fermion multi-orbital implementation.

def _system_tensor_to_rdo(system_tensor,basis_order="orbital_spin"):
    n=pa.nsite_system
    expected=(2,)*(2*n)
    if system_tensor.shape!=expected:
        raise ValueError(f"system tensor {system_tensor.shape}, expected {expected}")
    # chain axes: ket logical order, then bra reverse logical order
    ordered=np.transpose(system_tensor,tuple(range(n))+tuple(range(2*n-1,n-1,-1)))
    return ordered.reshape(pa.ndvr_dense,pa.ndvr_dense,order="C")

def calc_rho(rin,basis_order="orbital_spin"):
    return _system_tensor_to_rdo(_project_system_tensor(rin),basis_order)

def calc_first_tier_rdo(rin,dissipaton_site,basis_order="orbital_spin"):
    site=int(dissipaton_site)
    if site in _SYSTEM_SITE_SET: raise ValueError("requested site is a system site")
    return _system_tensor_to_rdo(_project_system_tensor(rin,{site:1}),basis_order)

def _dense_system_fermion_operators(spin,orbital=0):
    q=orbital*pa.nspinf+spin; factors=[]
    for j in range(pa.nsite_system):
        factors.append(pa.sigma_z if j<q else (pa.sigma_plus if j==q else pa.Id))
    d=factors[0]
    for op in factors[1:]: d=np.kron(d,op)
    return d.astype(np.complex128),d.conj().T

def _system_fermion_parity():
    p=pa.sigma_z
    for _ in range(1,pa.nsite_system): p=np.kron(p,pa.sigma_z)
    return p.astype(np.complex128)

def _fermion_mode_pairs():
    ket={(x.spin,x.orbital,x.lead,x.pole):x for x in pa.fermion_modes if x.side=="ket"}
    bra={(x.spin,x.orbital,x.lead,x.pole):x for x in pa.fermion_modes if x.side=="bra"}
    if set(ket)!=set(bra): raise ValueError("ket/bra dissipaton labels differ")
    for label in sorted(ket): yield label,ket[label],bra[label]

def calc_spin_resolved_currents(rin):
    resolved={(a,s,m):0j for a in range(pa.nalphaf)
              for s in range(pa.nspinf) for m in range(pa.nvarf)}
    parity=_system_fermion_parity()
    for label,mode_m,mode_n in _fermion_mode_pairs():
        s,m,a,_=label
        rho_m=parity@calc_first_tier_rdo(rin,mode_m.site)
        rho_n=calc_first_tier_rdo(rin,mode_n.site)@parity
        d,dd=_dense_system_fermion_operators(s,m)
        resolved[(a,s,m)]+=1j*(pa.zeta_fermion[mode_n.chain_index]*np.trace(rho_n@d)
            -pa.zeta_fermion[mode_m.chain_index]*np.trace(dd@rho_m))
    return CurrentResult(resolved)

def calc_observables_from_rho(rho):
    values={"trace":np.trace(rho)}; ntotal=0j
    for m in range(pa.nvarf):
        for s in range(pa.nspinf):
            d,dd=_dense_system_fermion_operators(s,m)
            val=np.trace(dd@d@rho); values[f"n_m{m}_s{s}"]=val; ntotal+=val
    values["number_total"]=ntotal
    if pa.nvarf>=1:
        values["n_up"]=values["n_m0_s0"]; values["n_down"]=values["n_m0_s1"]
    return values

def calc_observables(rin): return calc_observables_from_rho(calc_rho(rin))

def _rank_one_product_mps(physical_indices):
    rho=pa.Density(); rho.nb=pa.nb.copy()
    for site,index in enumerate(physical_indices):
        node=np.zeros((1,int(pa.nb[site]),1),dtype=np.complex128)
        node[0,int(index),0]=1.0; rho.nodes.append(node)
    return rho

def validate_calc_rho():
    vacuum=np.zeros(pa.nsite_total,dtype=np.int64)
    x=_rank_one_product_mps(vacuum); rho=calc_rho(x)
    ref=np.zeros_like(rho); ref[0,0]=1
    if not np.array_equal(rho,ref): raise AssertionError("vacuum RDO mismatch")
    if not np.allclose(calc_spin_resolved_currents(x).by_lead_spin(),0):
        raise AssertionError("vacuum current is nonzero")

if __name__ == "__main__":
    validate_calc_rho(); print("multi-orbital calc_rho validation passed")
