"""Construct the pure-fermion multi-orbital AIM DQME Liouvillian MPO.

Every contribution is first represented as a bond-dimension-one product MPO.
The existing ``add_tensor.add_tensor`` routine is then used to add and compress
those products.  No TDVP routine is modified here.

Fermionic convention
--------------------
The ket-system and m-dissipaton sites use Jordan-Wigner strings starting at the
left end.  The n-dissipaton and bra-system sites use strings starting at the
right end.  Bosonic sites never carry parity.  Products are multiplied in their
mathematical order, so overlapping Z strings cancel automatically and endpoint
signs are retained.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import params as pa


Array = np.ndarray


@dataclass
class ProductTerm:
    name: str
    coefficient: complex
    local_ops: Dict[int, Array]


def _identity(d: int) -> Array:
    return np.eye(d, dtype=np.complex128)


def _number(d: int) -> Array:
    return np.diag(np.arange(d, dtype=np.complex128))


def _multiply_local(target: Dict[int, Array], site: int, op: Array) -> None:
    """Append an operator factor on one site in global product order."""
    op = np.asarray(op, dtype=np.complex128)
    d = int(pa.nb[site])
    if op.shape != (d, d):
        raise ValueError(
            f"site {site}: expected a {(d, d)} operator, got {op.shape}"
        )
    if site in target:
        target[site] = target[site] @ op
    else:
        target[site] = op.copy()


_LEFT_FERMION_SITES = tuple(
    list(map(int, pa.site_system_fermion_ket))
    + list(map(int, pa.site_fermion_diss_ket))
)
_RIGHT_FERMION_SITES = tuple(
    list(map(int, pa.site_fermion_diss_bra))
    + list(map(int, pa.site_system_fermion_bra))
)


def _single_jw_factor(site: int, op: Array, origin: str) -> Dict[int, Array]:
    """Return one fermion operator including its boundary-anchored JW string."""
    site = int(site)
    if origin == "left":
        fermion_sites = _LEFT_FERMION_SITES
        parity_sites = [j for j in fermion_sites if j < site]
    elif origin == "right":
        fermion_sites = _RIGHT_FERMION_SITES
        parity_sites = [j for j in fermion_sites if j > site]
    else:
        raise ValueError(f"unknown Jordan-Wigner origin {origin!r}")

    if site not in fermion_sites:
        raise ValueError(f"site {site} is not a {origin}-chain fermionic site")

    result: Dict[int, Array] = {}
    for parity_site in parity_sites:
        result[parity_site] = np.asarray(pa.sigma_z, dtype=np.complex128)
    result[site] = np.asarray(op, dtype=np.complex128)
    return result


def _fermion_product(
    factors: Sequence[Tuple[int, Array, str]],
) -> Dict[int, Array]:
    """Multiply boundary-JW fermion factors in the supplied algebraic order."""
    result: Dict[int, Array] = {}
    for site, op, origin in factors:
        factor = _single_jw_factor(site, op, origin)
        for factor_site, factor_op in factor.items():
            _multiply_local(result, factor_site, factor_op)

    # Remove explicit identities produced by Z @ Z collisions.
    cleaned = {}
    for site, op in result.items():
        if not np.allclose(op, _identity(int(pa.nb[site])), atol=pa.small):
            cleaned[site] = op
    return cleaned


def _term(name: str, coefficient: complex, ops=None) -> ProductTerm:
    return ProductTerm(name, complex(coefficient), {} if ops is None else ops)


def _plain_ops(*site_operator_pairs: Tuple[int, Array]) -> Dict[int, Array]:
    ops: Dict[int, Array] = {}
    for site, op in site_operator_pairs:
        _multiply_local(ops, int(site), op)
    return ops


def _system_pair(name, coefficient, factors):
    """Create -i H rho and +i rho H from one ordered fermion monomial.

    factors are ``(orbital, spin, local_operator)`` in mathematical order.
    Transposition on the bra reverses that order.
    """
    ket=[(pa.system_fermion_ket_site[(m,s)],op,"left") for m,s,op in factors]
    bra=[(pa.system_fermion_bra_site[(m,s)],op.T,"right")
          for m,s,op in reversed(factors)]
    return [
        _term("H_"+name+"_ket",-1j*coefficient,_fermion_product(ket)),
        _term("H_"+name+"_bra",+1j*coefficient,_fermion_product(bra)),
    ]


def _system_hamiltonian_terms() -> List[ProductTerm]:
    """Exact product-term decomposition of the agreed multi-orbital Hsys."""
    terms=[]; create=pa.sigma_minus; annih=pa.sigma_plus
    # sum_mm's epsilon_mm' d^dag_m d_m'
    for m in range(pa.nvarf):
        for mp in range(pa.nvarf):
            for s in range(pa.nspinf):
                c=pa.epsilon[m,mp]
                if abs(c)>pa.small:
                    terms += _system_pair(f"eps_m{m}_mp{mp}_s{s}",c,
                                          [(m,s,create),(mp,s,annih)])
    # U sum_m n_mup n_mdown
    for m in range(pa.nvarf):
        if abs(pa.U)>pa.small:
            terms += _system_pair(f"U_m{m}",pa.U,
                [(m,pa.SPIN_UP,create),(m,pa.SPIN_UP,annih),
                 (m,pa.SPIN_DOWN,create),(m,pa.SPIN_DOWN,annih)])
    for m in range(pa.nvarf):
        for mp in range(m+1,pa.nvarf):
            # opposite-spin U'
            for s in range(pa.nspinf):
                sb=1-s
                if abs(pa.Uprime)>pa.small:
                    terms += _system_pair(f"Up_m{m}_mp{mp}_s{s}",pa.Uprime,
                        [(m,s,create),(m,s,annih),(mp,sb,create),(mp,sb,annih)])
            # same-spin U'-J1
            same=pa.Uprime-pa.J1
            for s in range(pa.nspinf):
                if abs(same)>pa.small:
                    terms += _system_pair(f"Usame_m{m}_mp{mp}_s{s}",same,
                        [(m,s,create),(m,s,annih),(mp,s,create),(mp,s,annih)])
    # The user's formula explicitly sums over ordered pairs m != m'.
    for m in range(pa.nvarf):
        for mp in range(pa.nvarf):
            if m==mp: continue
            if abs(pa.J2)>pa.small:
                terms += _system_pair(f"J2_m{m}_mp{mp}",-pa.J2,
                    [(m,pa.SPIN_UP,create),(m,pa.SPIN_DOWN,annih),
                     (mp,pa.SPIN_DOWN,create),(mp,pa.SPIN_UP,annih)])
            if abs(pa.J3)>pa.small:
                terms += _system_pair(f"J3_m{m}_mp{mp}",pa.J3,
                    [(m,pa.SPIN_UP,create),(m,pa.SPIN_DOWN,create),
                     (mp,pa.SPIN_DOWN,annih),(mp,pa.SPIN_UP,annih)])
    return terms


def _fermion_mode_pairs():
    left = {
        (m.spin, m.orbital, m.lead, m.pole): m
        for m in pa.fermion_modes
        if m.side == "ket"
    }
    right = {
        (m.spin, m.orbital, m.lead, m.pole): m
        for m in pa.fermion_modes
        if m.side == "bra"
    }
    if set(left) != set(right):
        raise ValueError("ket/bra fermionic dissipaton labels are not mirror pairs")
    for label in sorted(left):
        yield label, left[label], right[label]


def _fermion_damping_terms() -> List[ProductTerm]:
    occupation = np.diag([0.0, 1.0]).astype(np.complex128)
    terms = []
    for mode in pa.fermion_modes:
        gamma = pa.gamma_fermion[mode.chain_index]
        terms.append(
            _term(
                f"gamma_f_q{mode.chain_index}",
                -gamma,
                _plain_ops((mode.site, occupation)),
            )
        )
    return terms


def _fermion_coupling_terms() -> List[ProductTerm]:
    """Return the four zeta and four xi products for every physical mode."""
    terms: List[ProductTerm] = []

    # In basis [|0>,|1>]: sigma_minus creates, sigma_plus annihilates.
    ddag_ket = pa.sigma_minus
    d_ket = pa.sigma_plus
    ddag_bra = pa.sigma_minus.T
    d_bra = pa.sigma_plus.T

    bminus_ket = pa.sigma_plus
    bplus_ket = pa.sigma_minus
    bminus_bra = pa.sigma_plus.T
    bplus_bra = pa.sigma_minus.T

    for label, mode_m, mode_n in _fermion_mode_pairs():
        spin, orbital, lead, pole = label
        system_ket = pa.system_fermion_ket_site[(orbital, spin)]
        system_bra = pa.system_fermion_bra_site[(orbital, spin)]
        qm = mode_m.chain_index
        qn = mode_n.chain_index
        tag = f"s{spin}_v{orbital}_a{lead}_k{pole}"

        zeta_minus = pa.zeta_fermion[qm]
        zeta_plus = pa.zeta_fermion[qn]
        xi_minus = pa.ksi_fermion[qm]
        xi_minus_star = pa.ksi_star_fermion[qm]
        xi_plus = pa.ksi_fermion[qn]
        xi_plus_star = pa.ksi_star_fermion[qn]

        # -i zeta^- d^dagger b^- rho
        terms.append(
            _term(
                f"zeta_minus_dDbM_rho_{tag}",
                -1.0j * zeta_minus,
                _fermion_product(
                    [
                        (system_ket, ddag_ket, "left"),
                        (mode_m.site, bminus_ket, "left"),
                    ]
                ),
            )
        )

        # +i zeta^- b^- rho d^dagger
        terms.append(
            _term(
                f"zeta_minus_bM_rho_dD_{tag}",
                +1.0j * zeta_minus,
                _fermion_product(
                    [
                        (mode_m.site, bminus_ket, "left"),
                        (system_bra, ddag_bra, "right"),
                    ]
                ),
            )
        )

        # -i zeta^+ d rho b^+
        terms.append(
            _term(
                f"zeta_plus_d_rho_bP_{tag}",
                -1.0j * zeta_plus,
                _fermion_product(
                    [
                        (system_ket, d_ket, "left"),
                        (mode_n.site, bplus_bra, "right"),
                    ]
                ),
            )
        )

        # +i zeta^+ rho b^+ d; transpose reverses the two right factors.
        terms.append(
            _term(
                f"zeta_plus_rho_bP_d_{tag}",
                +1.0j * zeta_plus,
                _fermion_product(
                    [
                        (system_bra, d_bra, "right"),
                        (mode_n.site, bplus_bra, "right"),
                    ]
                ),
            )
        )

        # -i xi^+ d^dagger rho b^-
        terms.append(
            _term(
                f"xi_plus_dD_rho_bM_{tag}",
                -1.0j * xi_plus,
                _fermion_product(
                    [
                        (system_ket, ddag_ket, "left"),
                        (mode_n.site, bminus_bra, "right"),
                    ]
                ),
            )
        )

        # +i xi^{+*} b^+ rho d
        terms.append(
            _term(
                f"xi_plus_star_bP_rho_d_{tag}",
                +1.0j * xi_plus_star,
                _fermion_product(
                    [
                        (mode_m.site, bplus_ket, "left"),
                        (system_bra, d_bra, "right"),
                    ]
                ),
            )
        )

        # +i xi^- d b^+ rho
        terms.append(
            _term(
                f"xi_minus_d_bP_rho_{tag}",
                +1.0j * xi_minus,
                _fermion_product(
                    [
                        (system_ket, d_ket, "left"),
                        (mode_m.site, bplus_ket, "left"),
                    ]
                ),
            )
        )

        # -i xi^{-*} rho b^- d^dagger; transpose reverses right factors.
        terms.append(
            _term(
                f"xi_minus_star_rho_bM_dD_{tag}",
                -1.0j * xi_minus_star,
                _fermion_product(
                    [
                        (system_bra, ddag_bra, "right"),
                        (mode_n.site, bminus_bra, "right"),
                    ]
                ),
            )
        )
    return terms


def build_product_terms() -> List[ProductTerm]:
    terms = []
    terms.extend(_system_hamiltonian_terms())
    terms.extend(_fermion_damping_terms())
    terms.extend(_fermion_coupling_terms())
    return terms


def _product_term_to_mpo(term: ProductTerm):
    mpo = pa.Density()
    mpo.nb = pa.nbmat.copy()
    for site, d in enumerate(map(int, pa.nb)):
        op = term.local_ops.get(site, _identity(d))
        if site == 0:
            op = term.coefficient * op
        flat = np.reshape(op, d * d, order="F")
        mpo.nodes.append(flat.reshape(1, d * d, 1))
    return mpo


def construct(Iscorr=False, Iscommu=None):
    """Build the Liouvillian MPO using the project's existing TT summation."""
    if Iscorr:
        raise NotImplementedError(
            "correlation-function MPOs will be rebuilt after construct.py and "
            "calc_rho.py pass the DQME regression tests"
        )

    terms = build_product_terms()
    if not terms:
        raise RuntimeError("no DQME terms were generated")

    # Lazy import permits parameter/term self-tests without tensornetwork.
    import add_tensor as at

    result = _product_term_to_mpo(terms[0])
    for index, term in enumerate(terms[1:], start=1):
        result = at.add_tensor(result, _product_term_to_mpo(term), 1.0)
        if index % 25 == 0:
            print(f"construct: added {index + 1}/{len(terms)} product terms")
    print(f"construct: completed {len(terms)} product terms")
    return result


def validate_term_structure() -> None:
    """Pure-fermion structural checks (overrides the archived mixed check)."""
    terms=build_product_terms(); names=[x.name for x in terms]
    if not terms or len(names)!=len(set(names)):
        raise AssertionError("empty product list or duplicate term names")
    for term in terms:
        if not np.isfinite(term.coefficient):
            raise AssertionError(f"non-finite coefficient in {term.name}")
        for site,op in term.local_ops.items():
            if op.shape!=(int(pa.nb[site]),int(pa.nb[site])):
                raise AssertionError(f"bad local operator in {term.name} at {site}")
    # Every physical mode has eight DQME coupling terms plus one damping term.
    nf=sum(x.name.startswith(("zeta_","xi_")) for x in terms)
    if nf!=8*pa.nlevelhalf:
        raise AssertionError(f"expected {8*pa.nlevelhalf} coupling terms, got {nf}")
    if sum(x.name.startswith("gamma_f_") for x in terms)!=pa.nlevelf:
        raise AssertionError("fermionic damping term count mismatch")


if __name__ == "__main__":
    validate_term_structure()
    generated = build_product_terms()
    print("construct.py structural validation passed")
    print("number of product terms =", len(generated))
    print("first term =", generated[0].name)
    print("last term =", generated[-1].name)
