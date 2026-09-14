"""Construct the split-system Anderson-Holstein DQME Liouvillian MPO.

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


def _boson_annihilation(d: int) -> Array:
    op = np.zeros((d, d), dtype=np.complex128)
    for n in range(1, d):
        op[n - 1, n] = np.sqrt(n)
    return op


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


def _system_hamiltonian_terms() -> List[ProductTerm]:
    """Return -i(H rho-rho H) for the explicitly split AH system."""
    if pa.nvarf != 1:
        raise NotImplementedError(
            "the present chain has one system phonon endpoint and currently "
            "implements the single-orbital Anderson-Holstein model"
        )
    if getattr(pa, "use_lang_firsov", False):
        raise NotImplementedError(
            "construct.py currently implements the displayed untransformed "
            "Anderson-Holstein Hamiltonian; set use_lang_firsov=False"
        )

    terms: List[ProductTerm] = []
    n_f = np.asarray(pa.number_f, dtype=np.complex128)
    q_b = np.asarray(pa.Q_system_boson, dtype=np.complex128)
    n_b = np.asarray(pa.N_system_boson, dtype=np.complex128)

    # Single-particle electronic energies.
    for spin in range(pa.nspinf):
        eps = pa.epsilon[0, spin]
        ket_site = pa.system_fermion_ket_site[(0, spin)]
        bra_site = pa.system_fermion_bra_site[(0, spin)]
        terms.append(
            _term(f"H_eps_ket_s{spin}", -1.0j * eps, _plain_ops((ket_site, n_f)))
        )
        terms.append(
            _term(
                f"H_eps_bra_s{spin}",
                +1.0j * eps,
                _plain_ops((bra_site, n_f.T)),
            )
        )

    # Hubbard U n_up n_down.
    kup = pa.system_fermion_ket_site[(0, pa.SPIN_UP)]
    kdn = pa.system_fermion_ket_site[(0, pa.SPIN_DOWN)]
    bup = pa.system_fermion_bra_site[(0, pa.SPIN_UP)]
    bdn = pa.system_fermion_bra_site[(0, pa.SPIN_DOWN)]
    terms.append(
        _term("H_U_ket", -1.0j * pa.Uu, _plain_ops((kup, n_f), (kdn, n_f)))
    )
    terms.append(
        _term(
            "H_U_bra",
            +1.0j * pa.Uu,
            _plain_ops((bup, n_f.T), (bdn, n_f.T)),
        )
    )

    # Local system phonon Omega a^dagger a.
    terms.append(
        _term(
            "H_phonon_ket",
            -1.0j * pa.omega,
            _plain_ops((pa.site_system_boson_ket, n_b)),
        )
    )
    terms.append(
        _term(
            "H_phonon_bra",
            +1.0j * pa.omega,
            _plain_ops((pa.site_system_boson_bra, n_b.T)),
        )
    )

    # g Omega (a^dagger+a) n_s.
    for spin in range(pa.nspinf):
        ket_site = pa.system_fermion_ket_site[(0, spin)]
        bra_site = pa.system_fermion_bra_site[(0, spin)]
        coupling = pa.g_ep * pa.omega
        terms.append(
            _term(
                f"H_ep_ket_s{spin}",
                -1.0j * coupling,
                _plain_ops(
                    (pa.site_system_boson_ket, q_b),
                    (ket_site, n_f),
                ),
            )
        )
        terms.append(
            _term(
                f"H_ep_bra_s{spin}",
                +1.0j * coupling,
                _plain_ops(
                    (bra_site, n_f.T),
                    (pa.site_system_boson_bra, q_b.T),
                ),
            )
        )
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


def _boson_terms() -> List[ProductTerm]:
    """Return scaled hierarchy terms for A=Q+2*lambda*(n_up+n_down).

    The split-system MPS stores Q, n_up and n_down on different sites. Each
    occurrence of A is therefore expanded into three bond-dimension-one MPO
    products. Number operators are fermion-even and require no JW string.  The
    lowering ladder is a pure commutator; all eta-dependent asymmetric terms
    belong to the raising ladder.
    """
    terms: List[ProductTerm] = []
    q_system = np.asarray(pa.Q_system_boson, dtype=np.complex128)
    number_f = np.asarray(pa.number_f, dtype=np.complex128)

    ket_components = [
        ("Q", int(pa.site_system_boson_ket), q_system, 1.0),
        ("Nup", int(pa.system_fermion_ket_site[(0, pa.SPIN_UP)]),
         number_f, pa.boson_bath_charge_shift),
        ("Ndown", int(pa.system_fermion_ket_site[(0, pa.SPIN_DOWN)]),
         number_f, pa.boson_bath_charge_shift),
    ]
    bra_components = [
        ("Q", int(pa.site_system_boson_bra), q_system.T, 1.0),
        ("Nup", int(pa.system_fermion_bra_site[(0, pa.SPIN_UP)]),
         number_f.T, pa.boson_bath_charge_shift),
        ("Ndown", int(pa.system_fermion_bra_site[(0, pa.SPIN_DOWN)]),
         number_f.T, pa.boson_bath_charge_shift),
    ]

    for k, site in enumerate(map(int, pa.boson_diss_to_site)):
        d = int(pa.nb[site])
        bminus = _boson_annihilation(d)
        bplus = bminus.conj().T
        number = _number(d)

        gamma = pa.gamma_boson[k]
        norm = pa.normfac_boson[k]
        eta_re_scaled = pa.eta_re_scaled_boson[k]
        eta_im_scaled = pa.eta_im_scaled_boson[k]

        terms.append(
            _term(f"gamma_b_k{k}", -gamma, _plain_ops((site, number)))
        )

        # Raising direction: eta-dependent asymmetric coefficients.
        for label, system_site, operator, scale in ket_components:
            terms.append(
                _term(
                    f"eta_bplus_{label}rho_k{k}",
                    (eta_im_scaled - 1.0j * eta_re_scaled) * scale,
                    _plain_ops((system_site, operator), (site, bplus)),
                )
            )
        for label, system_site, operator, scale in bra_components:
            terms.append(
                _term(
                    f"eta_bplus_rho{label}_k{k}",
                    (eta_im_scaled + 1.0j * eta_re_scaled) * scale,
                    _plain_ops((site, bplus), (system_site, operator)),
                )
            )

        # Lowering direction: pure commutator, hence zero physical trace.
        for label, system_site, operator, scale in ket_components:
            terms.append(
                _term(
                    f"norm_bminus_{label}rho_k{k}",
                    -1.0j * norm * scale,
                    _plain_ops((system_site, operator), (site, bminus)),
                )
            )
        for label, system_site, operator, scale in bra_components:
            terms.append(
                _term(
                    f"norm_bminus_rho{label}_k{k}",
                    +1.0j * norm * scale,
                    _plain_ops((site, bminus), (system_site, operator)),
                )
            )
    return terms


def build_product_terms() -> List[ProductTerm]:
    terms = []
    terms.extend(_system_hamiltonian_terms())
    terms.extend(_fermion_damping_terms())
    terms.extend(_boson_terms())
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
    """Checks that do not require tensornetwork or a full dense Liouvillian."""
    terms = build_product_terms()
    expected = (
        (2 * pa.nspinf + 2 + 2 + 2 * pa.nspinf)
        + pa.nlevelf
        + 13 * pa.nlevelb
        + 8 * pa.nlevelhalf
    )
    if len(terms) != expected:
        raise AssertionError(f"expected {expected} product terms, got {len(terms)}")

    names = [term.name for term in terms]
    if len(names) != len(set(names)):
        raise AssertionError("product-term names are not unique")

    by_name = {term.name: term for term in terms}
    component_scales = {
        "Q": 1.0,
        "Nup": pa.boson_bath_charge_shift,
        "Ndown": pa.boson_bath_charge_shift,
    }
    for k, site in enumerate(map(int, pa.boson_diss_to_site)):
        bminus = _boson_annihilation(int(pa.nb[site]))
        bplus = bminus.conj().T
        for label, scale in component_scales.items():
            left_up = by_name[f"eta_bplus_{label}rho_k{k}"]
            right_up = by_name[f"eta_bplus_rho{label}_k{k}"]
            left_down = by_name[f"norm_bminus_{label}rho_k{k}"]
            right_down = by_name[f"norm_bminus_rho{label}_k{k}"]

            expected_left_up = scale * (
                pa.eta_im_scaled_boson[k]
                - 1.0j * pa.eta_re_scaled_boson[k]
            )
            expected_right_up = scale * (
                pa.eta_im_scaled_boson[k]
                + 1.0j * pa.eta_re_scaled_boson[k]
            )
            if not np.allclose(left_up.coefficient, expected_left_up):
                raise AssertionError(
                    f"incorrect left raising coefficient at pole {k}, {label}"
                )
            if not np.allclose(right_up.coefficient, expected_right_up):
                raise AssertionError(
                    f"incorrect right raising coefficient at pole {k}, {label}"
                )
            if not np.allclose(left_down.coefficient, -1.0j * scale * pa.normfac_boson[k]):
                raise AssertionError(
                    f"incorrect left lowering coefficient at pole {k}, {label}"
                )
            if not np.allclose(right_down.coefficient, +1.0j * scale * pa.normfac_boson[k]):
                raise AssertionError(
                    f"incorrect right lowering coefficient at pole {k}, {label}"
                )
            if not np.allclose(left_down.coefficient + right_down.coefficient, 0.0):
                raise AssertionError(
                    f"boson lowering is not a commutator at pole {k}, {label}"
                )
            if not np.allclose(left_up.local_ops[site], bplus):
                raise AssertionError(f"raising term uses wrong ladder at pole {k}")
            if not np.allclose(left_down.local_ops[site], bminus):
                raise AssertionError(f"lowering term uses wrong ladder at pole {k}")

    for term in terms:
        if not np.isfinite(term.coefficient):
            raise AssertionError(f"non-finite coefficient in {term.name}")
        for site, op in term.local_ops.items():
            d = int(pa.nb[site])
            if op.shape != (d, d):
                raise AssertionError(
                    f"{term.name}: site {site} has {op.shape}, expected {(d, d)}"
                )

    # No bosonic site is allowed to acquire a Pauli-Z parity operator.
    boson_sites = {
        int(pa.site_system_boson_ket),
        int(pa.site_system_boson_bra),
        *map(int, pa.site_boson_diss),
    }
    fermion_terms = [t for t in terms if t.name.startswith(("zeta_", "xi_"))]
    for term in fermion_terms:
        if "_b_" in term.name:
            continue
        for site in boson_sites:
            op = term.local_ops.get(site)
            if op is not None:
                raise AssertionError(
                    f"fermionic parity leaked onto bosonic site {site} in {term.name}"
                )


if __name__ == "__main__":
    validate_term_structure()
    generated = build_product_terms()
    print("construct.py structural validation passed")
    print("number of product terms =", len(generated))
    print("first term =", generated[0].name)
    print("last term =", generated[-1].name)
