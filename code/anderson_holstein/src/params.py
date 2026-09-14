"""Parameters and index maps for the split-system Anderson-Holstein MPS-DQME.

This file is based on ``Full_V1.39/params.py``.  It deliberately contains no
TDVP code.  Its main purpose is to make three index spaces unambiguous:

1. physical fermionic dissipaton modes ``(sgn, spin, orbital, lead, pole)``;
2. their mirror-symmetric positions on the MPS chain;
3. the MPS sites, including the explicitly split system spin orbitals.

The two end sites are the ket/bra system phonons.  The electronic system sites
are placed immediately inside the ends, so the legacy TDVP boundary treatment
can be retained.
"""

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from scipy.linalg import expm


# =============================================================================
# Numerical propagation parameters (kept compatible with program b)
# =============================================================================

small = 1.0e-14
mmax = 20
nrtt = 240
nrmax = 240

hbar = 0.658211957
nsteps = 6000
ncorrstep = 1
dt = 0.025

maxIt = 100
lk_mmax = 30
gmres_tol = 1.0e-5

read_pall = False
write_pall = False
pall_file = "pall_aim"

Isread_rho = False
Isprop_rho = True
Iswrite_rho = True
rho_checkpoint_file = "rhoall.dat"

# The legacy two-time correlation driver is retained for compatibility but is
# disabled in the validated transport workflow.
run_correlation = False


# =============================================================================
# Physical model and bath-decomposition sizes
# =============================================================================

nsgnf = 2
nspinf = 2
nvarf = 1
nalphaf = 2
Mf = 6

nmodeb = 1
Mb = 6

# Local Hilbert-space cutoffs.  Both are convergence parameters.
nboson = 10                    # system phonon: 0, ..., nboson-1
nb2 = 10                       # each bosonic dissipaton cutoff
nb1 = 2                        # every fermionic site is empty/occupied

# Anderson-Holstein parameters.  The split-system DQME defaults to the original
# (untransformed) Hamiltonian written in the paper.  The optional Lang-Firsov
# objects below are retained only as references for later comparisons.
omega = 1.0
lamb = np.sqrt(0.1)
g_ep = lamb
use_lang_firsov = False
e1 = -4.9
e2 = -4.9
Uu = 10.2

# This comparison build treats the existing LF-frame HEOM program as the
# reference model. Since U (Q + 2 lambda N) U^dagger = Q for the convention
# used by Full_V1.39, the pre-LF DQME bosonic bath couples to
# Q + 2 lambda (n_up + n_down), rather than to Q alone.
boson_bath_charge_shift = 2.0 * lamb

# Useful aliases for a later multi-orbital extension.
epsilon = np.array([[e1, e2]], dtype=np.float64)  # (orbital, spin)
SPIN_UP = 0
SPIN_DOWN = 1
LEAD_LEFT = 0
LEAD_RIGHT = 1
SGN_MINUS = 0                 # gamma shift -i V
SGN_PLUS = 1                  # gamma shift +i V


# =============================================================================
# MPS layout
# =============================================================================

nlevelhalf = nspinf * nvarf * nalphaf * Mf
nlevelf = nsgnf * nlevelhalf
nlevelb = nmodeb * Mb
ndissipaton = nlevelf + nlevelb

# vmid = system ket fermions + m + boson dissipatons + n + system bra fermions
nsite_system_fermion = nspinf * nvarf
nsite_mid = 2 * nsite_system_fermion + nlevelf + nlevelb
nsite_total = nsite_mid + 2

# Legacy code uses nlevel as the number of sites between the two endpoints.
nlevel = nsite_mid

site_system_boson_ket = 0
site_system_fermion_ket = np.arange(
    1, 1 + nsite_system_fermion, dtype=np.int64
)

_site_m_start = int(site_system_fermion_ket[-1]) + 1
site_fermion_diss_ket = np.arange(
    _site_m_start, _site_m_start + nlevelhalf, dtype=np.int64
)

_site_b_start = int(site_fermion_diss_ket[-1]) + 1
site_boson_diss = np.arange(
    _site_b_start, _site_b_start + nlevelb, dtype=np.int64
)

_site_n_start = int(site_boson_diss[-1]) + 1
site_fermion_diss_bra = np.arange(
    _site_n_start, _site_n_start + nlevelhalf, dtype=np.int64
)

_site_system_bra_start = int(site_fermion_diss_bra[-1]) + 1
site_system_fermion_bra = np.arange(
    _site_system_bra_start,
    _site_system_bra_start + nsite_system_fermion,
    dtype=np.int64,
)
site_system_boson_bra = nsite_total - 1

# Local physical dimensions of the MPS sites.
nb = np.full(nsite_total, nb1, dtype=np.int64)
nb[site_system_boson_ket] = nboson
nb[site_boson_diss] = nb2
nb[site_system_boson_bra] = nboson
nbmat = nb**2


@dataclass(frozen=True)
class FermionMode:
    """Physical labels and chain position of one fermionic dissipaton."""

    chain_index: int           # 0, ..., nlevelf-1 among fermion dissipatons
    site: int                  # absolute MPS site
    side: str                  # "ket" (m) or "bra" (n)
    sgn: int                   # SGN_MINUS or SGN_PLUS
    spin: int
    orbital: int
    lead: int
    pole: int


def _canonical_half_modes():
    """Return (spin, orbital, lead, pole) in the input-file block order."""
    return [
        (spin, orbital, lead, pole)
        for spin in range(nspinf)
        for orbital in range(nvarf)
        for lead in range(nalphaf)
        for pole in range(Mf)
    ]


_left_labels = _canonical_half_modes()
_right_labels = list(reversed(_left_labels))

fermion_modes = []
for q, (spin, orbital, lead, pole) in enumerate(_left_labels):
    fermion_modes.append(
        FermionMode(
            chain_index=q,
            site=int(site_fermion_diss_ket[q]),
            side="ket",
            sgn=SGN_MINUS,
            spin=spin,
            orbital=orbital,
            lead=lead,
            pole=pole,
        )
    )

for q_right, (spin, orbital, lead, pole) in enumerate(_right_labels):
    q = nlevelhalf + q_right
    fermion_modes.append(
        FermionMode(
            chain_index=q,
            site=int(site_fermion_diss_bra[q_right]),
            side="bra",
            sgn=SGN_PLUS,
            spin=spin,
            orbital=orbital,
            lead=lead,
            pole=pole,
        )
    )

fermion_diss_to_site = np.array([mode.site for mode in fermion_modes], dtype=np.int64)
boson_diss_to_site = site_boson_diss.copy()

# The system bra sites are the mirror of the system ket sites.  With one orbital:
# ket = [up, down], bra = [down, up].
system_fermion_ket_site = {
    (orbital, spin): int(site_system_fermion_ket[orbital * nspinf + spin])
    for orbital in range(nvarf)
    for spin in range(nspinf)
}
system_fermion_bra_site = {
    (orbital, spin): int(
        site_system_fermion_bra[
            nsite_system_fermion - 1 - (orbital * nspinf + spin)
        ]
    )
    for orbital in range(nvarf)
    for spin in range(nspinf)
}

site_kind = np.full(nsite_total, "fermion", dtype=object)
site_kind[site_system_boson_ket] = "system_boson_ket"
site_kind[site_boson_diss] = "boson_dissipaton"
site_kind[site_system_boson_bra] = "system_boson_bra"


# =============================================================================
# Voltage and correlation parameters
# =============================================================================

# Shape is (lead, spin, orbital).  Edit this table instead of four index loops.
voltage = np.zeros((nalphaf, nspinf, nvarf), dtype=np.float64)
voltage[LEAD_LEFT, SPIN_UP, 0] = 0.0
voltage[LEAD_LEFT, SPIN_DOWN, 0] = 0.0
voltage[LEAD_RIGHT, SPIN_UP, 0] = 1.39
voltage[LEAD_RIGHT, SPIN_DOWN, 0] = 1.39


def _voltage_shift(mode):
    sign = -1.0 if mode.sgn == SGN_MINUS else 1.0
    return 1.0j * sign * voltage[mode.lead, mode.spin, mode.orbital]


# Fermionic voltage shifts are already in mirror-chain order.
vol_fermion = np.array([_voltage_shift(mode) for mode in fermion_modes])

# Backward-compatible dissipaton array: fermions first, then bosons.
# It is NOT indexed by an absolute MPS site.
vol = np.concatenate(
    [vol_fermion, np.zeros(nlevelb, dtype=np.complex128)]
)

# In the release layout the Prony tables live in ``../data``.  The environment
# variable keeps batch jobs free to select a different decomposition without
# editing the source tree.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR = Path(os.environ.get("MPS_DQME_DATA_DIR", _DEFAULT_DATA_DIR))

gamma_file_fer = DATA_DIR / "expn_fer_T-0.01.dat"
eta_file_fer = DATA_DIR / "etal_fer_T-0.01.dat"
gamma_file_bos = DATA_DIR / "expn_bos_T-0.01.dat"
eta_file_bos = DATA_DIR / "etal_bos_T-0.01.dat"


def _load_vector(path, expected_size, name):
    data = np.atleast_1d(np.loadtxt(path, dtype=np.complex128)).reshape(-1)
    if data.size != expected_size:
        raise ValueError(
            f"{name}: expected {expected_size} entries in {path}, got {data.size}"
        )
    return data


gamma_poles_fer = _load_vector(gamma_file_fer, Mf, "fermion gamma")
eta_poles_fer = _load_vector(eta_file_fer, Mf, "fermion eta")
gamma_poles_bos = _load_vector(gamma_file_bos, Mb, "boson gamma")
eta_poles_bos = _load_vector(eta_file_bos, Mb, "boson eta")

# Program b supplies one pole table and repeats it for spin/orbital/lead/sign.
# Values below are generated from mode metadata, so replacing this assumption
# later does not change the MPS ordering.
gamma_fermion = np.array(
    [gamma_poles_fer[mode.pole] for mode in fermion_modes],
    dtype=np.complex128,
)
eta_fermion = np.array(
    [eta_poles_fer[mode.pole] for mode in fermion_modes],
    dtype=np.complex128,
)

# gamma convention inherited from program b: gamma + ( +/- i V ).
gamma_fermion = gamma_fermion + vol_fermion

# The current input has no separate eta-bar file.  This reproduces the pairing
# used in the benchmark params.py of program a: the opposite-sign half is the
# mirror partner, and eta_bar_star is the complex conjugate of that partner.
eta_bar_fermion = np.concatenate(
    [eta_fermion[nlevelhalf:][::-1], eta_fermion[:nlevelhalf][::-1]]
)
eta_bar_star_fermion = np.conj(eta_bar_fermion)

zeta_fermion = np.sqrt(np.sqrt(eta_fermion * eta_bar_star_fermion))
if np.any(np.abs(zeta_fermion) < small):
    bad = np.flatnonzero(np.abs(zeta_fermion) < small)
    raise ValueError(f"fermionic zeta is zero at dissipaton indices {bad.tolist()}")

ksi_fermion = eta_fermion / zeta_fermion
ksi_star_fermion = np.conj(ksi_fermion)

# Bosonic scaling inherited from the active (non-commented) implementation in
# program b.  Using sqrt(abs(eta)) is important here: the last supplied pole has
# Re(eta)=0 but Im(eta)=-0.1, so sqrt(Re(eta)) would make the representation
# singular.  construct.py can use eta_re/normfac, eta_im/normfac and normfac
# directly, exactly as program b does.
gamma_boson = np.tile(gamma_poles_bos, nmodeb)
eta_boson = np.tile(eta_poles_bos, nmodeb)
eta_re_boson = np.real(eta_boson).astype(np.complex128)
eta_im_boson = np.imag(eta_boson).astype(np.complex128)
normfac_boson = np.sqrt(np.abs(eta_boson))
normfac_boson[np.abs(eta_boson) < small] = 1.0e-7
eta_re_scaled_boson = eta_re_boson / normfac_boson
eta_im_scaled_boson = eta_im_boson / normfac_boson

# Compatibility aliases for the later DQME constructor.  Their precise use is
# kept explicit through the scaled arrays above rather than hidden in a formula.
zeta_boson = normfac_boson.astype(np.complex128)
ksi_boson = eta_im_scaled_boson.astype(np.complex128)

# Backward-compatible names.  These arrays live in dissipaton-index space.
wval_ct = np.concatenate([gamma_fermion, gamma_boson])
gval_ct = np.concatenate(
    [eta_fermion, np.zeros(nlevelb, dtype=np.complex128)]
)
gval_ct_bar = np.concatenate(
    [eta_bar_fermion, np.zeros(nlevelb, dtype=np.complex128)]
)
gval = np.concatenate(
    [np.zeros(nlevelf, dtype=np.complex128), eta_re_boson]
)
gval_i = np.concatenate(
    [np.zeros(nlevelf, dtype=np.complex128), eta_im_boson]
)
zeta = np.concatenate([zeta_fermion, zeta_boson])
ksi = np.concatenate([ksi_fermion, ksi_boson])
ksi_star = np.concatenate([ksi_star_fermion, np.conj(ksi_boson)])
normfac = np.concatenate(
    [np.ones(nlevelf, dtype=np.complex128), normfac_boson]
)


# =============================================================================
# Local operators for the split system and dense reference operators for tests
# =============================================================================

# Local occupation basis is [|0>, |1>].  Historical names are retained:
# sigma_minus creates a fermion and sigma_plus annihilates one.
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
sigma_plus = np.array([[0.0, 1.0], [0.0, 0.0]])
sigma_minus = np.array([[0.0, 0.0], [1.0, 0.0]])
Id = np.eye(2)
number_f = sigma_minus @ sigma_plus

boson_a = np.zeros((nboson, nboson), dtype=np.complex128)
for n in range(1, nboson):
    boson_a[n - 1, n] = np.sqrt(n)
boson_ad = boson_a.conj().T
I_boson = np.eye(nboson)
Q_system_boson = boson_ad + boson_a
N_system_boson = boson_ad @ boson_a

# Dense system matrices are references for unit tests only.  The production MPS
# stores the three system factors on separate sites.
nfermion = 2 ** nsite_system_fermion
ndvr_dense = nfermion * nboson
I_fermion = np.eye(nfermion)

A_up_plus = np.kron(sigma_plus, Id)
A_up_minus = np.kron(sigma_minus, Id)
A_down_plus = np.kron(sigma_z, sigma_plus)
A_down_minus = np.kron(sigma_z, sigma_minus)

b_plus_dense = np.kron(I_fermion, boson_ad)
b_minus_dense = np.kron(I_fermion, boson_a)
Qb_dense = b_plus_dense + b_minus_dense

TransU_plus = expm(lamb * (b_plus_dense - b_minus_dense))
TransU_minus = expm(-lamb * (b_plus_dense - b_minus_dense))

d_1_plus_dense = TransU_plus @ np.kron(A_up_plus, I_boson)
d_1_minus_dense = TransU_minus @ np.kron(A_up_minus, I_boson)
d_2_plus_dense = TransU_plus @ np.kron(A_down_plus, I_boson)
d_2_minus_dense = TransU_minus @ np.kron(A_down_minus, I_boson)

n_up_dense = A_up_minus @ A_up_plus
n_down_dense = A_down_minus @ A_down_plus
fermion_hsys_dense = (
    e1 * n_up_dense
    + e2 * n_down_dense
    + Uu * (n_up_dense @ n_down_dense)
)
hsys_dense = (
    np.kron(fermion_hsys_dense, I_boson)
    + omega * np.kron(I_fermion, N_system_boson)
    + g_ep * omega * (Qb_dense @ np.kron(n_up_dense + n_down_dense, I_boson))
)

# Legacy aliases are kept for diagnostic code only.  New construct.py must use
# local operators and site maps, not a 40-dimensional endpoint operator.
b_plus = b_plus_dense
b_minus = b_minus_dense
Qb = Qb_dense
d_1_plus = d_1_plus_dense
d_1_minus = d_1_minus_dense
d_2_plus = d_2_plus_dense
d_2_minus = d_2_minus_dense
hsys = hsys_dense

# Endpoint dimension in the new MPS is the system-phonon cutoff, not ndvr_dense.
ndvr = nboson


# =============================================================================
# Sanity checks and lightweight tensor containers
# =============================================================================

def validate_parameters():
    assert nsgnf == 2, "the present mirror construction assumes two sign halves"
    assert nspinf == 2, "the present SIAM implementation assumes spin up/down"
    assert len(fermion_modes) == nlevelf
    assert nsite_total == len(nb)
    assert site_system_boson_ket == 0
    assert site_system_boson_bra == nsite_total - 1
    assert np.array_equal(
        site_fermion_diss_bra,
        nsite_total - 1 - site_fermion_diss_ket[::-1],
    )
    assert system_fermion_bra_site[(0, SPIN_UP)] == nsite_total - 1 - system_fermion_ket_site[(0, SPIN_UP)]
    assert system_fermion_bra_site[(0, SPIN_DOWN)] == nsite_total - 1 - system_fermion_ket_site[(0, SPIN_DOWN)]
    assert wval_ct.shape == (ndissipaton,)
    assert zeta.shape == (ndissipaton,)
    assert ksi.shape == (ndissipaton,)
    assert nb[0] == nb[-1] == nboson
    assert np.all(nb[site_system_fermion_ket] == 2)
    assert np.all(nb[site_system_fermion_bra] == 2)


validate_parameters()


class Density:
    def __init__(self):
        self.nb = np.empty(nsite_total, dtype=np.int64)
        self.nodes = []

    def copy(self):
        rout = Density()
        rout.nb = self.nb.copy()
        rout.nodes = [node.copy() for node in self.nodes]
        return rout

    def ndim(self):
        dims = np.empty((3, len(self.nodes)), dtype=int)
        for i, node in enumerate(self.nodes):
            dims[:, i] = node.shape
        return dims


class Tensor_Train:
    def __init__(self):
        self.tt = []


if __name__ == "__main__":
    print("Anderson-Holstein split-system MPS-DQME parameters")
    print("nlevelhalf =", nlevelhalf)
    print("nlevelf =", nlevelf)
    print("nlevelb =", nlevelb)
    print("ndissipaton =", ndissipaton)
    print("nsite_mid =", nsite_mid)
    print("nsite_total =", nsite_total)
    print("nb =", nb)
    print("fermion_diss_to_site =", fermion_diss_to_site)
    print("boson_diss_to_site =", boson_diss_to_site)
    print("vol_fermion =", vol_fermion)
