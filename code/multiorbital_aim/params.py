"""Parameters and mirror-symmetric site maps for the pure-fermion AIM DQME."""
from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np

small=1e-14; mmax=20; nrtt=200; nrmax=200; hbar=0.658211957
nsteps=1000; ncorrstep=1; dt=0.025; maxIt=100; lk_mmax=30; gmres_tol=1e-5
read_pall=False; write_pall=False; pall_file="pall_aim"
Isread_rho=False; Isprop_rho=True; Iswrite_rho=True
rho_checkpoint_file="rhoall_multiorbital_dqme.dat"; run_correlation=False

# Physical sizes.  Change nvarf and epsilon together.
nsgnf=2; nspinf=2; nvarf=2; nalphaf=2; Mf=6; nb1=2
SPIN_UP=0; SPIN_DOWN=1; LEAD_LEFT=0; LEAD_RIGHT=1
SGN_MINUS=0; SGN_PLUS=1

epsilon = np.array(
    [
        [-4.8,          0.35 + 0.12j],
        [0.35 - 0.12j, -4.2         ],
    ],
    dtype=np.complex128,
)

U = 10.0
Uprime = 6.0
J1 = 0.9
J2 = 0.7
J3 = 0.5

# voltage[lead,spin,orbital]
voltage=np.zeros((nalphaf,nspinf,nvarf),dtype=float)
voltage[LEAD_RIGHT,:,:]=-1.0

nsite_system=nspinf*nvarf
nlevelhalf=nspinf*nvarf*nalphaf*Mf
nlevelf=2*nlevelhalf
nsite_total=2*nsite_system+nlevelf
nlevel=nsite_total-2                 # sites in vmid; TDVP interface is unchanged
ndissipaton=nlevelf

site_system_fermion_ket=np.arange(0,nsite_system,dtype=np.int64)
site_fermion_diss_ket=np.arange(nsite_system,nsite_system+nlevelhalf,dtype=np.int64)
site_fermion_diss_bra=np.arange(nsite_system+nlevelhalf,nsite_system+nlevelf,dtype=np.int64)
site_system_fermion_bra=np.arange(nsite_system+nlevelf,nsite_total,dtype=np.int64)

system_fermion_ket_site={(m,s):int(site_system_fermion_ket[m*nspinf+s])
    for m in range(nvarf) for s in range(nspinf)}
system_fermion_bra_site={(m,s):int(site_system_fermion_bra[nsite_system-1-(m*nspinf+s)])
    for m in range(nvarf) for s in range(nspinf)}

nb=np.full(nsite_total,nb1,dtype=np.int64); nbmat=nb**2

@dataclass(frozen=True)
class FermionMode:
    chain_index:int; site:int; side:str; sgn:int
    spin:int; orbital:int; lead:int; pole:int

def _half_labels():
    return [(s,m,a,k) for s in range(nspinf) for m in range(nvarf)
            for a in range(nalphaf) for k in range(Mf)]

_left=_half_labels(); _right=list(reversed(_left)); fermion_modes=[]
for q,(s,m,a,k) in enumerate(_left):
    fermion_modes.append(FermionMode(q,int(site_fermion_diss_ket[q]),"ket",SGN_MINUS,s,m,a,k))
for qr,(s,m,a,k) in enumerate(_right):
    fermion_modes.append(FermionMode(nlevelhalf+qr,int(site_fermion_diss_bra[qr]),"bra",SGN_PLUS,s,m,a,k))
fermion_diss_to_site=np.array([x.site for x in fermion_modes],dtype=np.int64)

def _vshift(mode):
    return ( -1j if mode.sgn==SGN_MINUS else 1j)*voltage[mode.lead,mode.spin,mode.orbital]
vol_fermion=np.array([_vshift(x) for x in fermion_modes])
vol=vol_fermion

DATA_DIR=Path(os.environ.get("MPS_DQME_DATA_DIR",Path(__file__).resolve().parent))
def _load(name):
    x=np.atleast_1d(np.loadtxt(DATA_DIR/name,dtype=np.complex128)).reshape(-1)
    if x.size!=Mf: raise ValueError(f"{name}: expected {Mf} poles, got {x.size}")
    return x
gamma_poles_fer=_load("expn1.dat")
eta_poles_fer=_load("etal1.dat")
gamma_fermion=np.array([gamma_poles_fer[x.pole] for x in fermion_modes])+vol_fermion
eta_fermion=np.array([eta_poles_fer[x.pole] for x in fermion_modes])
eta_bar_fermion=np.concatenate([eta_fermion[nlevelhalf:][::-1],eta_fermion[:nlevelhalf][::-1]])
eta_bar_star_fermion=np.conj(eta_bar_fermion)
zeta_fermion=np.sqrt(np.sqrt(eta_fermion*eta_bar_star_fermion))
if np.any(np.abs(zeta_fermion)<small): raise ValueError("zero fermionic zeta")
ksi_fermion=eta_fermion/zeta_fermion; ksi_star_fermion=np.conj(ksi_fermion)
wval_ct=gamma_fermion; gval_ct=eta_fermion; gval_ct_bar=eta_bar_fermion
zeta=zeta_fermion; ksi=ksi_fermion; ksi_star=ksi_star_fermion
normfac=np.ones(nlevelf,dtype=np.complex128)

# Historical convention: sigma_minus creates; sigma_plus annihilates.
sigma_plus=np.array([[0.,1.],[0.,0.]],dtype=np.complex128)
sigma_minus=np.array([[0.,0.],[1.,0.]],dtype=np.complex128)
sigma_z=np.diag([1.,-1.]).astype(np.complex128); Id=np.eye(2,dtype=np.complex128)
number_f=sigma_minus@sigma_plus
nfermion=2**nsite_system; ndvr_dense=nfermion; ndvr=2

def validate_parameters():
    assert nsgnf==2 and nspinf==2
    assert epsilon.shape==(nvarf,nvarf)
    if not np.allclose(epsilon,epsilon.conj().T): raise ValueError("epsilon must be Hermitian")
    assert site_system_fermion_ket[0]==0 and site_system_fermion_bra[-1]==nsite_total-1
    assert np.array_equal(site_fermion_diss_bra,nsite_total-1-site_fermion_diss_ket[::-1])
    for key,left in system_fermion_ket_site.items():
        assert system_fermion_bra_site[key]==nsite_total-1-left
validate_parameters()

class Density:
    def __init__(self): self.nb=np.empty(nsite_total,dtype=np.int64); self.nodes=[]
    def copy(self):
        x=Density(); x.nb=self.nb.copy(); x.nodes=[n.copy() for n in self.nodes]; return x
    def ndim(self): return np.array([n.shape for n in self.nodes]).T
class Tensor_Train:
    def __init__(self): self.tt=[]
