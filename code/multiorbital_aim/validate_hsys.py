"""Independent dense validation of the split-system Hamiltonian terms."""
import numpy as np
import params as pa
import construct as cst

def kron_all(ops):
    out=np.asarray(ops[0],dtype=np.complex128)
    for op in ops[1:]: out=np.kron(out,op)
    return out

def d_operator(m,s):
    q=m*pa.nspinf+s
    return kron_all([pa.sigma_z if j<q else (pa.sigma_plus if j==q else pa.Id)
                     for j in range(pa.nsite_system)])

def dense_reference():
    d={(m,s):d_operator(m,s) for m in range(pa.nvarf) for s in range(pa.nspinf)}
    dd={k:v.conj().T for k,v in d.items()}; n={k:dd[k]@d[k] for k in d}
    H=np.zeros((pa.ndvr_dense,pa.ndvr_dense),dtype=np.complex128)
    for m in range(pa.nvarf):
      for mp in range(pa.nvarf):
       for s in range(pa.nspinf): H += pa.epsilon[m,mp]*dd[m,s]@d[mp,s]
    for m in range(pa.nvarf): H += pa.U*n[m,0]@n[m,1]
    for m in range(pa.nvarf):
      for mp in range(m+1,pa.nvarf):
       for s in range(pa.nspinf):
        H += pa.Uprime*n[m,s]@n[mp,1-s]
        H += (pa.Uprime-pa.J1)*n[m,s]@n[mp,s]
    for m in range(pa.nvarf):
      for mp in range(pa.nvarf):
       if m==mp: continue
       H += -pa.J2*dd[m,0]@d[m,1]@dd[mp,1]@d[mp,0]
       H += pa.J3*dd[m,0]@dd[m,1]@d[mp,1]@d[mp,0]
    return H

def dense_from_product_terms():
    H=np.zeros((pa.ndvr_dense,pa.ndvr_dense),dtype=np.complex128)
    for term in cst._system_hamiltonian_terms():
        if not term.name.endswith("_ket"): continue
        ops=[term.local_ops.get(pa.system_fermion_ket_site[(m,s)],pa.Id)
             for m in range(pa.nvarf) for s in range(pa.nspinf)]
        H += (term.coefficient/(-1j))*kron_all(ops)
    return H

def validate():
    ref=dense_reference(); split=dense_from_product_terms()
    err=np.max(np.abs(ref-split)); herm=np.max(np.abs(ref-ref.conj().T))
    if err>1e-12: raise AssertionError(f"split/dense Hsys error {err}")
    if herm>1e-12: raise AssertionError(f"Hsys is not Hermitian: {herm}")
    print("Hsys dense/product validation passed")
    print("dimension =",pa.ndvr_dense,"max error =",err,"Hermiticity defect =",herm)

if __name__=="__main__": validate()
