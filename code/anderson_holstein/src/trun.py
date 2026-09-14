######################################################
# Python code of the HEOM+MPS
# Author: Qiang Shi, Meng Xu, Xiaohan Dan, ICCAS
# Main reference: Q. Shi, Y. Xu, Y.-M. Yan, and M. Xu, 
# “Efficient propagation of the hierarchical equations 
# of motion using the matrix product state method”, 
# J. Chem. Phys. 148, 174102 (2018).
#
# trun.py: subroutines to truncate an MPS, (or MPO)
#######################################################

import numpy as np
import params as pa
import split as sp
import scipy.linalg as lg


#==========================================
# truncation based on the 2-sweep method
# Note, added normaliation similar to the orginal fortran code....
# also use a custom svd truncation, the one with tn may cause problems somehow

def trun_tensor(rin):

  rout = pa.Density()
  rout.nb = rin.nb
  nlen = len(rin.nodes)
  nrm = []

#----------------------------------------------------
  # QR for the first matrix
  q, r = sp.split_qr(rin.nodes[0])
  rout.nodes.append(q)


  # normalize
  nrml = np.sqrt(np.sum(np.abs(r)**2))
  #print("nrml =", nrml)
  if (nrml < 1.e-2): nrml = 1.0

  nrm.append(nrml)
  r *= 1.0/nrml

#----------------------------------------------------
# middle ones
  for i in range(1, nlen-1,1):

    rtmp = np.tensordot(r, rin.nodes[i], (1,0))
    q, r = sp.split_qr(rtmp)
    rout.nodes.append(q)


    nrm.append(np.sqrt(np.sum(np.abs(r)**2)))
    if (nrm[i] < 1.e-2 ): nrm[i] = 1.0
    r *= 1.0/nrm[i]


#----------------------------------------------------
# the last one
  rtmp = np.tensordot(r, rin.nodes[nlen-1], ((1),(0)) )
  rout.nodes.append(rtmp)


#----------------------------------------------------
  # the real truncation from the right
  rin = trun_tensor_right(rout)


#----------------------------------------------------
  # get the renormalization factors back 
  nrml = np.sum(np.log(nrm))
  nrml = np.exp(nrml/nlen)
    
  for i in range(nlen):
    rin.nodes[i] *= nrml

  return rin


#==============================================
def trun_tensor_right(rin):
  #print("trun_right, shape of rin",rin.ndim())

  rout = rin.copy()
  nlen = len(rin.nodes)

#----------------------------------------------------
  # split useing svd, the right matrix
  u1, vt = sp.split_svd_rq(rin.nodes[nlen-1])

  #can not use u and vt directly, some issues with dangling edge
  rout.nodes[nlen-1] = vt


#----------------------------------------------------
  #intermediate terms
  for i in range(nlen-2,0,-1):
    rtmp = np.tensordot(rin.nodes[i], u1, ((2),(0)))


    u1, vt = sp.split_svd_rq(rtmp)

# can not use u and vt directly, some issues with dangling edge
    rout.nodes[i] = vt


#----------------------------------------------------
# the left matrix
  rtmp = np.tensordot(rin.nodes[0], u1, ((2),(0)))
  rout.nodes[0] = rtmp


  #print("shape of rout",rout.ndim())
  return rout


