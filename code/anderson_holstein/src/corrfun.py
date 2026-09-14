######################################################
# Python code of the HEOM+MPS
# Author: Qiang Shi, Meng Xu, Xiaohan Dan, ICCAS
# Main reference: Q. Shi, Y. Xu, Y.-M. Yan, and M. Xu, 
# “Efficient propagation of the hierarchical equations 
# of motion using the matrix product state method”, 
# J. Chem. Phys. 148, 174102 (2018).
#
# corrfun.py: calculate the correlation function
#######################################################

import numpy as np
import params as pa
import print_tensor as pt
import add_tensor as at
import write_file as wf
import ksltt as ksl
import calc_rho as cr


def corrfun(rin,pall,iscommu):
  print('in corrfun!')  
  print('rin.ndim',rin.ndim())

  matA = pa.aminus1
  matB = pa.aplus1
  nlen = len(rin.nodes)

  #sig_y,sig_z = construct_operator()
  #r1 = at.prod_tensor_mat(sig_y,rin)
  print('r1.ndim',rin.ndim())
  print('pall.ndim',pall.ndim()) 
  if(iscommu):
    rin.nodes[0][0,:,:] = np.dot(matB, rin.nodes[0][0,:,:])
  else:
    rin.nodes[nlen-1][:,:,0] = np.dot(rin.nodes[nlen-1][:,:,0],matB)

  r1 = rin 
  s1='   '
  if(iscommu):
    file1='1crr_c.dat'
  else: 
    file1='1crr_a.dat'

  rhotmp = cr.calc_rho(r1)
  if(iscommu):
    rho2 = np.dot(matA, rhotmp)
  else:
    rho2 = np.dot( rhotmp,matA)
    #test the invariance of trace
    #rho3 = np.dot( matA,rhotmp)

  out1=str(0)+s1+str(np.trace(rho2).real)+s1+str(np.trace(rho2).imag)+'\n'
  wf.write(file1,out1,newfile=True)
  #if(not iscommu):
  #  out_invt=str(0)+s1+str(np.trace(rho3).real)+s1+str(np.trace(rho3).imag)+'\n'
  #  wf.write('2crr_at.dat',out_invt,newfile=True)

  for istep in range(1,pa.ncorrstep+1):
    print('corr step=',istep)    
    r1 = ksl.ksltt(r1,pall)
    print('to add')

    rhotmp = cr.calc_rho(r1)
    if(iscommu):
      rho2 = np.dot(matA, rhotmp)
    else:
      rho2 = np.dot( rhotmp,matA)
      #rho3 = np.dot( matA,rhotmp)

    out1=str(istep*pa.dt)+s1+str(np.trace(rho2).real)+s1+str(np.trace(rho2).imag)+'\n'
    wf.write(file1,out1)
    #if(not iscommu):
    #  out_invt=str(istep*pa.dt)+s1+str(np.trace(rho3).real)+s1+str(np.trace(rho3).imag)+'\n'
    #  wf.write('2crr_at.dat',out_invt)
  


#==========================================================
def construct_operator():
  p0 = pa.Density()
  p0.nb = pa.nbmat


# vl and vr
  vl = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
  vr = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
  for i in range(pa.ndvr):
    ii = i*pa.ndvr+i
    vl[0,ii,0] = 1.0
    vr[0,ii,0] = 1.0


# vmid
  vmid = []

  for i in range(pa.nlevel):

    vtmp = np.zeros((1, pa.nbmat[i+1], 1),dtype=np.complex128)

    for j in range(pa.nb[i+1]):
      ii = j*pa.nb[i+1]+j
      vtmp[0,ii,0] = 1.0


    vmid.append(vtmp)


# add to p0.nodes
  p0.nodes.append(vl)

  for i in range(pa.nlevel):
    p0.nodes.append(vmid[i])

  p0.nodes.append(vr)


  print("Operater calculation done!")
  Sigy = sigma_y(p0)
  Sigz = sigma_y(p0)
  
  Sigynew = copy2tensorMat0(Sigy)
  Sigznew = copy2tensorMat0(Sigz)

  return Sigynew,Sigznew 


def sigma_y(rho):
  #MPS right multiply sigma_y operates
  drho = rho.copy()
  nlen = len(rho.nodes)

  sigy=np.zeros((pa.ndvr,pa.ndvr),dtype=np.complex128)
  sigy[0,1]=-1.0j
  sigy[1,0]=1.0j
  
  #vl = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
  vr = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
  for i in range(pa.ndvr):
    for j in range(pa.ndvr):
      ii = j*pa.ndvr + i
      vr[0,ii,0] =  sigy[j,i]

  drho.nodes[nlen-1] = vr
  return drho


def sigma_z(rho):
  #MPS right multiply sigma_z operates
  #just for compare 
  #20201109--dxh
  drho = rho.copy()
  nlen = len(rho.nodes)

  sigz=np.zeros((pa.ndvr,pa.ndvr),dtype=np.complex128)
  sigz[0,0]=1.0
  sigz[1,1]=-1.0

  vr = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
  for i in range(pa.ndvr):
    for j in range(pa.ndvr):
      ii = j*pa.ndvr + i
      vr[0,ii,0] =  sigz[j,i]

  drho.nodes[nlen-1] = vr
  return drho

#==============================================
# copy the TT to a new one...
def copy2tensorMat0(node1):

  rout = pa.Density()

  rout.nb = node1.nb.copy()

  nlen = len(node1.nodes)


  for i in range(nlen):
    # need to zero and then add, otherwise strange problems
    ra1, rmid, ra2 = node1.nodes[i].shape

    #print(ra1, rmid, ra2)
    vtmp = np.reshape(node1.nodes[i],(ra1,pa.nb[i],pa.nb[i],ra2),order='F')

    rout.nodes.append(vtmp)
  return rout


##==========================================================
#def delta_all5(rho):
#
#  drho = rho.copy()
#  rtmp1 = rho.copy()
#  nlen = len(rho.nodes)
#
##========================================================
## the Hamiltonian term
#  vl = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
#  vr = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
#  for i in range(pa.ndvr):
#    for j in range(pa.ndvr):
#      ii = j*pa.ndvr + i
#      vl[0,ii,0] = -1.0j*pa.hsys[i,j]
#      vr[0,ii,0] =  1.0j*pa.hsys[j,i]
#
#
#  drho.nodes[0] = vl
#  rtmp1.nodes[nlen-1] = vr
#  drho = at.add_tensor(drho,rtmp1,1.0)
#  print("after hsys, shape of rout",drho.ndim())
#
##========================================================
## the k0 term
#  print("k0=",pa.k0)
#  rtmp1 = rho.copy()
#  rtmp2 = rho.copy()
#  vl = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
#  vr = np.zeros((1, pa.ndvr*pa.ndvr, 1),dtype=np.complex128)
#
#
#  for i in range(pa.ndvr):
#    ii = i*pa.ndvr+i
#    vl[0,ii,0] = pa.sdvr[i]**2
#    vr[0,ii,0] = pa.sdvr[i]**2
#
#
#  rtmp1.nodes[0] = vl
#  rtmp2.nodes[nlen-1] = vr
#  rtmp1 = at.add_tensor(rtmp1,rtmp2,1.0)
#
#
#  rtmp2 = rho.copy()
#  for i in range(pa.ndvr):
#    for j in range(pa.ndvr):
#      ii = i*pa.ndvr+i
#      jj = j*pa.ndvr+j
#      vl[0,ii,0] = -2.0*pa.sdvr[i]
#      vr[0,jj,0] = pa.sdvr[j]
#
#
#  rtmp2.nodes[0] = vl
#  rtmp2.nodes[nlen-1] = vr
#
#
#  rtmp1 = at.add_tensor(rtmp1,rtmp2,1.0)
#  drho = at.add_tensor(drho,rtmp1,-1.0*pa.k0)
#
#
#  print("after k0, shape of rout",drho.ndim())
#
##========================================================
## the diagonal terms
#  for i in range(1,nlen-1):
#    rtmp1 = rho.copy()
#    vtmp = np.zeros((1, pa.nb[i]*pa.nb[i], 1),dtype=np.complex128)
#    for j in range(pa.nb[i]):
#      jj = j*pa.nb[i]+j
#      vtmp[0,jj,0] = -1.0*j*pa.wval[i-1]
#
#
#    rtmp1.nodes[i] = vtmp
#    drho = at.add_tensor(drho,rtmp1,1.0)
#
#
#  print("after diagonal, shape of rout",drho.ndim())
##========================================================
## the hierarchical terms
#  for i in range(1,nlen-1):
#
#
#    rtmp1 = rho.copy()
#    rtmp2 = rho.copy()
#    vl = rtmp1.nodes[0].copy()
#    vr = rtmp2.nodes[nlen-1].copy()
#
#
#    for j in range(pa.ndvr):
#      jj = j*pa.ndvr+j
#      vl[0,jj,0] = pa.sdvr[j]
#      vr[0,jj,0] = pa.sdvr[j]
#
#
#    vtmp1 = np.zeros((1, pa.nb[i]*pa.nb[i], 1),dtype=np.complex128)
#    vtmp2 = np.zeros((1, pa.nb[i]*pa.nb[i], 1),dtype=np.complex128)
#
#
#    for j in range(pa.nb[i]):
#
##-------------------------------------------------------------------------
#      j1 = j-1
#      if j1>=0:
#        jj = j1*pa.nb[i]+j
#        vtmp1[0,jj,0] += pa.gval_i[i-1]/pa.normfac[i-1]*np.sqrt(1.0+j1)
#        vtmp1[0,jj,0] -= 1.0j*pa.gval[i-1]/pa.normfac[i-1]*np.sqrt(1.0+j1)
#
#        vtmp2[0,jj,0] += pa.gval_i[i-1]/pa.normfac[i-1]*np.sqrt(1.0+j1)
#        vtmp2[0,jj,0] += 1.0j*pa.gval[i-1]/pa.normfac[i-1]*np.sqrt(1.0+j1)
#
##-------------------------------------------------------------------------
#      j1 = j+1
#      if j1<pa.nb[i]:
#        jj = j1*pa.nb[i]+j
#        vtmp1[0,jj,0] -= 1.0j*pa.normfac[i-1]*np.sqrt(j1*1.0)
#        vtmp2[0,jj,0] += 1.0j*pa.normfac[i-1]*np.sqrt(j1*1.0)
#
#
#    rtmp1.nodes[0] = vl
#    rtmp1.nodes[i] = vtmp1
#
#    rtmp2.nodes[nlen-1] = vr
#    rtmp2.nodes[i] = vtmp2
#
#    drho = at.add_tensor(drho,rtmp1,1.0)
#    drho = at.add_tensor(drho,rtmp2,1.0)
#
#  print("after hierachial, shape of rout",drho.ndim())
#  return drho
