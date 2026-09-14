######################################################
# Python code of the HEOM+MPS
# Author: Qiang Shi, Meng Xu, Xiaohan Dan, ICCAS
# Main reference: Q. Shi, Y. Xu, Y.-M. Yan, and M. Xu, 
# “Efficient propagation of the hierarchical equations 
# of motion using the matrix product state method”, 
# J. Chem. Phys. 148, 174102 (2018).
#
# ttfunc.py: subroutines for tensor contraction, 
# rk4 and krylov subspace method to update node
#######################################################

import sys
import numpy as np
from scipy.linalg import expm
import params as pa
#import tensornetwork as tn
#from tensornetwork.network_components import contract, Node


def phia_next(phi0,mat1,x1,y1,iright):

#  print("in next_phi, iright=", iright)
  
  dim1 = x1.shape
  dim2 = y1.shape
  dim3 = phi0.shape
  dima = mat1.shape

#  print("x1.shape", x1.shape)
#  print("y1.shape", y1.shape)
#  print("phi0.shape", phi0.shape)
#  print("mat1.shape", mat1.shape)

  rx1,m,rx2 = dim1
  ry1,n,ry2 = dim2
  ra1 = dima[0]
  ra2 = dima[3]

  if iright == 1: # from right to left
    if (dim3[0]!=rx2 or dim3[1]!=ra2 or dim3[2]!=ry2):
        sys.exit("unmatched phi0 size in _next_phi")

    xx1 = np.conj(x1)
    yy1 = y1
    #vtmp1 = np.reshape(mat1.tensor,(ra1,m,n,ra2),order='F')
    #rtmp1 = Node(vtmp1)
#    print("rtmp1.shape", rtmp1.shape)

#    xx1[1] ^ rtmp1[1]
#    xx1[2] ^ phi0[0]
#    rtmp1[2] ^ yy1[1]
#    rtmp1[3] ^ phi0[1]
#    yy1[2] ^ phi0[2]

#    phi = tn.contractors.greedy([xx1,rtmp1,phi0,yy1], 
#          output_edge_order = [xx1[0],rtmp1[0],yy1[0]])
    rtmp2 = np.tensordot(xx1,phi0,axes=((2),(0)))
    rtmp3 = np.tensordot(rtmp2,mat1,axes=((1,2),(1,3)))
    phi = np.tensordot(rtmp3,yy1,axes=((1,3),(2,1)))


  else: # from left to right
    if (dim3[0]!=rx1 or dim3[1]!=ra1 or dim3[2]!=ry1):
          sys.exit("unmatched phi0 size in _next_phi")

    xx1 = np.conj(x1)
    yy1 = y1
    #vtmp1 = np.reshape(mat1.tensor,(ra1,m,n,ra2),order='F')
    #rtmp1 = Node(vtmp1)

    #xx1[0] ^ phi0[0]
    #xx1[1] ^ rtmp1[1]
    #rtmp1[2] ^ yy1[1]
    #rtmp1[0] ^ phi0[1]
    #yy1[0] ^ phi0[2]

    #phi = tn.contractors.greedy([xx1,rtmp1,phi0,yy1], 
    #      output_edge_order = [xx1[2],rtmp1[3],yy1[2]])
#    rtmp = tn.reachable(xx1)
#    phi = tn.contractors.greedy(rtmp)
#    phi.reorder_edges([xx1[2],rtmp1[3],y1[2]])
    rtmp2 = np.tensordot(xx1,phi0,axes=((0),(0)))
    rtmp3 = np.tensordot(rtmp2,mat1,axes=((0,2),(1,0)))
    phi = np.tensordot(rtmp3,yy1,axes=((1,2),(0,1)))


  return phi


#======================================================
def delta_ssol(phi1,phi2,y1):
  #dim1 = phi1.shape
  #dim2 = phi2.shape
  #dimx = y1.shape

  #rx1,ra1,ry1 = dim1
  #rx2,ra2,ry2 = dim2

  #if (ra1 != ra2):
  #    exit("ra1.ne.ra2 in delta_ssol")
  #if (ry1 != dimx[0] or ry2 != dimx[1]):
  #    exit("y1 and x1 have different shape")

  #phi1[1] ^ phi2[1]
  #phi1[2] ^ y1[0]
  #y1[1] ^ phi2[2]

  #dy = tn.contractors.greedy([phi1,phi2,y1], 
  #     output_edge_order = [phi1[0],phi2[0]])
  #dy.tensor = -1.0*dy.tensor

  rtmp1 = np.tensordot(phi1,y1,axes=((2),(0)))
  dy = np.tensordot(rtmp1,phi2,axes=((1,2),(1,2)))

  dy *= -1.0

  return dy


#==========================================
def delta_ksol(mat1,phi1,phi2,y1):
  dim1 = phi1.shape
  dim2 = phi2.shape
  dima = mat1.shape
  dimy = y1.shape

#  print("in delta_ksol")
#  print("phi1.shape", phi1.shape)
#  print("phi2.shape", phi2.shape)
#  print("mat1.shape", mat1.shape)
#  print("y1.shape", y1.shape)

  rx1,ra1,ry1 = dim1
  rx2,ra2,ry2 = dim2
  m = n = dimy[1]

  if (dima[0] != ra1 or dima[3] != ra2):
      sys.exit("array size does not match in delta_ksol for mat1")

  if ((dimy[0] != ry1 or dimy[2] != ry2)):
      sys.exit("array size does not match in delta_ksol for x1")

  #vtmp1 = np.reshape(mat1.tensor,(ra1,m,n,ra2),order='F')
  #rtmp1 = Node(vtmp1)

  #phi1[1] ^ rtmp1[0]
  #phi1[2] ^ y1[0]
  #phi2[1] ^ rtmp1[3]
  #phi2[2] ^ y1[2]
  #rtmp1[2] ^ y1[1]

  #dy = tn.contractors.greedy([phi1,rtmp1,phi2,y1], 
  #     output_edge_order = [phi1[0],rtmp1[1],phi2[0]])
#  rtmp = tn.reachable(phi1)
#  dy = tn.contractors.greedy(rtmp)
#  dy.reorder_edges[phi1[0],rtmp1[1],phi2[0]]

  rtmp2 = np.tensordot(phi1,y1,axes=((2),(0)))
  rtmp3 = np.tensordot(rtmp2,mat1,axes=((1,2),(0,2)))
  dy = np.tensordot(rtmp3,phi2,axes=((1,3),(2,1)))


  return dy

#=====================================
def ddot2_ssol(vec1, vec2):

  xx1 = np.conj(vec1)
  yy1 = vec2

  ddot2 = np.tensordot(xx1,yy1,axes=((0,1), (0,1)))

  #ddot2 = 0.0

  #xx1 = conj(vec1)
  #yy1 = Node(vec2)


  #xx1[0] ^ yy1[0]
  #xx1[1] ^ yy1[1]


  #ddot2 = tn.contractors.greedy([xx1,yy1]).tensor

  return ddot2


#=====================================
def dnorm2_ssol(vec1):

  dnorm2 = ddot2_ssol(vec1,vec1)

  return np.sqrt(dnorm2)

#=====================================
def ddot2_ksol(vec1, vec2):

  #ddot2 = 0.0
  xx1 = np.conj(vec1)
  yy1 = vec2

  ddot2 = np.tensordot(xx1,yy1,axes=((0,1,2), (0,1,2)))

  #xx1 = conj(vec1)
  #yy1 = Node(vec2)

  #xx1[0] ^ yy1[0]
  #xx1[1] ^ yy1[1]
  #xx1[2] ^ yy1[2]

  #ddot2 = tn.contractors.greedy([xx1,yy1]).tensor

  return ddot2

#=====================================
def dnorm2_ksol(vec1):

  dnorm2 = ddot2_ksol(vec1,vec1)

  return np.sqrt(dnorm2)

#=================================================
def update_s_rk4(yy,delta_t,phi1,phi2):
  dt2 = delta_t/2.0
  dt6 = delta_t/6.0

  dy1 = delta_ssol(phi1,phi2,yy)
  y0 = yy + dt2*dy1

  dy2 = delta_ssol(phi1,phi2,y0)
  y0 = yy + dt2*dy2

  dy3 = delta_ssol(phi1,phi2,y0)
  y0 = yy+delta_t*dy3

  dy3 = dy2 + dy3
  dy2 = delta_ssol(phi1,phi2,y0)
  yy = yy + dt6*(dy1+2.0*dy3+dy2)

  return yy


#===============================================
def update_k_rk4(yy,delta_t,mat1,phi1,phi2):
  dt2 = delta_t/2.0
  dt6 = delta_t/6.0

  dy1 = delta_ksol(mat1,phi1,phi2,yy)
  y0 = yy+dt2*dy1

  dy2 = delta_ksol(mat1,phi1,phi2,y0)
  y0 =yy + dt2*dy2

  dy3 = delta_ksol(mat1,phi1,phi2,y0)
  y0 = yy+delta_t*dy3

  dy3 = dy2 + dy3
  dy2 = delta_ksol(mat1,phi1,phi2,y0)
  yy = yy + dt6*(dy1+2.0*dy3+dy2)

  return yy

#=========================================

#============================================
# use expokit to update ssol
def update_s(yy,delta_t,phi1,phi2):

#----------------------
  vm = []
  hmat = np.zeros((pa.mmax,pa.mmax),dtype=np.complex128)

#--------------------------
# the first vector
  rtmp0 = dnorm2_ssol(yy)
  #print("rtmp0=", rtmp0)

  y0 = 1.0/rtmp0*yy
  vm.append(y0)

#--------------------------------------------
  for j in range(pa.mmax):
    # dy1
    dy1 = delta_ssol(phi1,phi2,vm[j])
    dy1 *= delta_t

    # calculate the overlap
    for i in range(j+1):
      hmat[i,j] = ddot2_ssol(vm[i], dy1)

    # orthogonaization 
    for i in range(j+1):
      dy1 -=  hmat[i,j]*vm[i]

  
    # new basis
    rtmp = dnorm2_ssol(dy1)

    if (rtmp > 1.e-13):
      if (j < pa.mmax-1):
        hmat[j+1,j] = rtmp

        y0 = 1.0/rtmp*dy1

        vm.append(y0)
    else:
      #print("small rtmp:", rtmp)
      break

  jmax = j + 1
  #print("jmax in _s =", jmax, "mmax =", pa.mmax)
#--------------------------------------------
# calculate exp(hmat)
#  print ("lenth of vm", len(vm))
#  print("hmat =", hmat)
  exph = expm(hmat[0:jmax,0:jmax])
#  print("exph =", exph)
#  exit()
#--------------------------------------------
# the new yy
  vtmp = np.zeros(yy.shape,dtype=np.complex128)

#  for i in range(pa.mmax):
  for i in range(jmax):
#      yy[j].tensor = yy[j].tensor + rtmp0*exph[i,0]*vm[i][j].tensor
#      yy[j].tensor += rtmp0*exph[i,0]*vm[i][j].tensor
    vtmp += rtmp0*exph[i,0]*vm[i]

  yy = vtmp

  return yy


#============================================
# use expokit to update ksol
def update_k(yy,delta_t,mat1,phi1,phi2):
#----------------------
  vm = []
  hmat = np.zeros((pa.mmax,pa.mmax),dtype=np.complex128)

#  print("mat1 =", mat1)
#  print("phi1 =", phi1)
#  print("phi2 =", phi2)
#  mat10 = mat1.copy()
#  phi10 = phi1.copy()
#  phi20 = phi2.copy()
#--------------------------
# the first vector
  rtmp0 = dnorm2_ksol(yy)
  #print("rtmp0=", rtmp0)
#  vtmp = np.zeros(yy[0].shape,dtype=np.complex128)
  y0 = 1.0/rtmp0*yy
  vm.append(y0)

#  print("y0=", y0)
#--------------------------------------------
  for j in range(pa.mmax):
    # dy1
    #print(np.isclose(mat10,mat1))
    dy1 = delta_ksol(mat1,phi1,phi2,vm[j])
#    print("dy1=", dy1)
    dy1 *= delta_t

    # calculate the overlap
    for i in range(j+1):
      hmat[i,j] = ddot2_ksol(vm[i], dy1)

    # orthogonaization 
    for i in range(j+1):
      dy1 -=  hmat[i,j]*vm[i]
  
    # new basis
    rtmp = dnorm2_ksol(dy1)
# important, if rtmp = 0, the break, and set array dimension to j
    if (rtmp > 1.e-13):
      if (j < pa.mmax-1):
        hmat[j+1,j] = rtmp
        #print("j =", j, "rtmp = ", rtmp)
    # need to get a new y0, otherwise not good....
    #  vtmp = np.zeros(yy[0].shape,dtype=np.complex128)
        y0 = 1.0/rtmp*dy1
        vm.append(y0)
    else:
      #print("small rtmp:", rtmp)
      break
  
  jmax = j + 1
  #print("jmax in _k =", jmax, "mmax =", pa.mmax)
#--------------------------------------------
# calculate exp(hmat)
#  print ("lenth of vm", len(vm))
#  print("hmat =", hmat)
  exph = expm(hmat[0:jmax,0:jmax])
#  print("exph =", exph)
#--------------------------------------------
# the new yy
    #yy[j].tensor = 0.0
  vtmp = np.zeros(yy.shape,dtype=np.complex128)

#  for i in range(pa.mmax):
  for i in range(jmax):
#      yy[j].tensor = yy[j].tensor + rtmp0*exph[i,0]*vm[i][j].tensor
#      yy[j].tensor += rtmp0*exph[i,0]*vm[i][j].tensor
    vtmp += rtmp0*exph[i,0]*vm[i]

  yy = vtmp

  return yy

