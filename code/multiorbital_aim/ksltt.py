######################################################
# Python code of the HEOM+MPS
# Author: Qiang Shi, Meng Xu, Xiaohan Dan, ICCAS
# Main reference: Q. Shi, Y. Xu, Y.-M. Yan, and M. Xu, 
# “Efficient propagation of the hierarchical equations 
# of motion using the matrix product state method”, 
# J. Chem. Phys. 148, 174102 (2018).
#
# ksltt.py: the main TDVP subroutines
#######################################################

import numpy as np
import scipy.linalg as lg
import params as pa
import ttfunc as ttf
#from tensornetwork.network_operations import split_node, split_node_qr, split_node_rq
#from tensornetwork.network_components import contract, Node


def ksltt(rin,pall):

  dt2 = 0.5*pa.dt

  nlen = len(rin.nodes)
  r1 = pa.Density()
  r2 = pa.Density()
  r3 = pa.Density()

  r1.nb = rin.nb
  r2.nb = rin.nb
  r3.nb = rin.nb

  phia = pa.Tensor_Train()


  # phia[nlevel+2]
  vtmp = np.ones((1,1,1),dtype=np.complex128)
  phia.tt.append(vtmp)

#======================================================
# ortho from right
  #r, q = split_node_rq(rin.nodes[nlen-1],
  #       left_edges=[rin.nodes[nlen-1][0]],
  #       right_edges=[rin.nodes[nlen-1][1], rin.nodes[nlen-1][2]])

  r, q = split_rq(rin.nodes[nlen-1])

  
  r1.nodes.append(q)
  u1 = r
# phia[nlevel+1]
  phia.tt.append(ttf.phia_next(phia.tt[0],
                 pall.nodes[nlen-1],q,q,1))
#-------------------------------------------------------
# intermediate terms
  for i in range(nlen-2,0,-1):
#    print("ortho from right, i =", i)
    #edge = rin.nodes[i][2] ^ u1[0]
    #rtmp = contract(edge)
    #r, q = split_node_rq(
    #       rtmp, left_edges=[rtmp[0]],
    #       right_edges=[rtmp[1],rtmp[2]])

    rtmp = np.tensordot(rin.nodes[i], u1, axes=((2),(0)))

    r, q = split_rq(rtmp)
    r1.nodes.append(q)
    u1 = r

    phia.tt.append(ttf.phia_next(phia.tt[nlen-1-i],
                   pall.nodes[i],q,q,1))

# the left matrix
#  edge = rin.nodes[0][2] ^ u1[0]
#  r1.nodes.append(contract(edge))

  rtmp = np.tensordot(rin.nodes[0], u1, axes=((2),(0)))
  r1.nodes.append(rtmp)

  r1.nodes.reverse()

# add phia[0] and reverse to normal
  vtmp = np.ones((1,1,1),dtype=np.complex128)
  phia.tt.append(vtmp)
  phia.tt.reverse()

  
#  for i in range(nlen):
#    print("shape of r1, i=", i, " is ", r1.nodes[i].shape)
#===============================================
###### the first part of KSL, from left to right
  for i in range(nlen-1):
#    print("update from left, i=", i)
    phi1 = phia.tt[i]
    phi2 = phia.tt[i+1]

    if (i == 0):
      ksol = r1.nodes[i].copy()
    else:
      ksol = r2.nodes[i].copy()


    ksol = ttf.update_k(ksol,dt2,pall.nodes[i],phi1,phi2)
#    q, r = split_node_qr(ksol, left_edges=[ksol[0], ksol[1]],
#           right_edges=[ksol[2]])
    q, r = split_qr(ksol)


    if (i == 0):
      r2.nodes.append(q)
    else:
      r2.nodes[i] = q


    phi1 = ttf.phia_next(phi1,pall.nodes[i],q,q,0)
    phia.tt[i+1] = phi1

# ?? need to copy
    ssol = r
    ssol = ttf.update_s(ssol,dt2,phi1,phi2)


    #edge = ssol[1]^r1.nodes[i+1][0]
    #r2.nodes.append(contract(edge))
    rtmp1 = np.tensordot(ssol,r1.nodes[i+1],axes=((1),(0)))
    r2.nodes.append(rtmp1)

#--------------------------------------------------------
### right most part
#  print("len of r2 =", len(r2.nodes))
#  for i in range(len(r2.nodes)):
#    print("shape of r2, i=", i, " is ", r2.nodes[i].shape)

  phi1 = phia.tt[nlen-1]
  phi2 = phia.tt[nlen]
  ksol = r2.nodes[nlen-1].copy()
  ksol = ttf.update_k(ksol,dt2,pall.nodes[nlen-1],phi1,phi2)
  r2.nodes[nlen-1] = ksol
#===================================================================
###### the second part of KSL, from right to left
  for i in range(nlen-1,0,-1):

#    print("update from right, i=", i)
    phi1 = phia.tt[i]
    phi2 = phia.tt[i+1]


    if (i==nlen-1):
      ksol = r2.nodes[i].copy()
    else:
      ksol = r3.nodes[nlen-1-i].copy()


    ksol = ttf.update_k(ksol,dt2,pall.nodes[i],phi1,phi2)
    #r, q = split_node_rq(ksol, left_edges=[ksol[0]],
    #       right_edges=[ksol[1], ksol[2]])
    r, q = split_rq(ksol)


    if (i==nlen-1):
      r3.nodes.append(q)
    else:
      r3.nodes[nlen-1-i] = q


    phi2 = ttf.phia_next(phi2,pall.nodes[i],q,q,1)
    phia.tt[i] = phi2


    ssol = r
    ssol = ttf.update_s(ssol,dt2,phi1,phi2)


    #edge = r2.nodes[i-1][2] ^ ssol[0]
    #r3.nodes.append(contract(edge))
    rtmp1 = np.tensordot(r2.nodes[i-1],ssol,axes=((2),(0)))
    r3.nodes.append(rtmp1)



  r3.nodes.reverse()
#  print("len of r3 =", len(r3.nodes))
#  for i in range(len(r3.nodes)):
#    print("shape of r3, i=", i, " is ", r3.nodes[i].shape)
  ### the left most matrix
  phi1 = phia.tt[0]
  phi2 = phia.tt[1]
  ksol = r3.nodes[0].copy()
#def update_k(yy,delta_t,mat1,phi1,phi2):
  ksol = ttf.update_k(ksol,dt2,pall.nodes[0],phi1,phi2)
  r3.nodes[0] = ksol

  ###### copy tensor to rin
  #rin = cp.deepcopy(r3)

  return r3

#=================================================================
# this do the rq split for a 3D tensor, based on (a,b,c) => (a,b*c)
# need to reshape q to 3D
def split_rq(xx):

#----------------------
  y1,m,y2 = np.shape(xx)
  yy = np.reshape(xx,(y1,y2*m),order='F')
  r, q = lg.rq(yy,mode='economic')
  q1 = np.reshape(q,(min(y1,y2*m),m,y2),order='F')

  return r, q1

#=================================================================
# this do the rq split for a 3D tensor, based on (a,b,c) => (a*b,c)
# need to reshape q to 3D
def split_qr(xx):

#----------------------
  y1,m,y2 = np.shape(xx)
  yy = np.reshape(xx,(y1*m,y2),order='F')
  q, r = lg.qr(yy,mode='economic')
  q1 = np.reshape(q, (y1, m, min(y1*m, y2)),order='F')

  return q1, r

