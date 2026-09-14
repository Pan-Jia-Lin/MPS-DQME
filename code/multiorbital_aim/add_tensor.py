######################################################
# Python code of the HEOM+MPS
# Author: Qiang Shi, Meng Xu, Xiaohan Dan, ICCAS
# Main reference: Q. Shi, Y. Xu, Y.-M. Yan, and M. Xu, 
# “Efficient propagation of the hierarchical equations 
# of motion using the matrix product state method”, 
# J. Chem. Phys. 148, 174102 (2018).
#
# add_tensor.py: add two tensors
#######################################################

import numpy as np
import sys
import params as pa
import copy as cp
import trun as tr
import print_tensor as pt
from tensornetwork import ncon
#from tensornetwork.network_components import Node

# note, coeff only applies to the first node

def add_tensor(r1, r2, coeff):
    rtmp = pa.Density()
    rtmp.nb = r1.nb
    nlen = len(r1.nodes)

    type_r1 = r1.nodes[0].dtype
    type_r2 = r2.nodes[0].dtype
    if type_r1 != type_r2:
        print("r1 and r2 have different types, not supported yet")
        sys.exit(1)

    # =============================================
    # 1. 头部节点 (Leftmost Node: i = 0)
    # 不假设 m1=m2=1，允许通用块对角拼接（块直和）
    m1, p1, n1 = r1.nodes[0].shape
    m2, p2, n2 = r2.nodes[0].shape

    if p1 != p2:
        sys.exit(f"Err in add_tensor: Node 0 physical dim mismatch {p1} vs {p2}")

    # 对于首节点，由于左边没有 Index，通常将左键拼在一起或者做块拼接
    # 如果确保 m1=m2=1，则按原逻辑横向拼接右键 n1+n2
    if m1 == 1 and m2 == 1:
        vtmp = np.zeros((1, p1, n1 + n2), dtype=type_r1)
        vtmp[0, :, 0:n1] = r1.nodes[0][0, :, :]
        vtmp[0, :, n1:n1+n2] = coeff * r2.nodes[0][0, :, :]
    else:
        # 通用块对角拼接
        vtmp = np.zeros((m1 + m2, p1, n1 + n2), dtype=type_r1)
        vtmp[0:m1, :, 0:n1] = r1.nodes[0]
        vtmp[m1:m1+m2, :, n1:n1+n2] = coeff * r2.nodes[0]

    rtmp.nodes.append(vtmp)

    # ================================================
    # 2. 中间节点 (Intermediate Nodes: i = 1 ~ nlen-2)
    for i in range(1, nlen - 1):
        m1, p1, n1 = r1.nodes[i].shape
        m2, p2, n2 = r2.nodes[i].shape

        if p1 != p2:
            sys.exit(f"Err in add_tensor: Node {i} physical dim mismatch {p1} vs {p2}")

        ii = m1 + m2
        jj = n1 + n2

        vtmp = np.zeros((ii, p1, jj), dtype=type_r1)
        vtmp[0:m1, :, 0:n1] = r1.nodes[i]
        vtmp[m1:ii, :, n1:jj] = r2.nodes[i]
        rtmp.nodes.append(vtmp)

    # =============================================
    # 3. 尾部节点 (Rightmost Node: i = nlen-1)
    m1, p1, n1 = r1.nodes[nlen-1].shape
    m2, p2, n2 = r2.nodes[nlen-1].shape

    if p1 != p2:
        sys.exit(f"Err in add_tensor: Node {nlen-1} physical dim mismatch {p1} vs {p2}")

    if n1 == 1 and n2 == 1:
        vtmp = np.zeros((m1 + m2, p1, 1), dtype=type_r1)
        vtmp[0:m1, :, 0] = r1.nodes[nlen-1][:, :, 0]
        vtmp[m1:m1+m2, :, 0] = r2.nodes[nlen-1][:, :, 0]
    else:
        vtmp = np.zeros((m1 + m2, p1, n1 + n2), dtype=type_r1)
        vtmp[0:m1, :, 0:n1] = r1.nodes[nlen-1]
        vtmp[m1:m1+m2, :, n1:n1+n2] = r2.nodes[nlen-1]

    rtmp.nodes.append(vtmp)

    # ============================================
    # 4. 截断/压缩 (Truncation)
    rtmp = tr.trun_tensor(rtmp)
    return rtmp


def add_init_rho(r1,r2):

  rtmp = pa.Density()
  rtmp.nb = r1.nb
  nlen = len(r1.nodes)

  type_r1 = r1.nodes[0].dtype
  type_r2 = r1.nodes[0].dtype
  if (type_r1 != type_r2):
    print("r1 and r2 has different type, not supported yet")
    sys.exit(1)
#=============================================
# the left matrix, we should have m1=m2=1 here
  m1 = r1.nodes[0].shape[0]
  n1 = r1.nodes[0].shape[2]
  m2 = r2.nodes[0].shape[0]
  n2 = r2.nodes[0].shape[2]
  
  jj = n1+n2
  # combine data
  vtmp = np.zeros((1,r1.nb[0],jj),dtype=type_r1)
  vtmp[0,:,0:n1] = r1.nodes[0][0,:,:]
  vtmp[0,:,n1:jj] = r2.nodes[0][0,:,:]
  # add to the nodes
  rtmp.nodes.append(vtmp)


#================================================
# add all the intermediate matrices
  for i in range(1,nlen-1):
    m1 = r1.nodes[i].shape[0]
    n1 = r1.nodes[i].shape[2]
    m2 = r2.nodes[i].shape[0]
    n2 = r2.nodes[i].shape[2]

    ii = m1+m2
    jj = n1+n2

    # combine data
    vtmp = np.zeros((ii,r1.nb[i],jj),dtype=type_r1)
    vtmp[0:m1,:,0:n1] = r1.nodes[i]
    vtmp[m1:ii,:,n1:jj] = r2.nodes[i]
    # add to the nodes
    rtmp.nodes.append(vtmp)


#=============================================
# the right matrix, we should have n1=n2=1 here

  m1 = r1.nodes[nlen-1].shape[0]
  n1 = r1.nodes[nlen-1].shape[2]
  m2 = r2.nodes[nlen-1].shape[0]
  n2 = r2.nodes[nlen-1].shape[2]
  
  ii = m1+m2
  # combine data
  vtmp = np.zeros((ii,r1.nb[nlen-1],1),dtype=type_r1)
  vtmp[0:m1,:,0] = r1.nodes[nlen-1][:,:,0]
  vtmp[m1:ii,:,0] = r2.nodes[nlen-1][:,:,0]


  # add to the nodes
  rtmp.nodes.append(vtmp)

  return rtmp


#===========================================
# this calculates the product of two tensors, with one one contraction
# also truncate the final result
def prod_tensor_mat(mat1,r1):

  rtmp = pa.Density()
  rtmp.nb = r1.nb
  nlen = len(r1.nodes)

  type_r1 = mat1.nodes[0].dtype
  type_r2 = r1.nodes[0].dtype

  if (type_r1 != type_r2):
    print("r1 and r2 has different type, not supported yet")
    sys.exit(1)


#----------------------------------------------------------
  for i in range(nlen):

    #ra1, rmid1, rmid2, ra2 = mat1.nodes[i].shape

    #vtmp1 = np.reshape(mat1.nodes[i].tensor,(ra1,pa.nb[i],pa.nb[i],ra2),order='F')
    vtmp1 = mat1.nodes[i]

    tensors  = [vtmp1, r1.nodes[i]]
    connects = [(-1,-2,1,-3),(-4,1,-5)]
    vtmp = ncon(tensors, connects)

    rtmp1 = vtmp.transpose((0,3,1,2,4))
    n1 = rtmp1.shape
    rtmp2 = np.reshape(rtmp1,(n1[0]*n1[1], n1[2], n1[3]*n1[4]),order='F')

    # add to the nodes
    rtmp.nodes.append(rtmp2)

#----------------------------------------------------------
# truncate the new tensor
  rtmp = tr.trun_tensor(rtmp)
  return rtmp

