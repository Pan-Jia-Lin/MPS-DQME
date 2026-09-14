######################################################
# Python code of the HEOM+MPS
# Author: Qiang Shi, Meng Xu, Xiaohan Dan, ICCAS
# Main reference: Q. Shi, Y. Xu, Y.-M. Yan, and M. Xu, 
# “Efficient propagation of the hierarchical equations 
# of motion using the matrix product state method”, 
# J. Chem. Phys. 148, 174102 (2018).
#
# print_tensor.py: print the tensor for debug
#######################################################

# prints a tensor, either with iflag="rank", or "data"
def print_tensor(rin,iflag="rank"):

  nlen = len(rin.nodes)
  print("========================================")
  print("tensor name")
  print("nlen=",nlen)


  for i in range(nlen):
    print("i=",i)
    print("shape of nodes", rin.nodes[i].tensor.shape)


  if (iflag == "rank"):
    print("========================================")
    return


  elif (iflag == "data"):
    print("data for the tensor")

    for i in range(nlen):
      print("----------- node:",i,"-----------------------------")
      for j in range(rin.nodes[i].shape[1]):
        print(rin.nodes[i].tensor[:,j,:])
      print("--------end node:",i,"-----------------------------")
    
