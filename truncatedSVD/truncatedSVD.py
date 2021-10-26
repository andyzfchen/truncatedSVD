import numpy as np
import EvolvingMatrix as EM

n_dim = 100
k_dim = 50
s_dim = 50

B = np.random.normal(size=(k_dim, n_dim))
E = np.random.normal(size=(s_dim, n_dim))

A = B.copy()

model = EM.EvolvingMatrix(B, k_dim=k_dim)
model.set_appendix_matrix(E)

for ii in range(s_dim):
  Uk, Sigmak, VHk = model.evolve_matrix()
  
  A = np.append(A, [E[ii,:]], axis=0)

  U, S, VH = np.linalg.svd(A)

  #print(Sigmak)
  #print(S)
  print(Sigmak-S)
  #exit()


