import numpy as np
import EvolvingMatrix as EM

n_dim = 100
m_dim = 70
s_dim = 30
k_dim = 50

B = np.random.normal(size=(m_dim, n_dim))
E = np.random.normal(size=(s_dim, n_dim))

A = B.copy()

model = EM.EvolvingMatrix(B, k_dim=k_dim)
model.set_appendix_matrix(E)

for ii in range(s_dim):
  Uk, Sigmak, VHk = model.evolve_matrix()
  
  A = np.append(A, [E[ii,:]], axis=0)

  U, S, VH = np.linalg.svd(A)

  print("Error:")
  print(Sigmak-S[:k_dim])
  print()





A_full = np.load("../datasets/CISI/CISI.npy")
m_percent = 0.50

(m_dim_full, n_dim) = np.shape(A_full)
m_dim = int(np.ceil(m_dim_full*m_percent))
s_dim = int(np.floor(m_dim_full*(1-m_percent)))
k_dim = 50

B = A_full[:m_dim,:]
E = A_full[m_dim:,:]

A = B.copy()

model = EM.EvolvingMatrix(B, k_dim=k_dim)
model.set_appendix_matrix(E)

for ii in range(s_dim):
  Uk, Sigmak, VHk = model.evolve_matrix(step_dim=s_dim)

  A = np.append(A, [E[ii,:]], axis=0)

  U, S, VH = np.linalg.svd(A)

  print("Sigma(50) Residual Norm:")
  print(Sigmak[k_dim-1]-S[k_dim-1])
  print("Sigma(50) Relative Error:")
  print(np.abs(Sigmak[k_dim-1]-S[k_dim-1])/S[k_dim-1])
  """
  print("u(50) Error:")
  print(np.linalg.norm(Uk[k_dim-1]-U[k_dim-1]))
  print("v(50) Error:")
  print(np.linalg.norm(Vk[k_dim-1]-VH[k_dim-1]))
  print()
  """
