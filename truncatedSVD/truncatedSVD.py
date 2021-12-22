import numpy as np
import EvolvingMatrix as EM
import os

datasets = [ "CISI", "CRAN", "MED", "ML1M", "Reuters" ]
batch_splits = [ 10 ]
phis = [ [ 1, 5, 10 ] ]
evolution_methods = [ "zha-simon", "bcg" ]
r_values = [ 10 ]
m_percent = 0.10


# debug mode
datasets = [ "CISI" ]
batch_splits = [ 1 ]
phis = [ [ 1 ] ]
evolution_methods = [ "zha-simon", "bcg" ]
r_values = [ 50 ]
m_percent = 0.10



if not os.path.exists("../cache"):
  os.mkdir("../cache")

for dataset in datasets:
  print(f"Using {dataset} dataset.")

  for r_value in r_values:
    print(f"Using r value of {str(r_value)}.")

    for evolution_method in evolution_methods:
      if evolution_method == "zha-simon" and r_value != r_values[0]:
        continue

      print(f"Using the {evolution_method} evolution method.")

      if not os.path.exists("../cache/" + evolution_method):
        os.mkdir("../cache/" + evolution_method)

      for batch_split, phi in zip(batch_splits, phis):
        print(f"Performing truncated SVD on dataset {dataset} using batch_split = {str(batch_split)}.")

        # Create directory to save data for this batch number
        temp_dir = f"../cache/{evolution_method}/{dataset}_batch_split_{str(batch_split)}" 
        if not os.path.exists(temp_dir):
          os.mkdir(temp_dir)

        # Load entire dataset
        A_full = np.load(f"../datasets/{dataset}/{dataset}.npy")

        # Calculate row index to split data
        (m_dim_full, n_dim) = np.shape(A_full)
        m_dim = int(np.ceil(m_dim_full * m_percent))
        s_dim = int(np.floor(m_dim_full * (1 - m_percent)))
        
        # Set desired rank of truncated SVD
        # TODO: loop through different values of k (25,50,100)
        k_dim = 50

        # Split into initial and update matrices
        B = A_full[:m_dim, :]
        E = A_full[m_dim:, :]

        # Initialize EM object with initial and update matrices
        model = EM.EvolvingMatrix(B, k_dim=k_dim)
        model.set_appendix_matrix(E)

        r_str = ""

        # Update truncated SVD with each batch update
        for ii in range(batch_split):
          print(f"Batch{str(ii+1)} of {str(batch_split)}.")
          if evolution_method == "zha-simon":
            Uk, Sigmak, VHk = model.evolve_matrix_zha_simon(step_dim=int(np.ceil(s_dim/batch_split)))
            r_str = ""
          elif evolution_method == "bcg":
            Uk, Sigmak, VHk = model.evolve_matrix_deflated_bcg(step_dim=int(np.ceil(s_dim/batch_split)), r_dim=r_value)
            r_str = "_rval_"+str(r_value)

          # Save results if batch number specified
          if ii + 1 in phi:
            model.calculate_new_svd(evolution_method, dataset, batch_split, ii)
            relative_errors = model.get_relative_error()
            residual_norms = model.get_residual_norm()

            print("Singular value Relative Error at phi = "+str(ii+1)+":")
            print(relative_errors)
            np.save(f"{temp_dir}/relative_errors_phi_{str(ii+1)}{r_str}.npy", relative_errors)

            print("Last singular vector Residual Norm at phi = "+str(ii+1)+":")
            print(residual_norms)
            np.save(f"{temp_dir}/residual_norms_phi_{str(ii+1)}{r_str}.npy", residual_norms)

            print()


# if __name__ == "__main__":
#   import argparse

#   parser = argparse.ArgumentParser()
#   parser.add_argument("")
  
#   args = parser.parse_args()
  
