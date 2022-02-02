from numpy.linalg import svd
from os import mkdir
from os.path import normpath, exists, join


def check_and_create_dir(dirname):
    """Check if directory exists. If it does not exist, create it.
    
    Parameters
    ----------
    dirname : str
        Name of directory
    """
    if not exists(normpath(dirname)):
        mkdir(normpath(dirname))

        
def get_truncated_svd(A, k):
    """Get truncated SVD using a deterministic method.
    
    Parameters
    ----------
    A : ndarray of shape (m, n)
        Real matrix
    
    k : int 
        Rank
    """
    u, s, vh = svd(A, full_matrices=False)
    return u[:, :k], s[:k], vh[:k, :]