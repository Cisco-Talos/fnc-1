import numpy as np
import scipy as sp
import pandas as pd

def count_feature(X, tbl_lst = None, min_cnt = 1):
    X_lst = [pd.Series(X[:, i]) for i in range(X.shape[1])]
    if tbl_lst is None:
        tbl_lst = [x.value_counts() for x in X_lst]
        if min_cnt > 1:
            tbl_lst = [s[s >= min_cnt] for s in tbl_lst]
    X = sp.column_stack([x.map(tbl).values for x, tbl in zip(X_lst, tbl_lst)])
    # NA(unseen values) to 0
    return np.nan_to_num(X), tbl_lst

# mat: A sparse matrix
def remove_duplicate_cols(mat):
    if not isinstance(mat, sp.sparse.coo_matrix):
        mat = mat.tocoo()
    row = mat.row
    col = mat.col
    data = mat.data
    crd = pd.DataFrame({'row':row, 'col':col, 'data':data}, columns = ['col', 'row', 'data'])
    col_rd = crd.groupby('col').apply(lambda x: str(np.array(x)[:,1:]))
    dup = col_rd.duplicated()
    return mat.tocsc()[:, col_rd.index.values[dup.values == False]]
    
def RImatrix(p, m, k, rm_dup_cols = False, seed = None):
    """ USAGE:
    Argument
      p: # of original varables
      m: The length of index vector
      k: # of 1s == # of -1s
    Rerurn value
      sparce.coo_matrix, shape:(p, s)
      If rm_dup_cols == False s == m
      else s <= m
    """
    if seed is not None: np.random.seed(seed)
    popu = range(m)
    row = np.repeat(range(p), 2 * k)
    col = np.array([np.random.choice(popu, 2 * k, replace = False) for i in range(p)]).reshape((p * k * 2,))
    data = np.tile(np.repeat([1, -1], k), p)
    mat = sp.sparse.coo_matrix((data, (row, col)), shape = (p, m), dtype = sp.int8)
    if rm_dup_cols:
        mat = remove_duplicate_cols(mat)
    return mat

# Random Indexing
def RI(X, m, k = 1, normalize = True, seed = None, returnR = False):
    R = RImatrix(X.shape[1], m, k, rm_dup_cols = True, seed = seed)
    Mat = X * R
    if normalize:
        Mat = pp.normalize(Mat, norm = 'l2')
    if returnR:
        return Mat, R
    else:
        return Mat

# Return a sparse matrix whose column has k_min to k_max 1s
def col_k_ones_matrix(p, m, k = None, k_min = 1, k_max = 1, seed = None, rm_dup_cols = True):
    if k is not None:
        k_min = k_max = k
    if seed is not None: np.random.seed(seed)
    k_col = np.random.choice(range(k_min, k_max + 1), m)
    col = np.repeat(range(m), k_col)
    popu = np.arange(p)
    l = [np.random.choice(popu, k_col[i], replace = False).tolist() for i in range(m)]
    row = sum(l, [])
    data = np.ones(k_col.sum())
    mat = sp.sparse.coo_matrix((data, (row, col)), shape = (p, m), dtype = np.float32)
    if rm_dup_cols:
        mat = remove_duplicate_cols(mat)
    return mat
