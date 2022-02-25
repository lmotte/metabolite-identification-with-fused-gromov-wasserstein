import numpy as np
import pandas as pd


def load_compounds_info(n, data_folder=None):
    """
    Load the compounds_info.txt file with the columns: (INCHIKEY, MOLECULAR_FORMULA, CV_FOLD)

    Parameters
    ------------
    - n : int
        Number of fingerprints to load
    - data_folder : str
        Path of folder containing the data
    """

    # if data_folder is None:
    #     data_folder = Path("Implementation/data/Donnees_metabolites/")
    file_to_open = data_folder / "compounds_info.txt"
    data = pd.read_csv(file_to_open, sep="\t", engine='python', nrows=n)
    return data


def center_gram_matrix(K, K_tr_x=None, K_tr_y=None, K_tr_tr=None):
    """
    Compute the centered gram matrix K_c

    Parameters
    ------------
    K : np.array
    Input not centered gram matrix between given sets X and Y
    K_tr_x : np.array
    Gram matrix between train and X
    K_tr_y : np.array
    Gram matrix between train and Y
    K_tr_tr : np.array
    Train gram matrix
    """

    if K_tr_x is None:
        K_tr_x = K
        K_tr_y = K
        K_tr_tr = K

    K_centered = K - (np.mean(K_tr_y, axis=0).reshape(1, -1) +
                      np.mean(K_tr_x, axis=0).reshape(-1, 1)) + np.mean(K_tr_tr) * np.ones(np.shape(K))

    return K_centered


def normalize_gram_matrix(K, K_tr_tr=None, K_te_te=None):
    """
    Compute the normalized gram matrix K_c

    Parameters
    ------------
    K : np.array
    Input not centered gram matrix between
    K_tr_tr : np.array
    Train gram matrix
    K_te_te : np.array
    Test gram matrix
    """

    if K_tr_tr is None:
        K_tr_tr = K
        K_te_te = K

    K_normalized = K / np.sqrt(np.multiply(np.diag(K_tr_tr).reshape(-1, 1), np.diag(K_te_te)))

    return K_normalized


def load_candidates(formula, n_max, random=False):

    # load candidates for a given molecular formula
    path_to_candidates = 'Implementation/Code/OEL/OEL/Data/Data_metabolites/Candidats/'

    try:
        with open(path_to_candidates + formula + '.csv') as f:
            candidates = f.readlines()
    except FileNotFoundError:
        return -1, -1
    # return error if too many candidates or zero candidate
    if len(candidates) == 0 or len(candidates) > n_max:
        return -1, -1

    # Parse load data to compute lists of inchikeys and fingerprints
    fingerprints = []  # will contain the fingerprints of the candidates
    inchikey_fingerprints = []  # will contain the inchikey of the candidates
    if random is False:
        for c in range(len(candidates)):
            inchikey_fingerprints.append(candidates[c].split('\t')[0])
            f = np.zeros(7593).astype(int)
            f[np.array(candidates[c].split('\t')[1:]).astype(int)] = 1
            fingerprints.append(f)

    else:
        random_idx = np.random.randint(len(candidates))
        inchikey_fingerprints.append(candidates[random_idx].split('\t')[0])
        f = np.zeros(7593).astype(int)
        f[np.array(candidates[random_idx].split('\t')[1:]).astype(int)] = 1
        fingerprints.append(f)

    cf = np.array(fingerprints)

    return cf, inchikey_fingerprints


def gaussian_tani_kernel(Y, Z, g):

    scalar_products = Y.dot(Z.T)
    Y_norms = np.linalg.norm(Y, axis=1) ** 2
    Z_norms = np.linalg.norm(Z, axis=1) ** 2
    nomi = scalar_products
    den = Y_norms.reshape(-1, 1) + Z_norms.reshape(1, -1) - scalar_products
    K_t = np.divide(nomi, den, where=den != 0)
    K_gt = np.exp(- g * 2 * (1 - K_t))

    return K_gt
