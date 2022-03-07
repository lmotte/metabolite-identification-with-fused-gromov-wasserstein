try:
    import numpy as np
    from time import time
    from Utils.load_data import load_dataset_kernel_graph
    from Utils.metabolites_utils import center_gram_matrix, normalize_gram_matrix
    from Methods.method_gromov_wasserstein import FgwEstimator
    from Utils.diffusion import diffuse
    from multiprocessing import Pool
    cluster = False
    method = 'finger'
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, '/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein')
    import numpy as np
    from time import time
    from Utils.load_data import load_dataset_kernel_graph
    from Utils.metabolites_utils import center_gram_matrix, normalize_gram_matrix
    from Methods.method_gromov_wasserstein import FgwEstimator
    from Utils.diffusion import diffuse
    from multiprocessing import Pool
    cluster = True
    method = sys.argv[1]


# 1) Load data
n_tr = 3000
n_te = 1148
D_tr, D_te = load_dataset_kernel_graph(n_tr)
K_tr, Y_tr = D_tr
K_tr_te, K_te_te, Y_te = D_te

# 2) Input pre-processing
center, normalize = True, True
if center:
    K_tr_te = center_gram_matrix(K_tr_te, K_tr, K_tr_te, K_tr)
    K_tr = center_gram_matrix(K_tr)
if normalize:
    K_tr_te = normalize_gram_matrix(K_tr_te, K_tr, K_te_te)
    K_tr = normalize_gram_matrix(K_tr)

# 3) Train
clf = FgwEstimator()
clf.ground_metric = 'diffuse'
L = 1e-4  # kernel ridge regularization parameter
clf.tau = 0.6  # the bigger tau is the more the neighbor atoms have similar feature. This impact the FGW's ground metric.
Y_Tr = diffuse(Y_tr, clf.tau)
clf.train(K_tr, Y_tr, L)

# 4) Predict and compute the test scores
n_bary = 5  # Number of kept alpha_i(x) when predicting
n_c_max = 500   # Do not predict test input with more than n_c_max candidates
fgw, topk, n_pred = clf.predict(K_tr_te, n_bary=n_bary, Y_te=Y_te, n_c_max=n_c_max)

print(f'fgw: {fgw}, topk: {topk}, n_pred: {n_pred}')

