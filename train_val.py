try:
    from time import time
    from Methods.method_gromov_wasserstein import FgwEstimator
    from Methods.method_graph_kernel import GraphKernelEstimator
    from Methods.method_fingerprint import IOKREstimator
    from Utils.metabolites_utils import center_gram_matrix, normalize_gram_matrix
    from Utils.load_data import load_dataset_kernel_graph, load_dataset_kernel_finger
    from Utils.diffusion import diffuse
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, '/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein')
    from time import time
    from Methods.method_gromov_wasserstein import FgwEstimator
    from Methods.method_graph_kernel import GraphKernelEstimator
    from Methods.method_fingerprint import IOKREstimator
    # from method_gw_bary_before import FgwEstimatorBefore
    from Utils.metabolites_utils import center_gram_matrix, normalize_gram_matrix
    from Utils.load_data import load_dataset_kernel_graph, load_dataset_kernel_finger
    from Utils.diffusion import diffuse


def exp_gw_onehot(n_tr, n_val, L, unused, n_bary, n_c_max):

    # Load data
    t0 = time()
    D_tr, D_te = load_dataset_kernel_graph(n_tr - n_val)
    K, Y = D_tr
    K_tr_te, K_te_te, Y_te = D_te
    n = K_tr_te.shape[0]
    K_tr_te, K_te_te = K_tr_te[:, :n_val], K_te_te[:n_val, :n_val]
    Y_te = [Y_te[0][: n_val], Y_te[1][: n_val], Y_te[2][: n_val], Y_te[3][: n_val]]
    print(f'Load time: {time() - t0}', flush=True)

    # Input pre-processing
    t0 = time()
    center, normalize = True, True
    if center:
        K_tr_te = center_gram_matrix(K_tr_te, K, K_tr_te, K)
        K = center_gram_matrix(K)
    if normalize:
        K_tr_te = normalize_gram_matrix(K_tr_te, K, K_te_te)
        K = normalize_gram_matrix(K)
    print(f'Pre-processing time: {time() - t0}', flush=True)

    # Train
    t0 = time()
    clf = FgwEstimator()
    clf.ground_metric = 'onehot'
    clf.train(K, Y, L)
    print(f'Train time: {time() - t0}', flush=True)

    # Predict
    t0 = time()
    fgw, topk, n_pred = clf.predict(K_tr_te, n_bary=n_bary, Y_te=Y_te, n_c_max=n_c_max)
    print(f'Test time: {time() - t0}', flush=True)

    print(f'{(n_tr, n_val, L, None, n_bary, n_c_max)}, mean fgw : {fgw}, topk = {topk}', flush=True)

    return fgw[0], topk, n, n_pred


def exp_gw_fine(n_tr, n_val, L, w, n_bary, n_c_max):

    # Load data
    t0 = time()
    D_tr, D_te = load_dataset_kernel_graph(n_tr - n_val)
    K, Y = D_tr
    K_tr_te, K_te_te, Y_te = D_te
    n = K_tr_te.shape[0]
    K_tr_te, K_te_te = K_tr_te[:, :n_val], K_te_te[:n_val, :n_val]
    Y_te = [Y_te[0][: n_val], Y_te[1][: n_val], Y_te[2][: n_val], Y_te[3][: n_val]]
    print(f'Load time: {time() - t0}', flush=True)

    # Input pre-processing
    t0 = time()
    center, normalize = True, True
    if center:
        K_tr_te = center_gram_matrix(K_tr_te, K, K_tr_te, K)
        K = center_gram_matrix(K)
    if normalize:
        K_tr_te = normalize_gram_matrix(K_tr_te, K, K_te_te)
        K = normalize_gram_matrix(K)
    print(f'Pre-processing time: {time() - t0}', flush=True)

    # Train
    t0 = time()
    clf = FgwEstimator()
    clf.ground_metric = 'fine'
    clf.w = w
    clf.train(K, Y, L)
    print(f'Train time: {time() - t0}', flush=True)

    # Predict
    t0 = time()
    fgw, topk, n_pred = clf.predict(K_tr_te, n_bary=n_bary, Y_te=Y_te, n_c_max=n_c_max)
    print(f'Test time: {time() - t0}', flush=True)

    print(f'{(n_tr, n_val, L, w, n_bary, n_c_max)}, mean fgw : {fgw}, topk = {topk}', flush=True)

    return fgw[0], topk, n, n_pred


def exp_gw_diffuse(n_tr, n_val, L, tau, n_bary, n_c_max):

    # Load data
    t0 = time()
    D_tr, D_te = load_dataset_kernel_graph(n_tr - n_val)
    K, Y = D_tr
    K_tr_te, K_te_te, Y_te = D_te
    n = K_tr_te.shape[0]
    K_tr_te, K_te_te = K_tr_te[:, :n_val], K_te_te[:n_val, :n_val]
    Y_te = [Y_te[0][: n_val], Y_te[1][: n_val], Y_te[2][: n_val], Y_te[3][: n_val]]
    print(f'Load time: {time() - t0}', flush=True)

    # Input pre-processing
    t0 = time()
    center, normalize = True, True
    if center:
        K_tr_te = center_gram_matrix(K_tr_te, K, K_tr_te, K)
        K = center_gram_matrix(K)
    if normalize:
        K_tr_te = normalize_gram_matrix(K_tr_te, K, K_te_te)
        K = normalize_gram_matrix(K)
    print(f'Pre-processing time: {time() - t0}', flush=True)

    # Train
    t0 = time()
    clf = FgwEstimator()
    clf.ground_metric = 'diffuse'
    clf.tau = tau
    Y = diffuse(Y, clf.tau)
    clf.train(K, Y, L)
    print(f'Train time: {time() - t0}', flush=True)

    # Predict
    t0 = time()
    fgw, topk, n_pred = clf.predict(K_tr_te, n_bary=n_bary, Y_te=Y_te, n_c_max=n_c_max)
    print(f'Test time: {time() - t0}', flush=True)

    print(f'{(n_tr, n_val, L, tau, n_bary, n_c_max)}, mean fgw : {fgw}, topk = {topk}', flush=True)

    return fgw[0], topk, n, n_pred


def exp_gk(n_tr, n_val, L, h, n_bary, n_c_max):

    # Load data
    t0 = time()
    D_tr, D_te = load_dataset_kernel_graph(n_tr - n_val)
    K, Y = D_tr
    K_tr_te, K_te_te, Y_te = D_te
    K_tr_te, K_te_te = K_tr_te[:, :n_val], K_te_te[:n_val, :n_val]
    Y_te = [Y_te[0][: n_val], Y_te[1][: n_val], Y_te[2][: n_val], Y_te[3][: n_val]]
    n = K_tr_te.shape[0]
    print(f'Load time: {time() - t0}', flush=True)

    # Input pre-processing
    t0 = time()
    center, normalize = True, True
    if center:
        K_tr_te = center_gram_matrix(K_tr_te, K, K_tr_te, K)
        K = center_gram_matrix(K)
    if normalize:
        K_tr_te = normalize_gram_matrix(K_tr_te, K, K_te_te)
        K = normalize_gram_matrix(K)
    print(f'Pre-processing time: {time() - t0}', flush=True)

    # Train
    t0 = time()
    clf = GraphKernelEstimator()
    clf.train(K, Y, L)
    print(f'Train time: {time() - t0}', flush=True)

    # Predict
    t0 = time()
    clf.h = h
    fgw, topk, n_pred = clf.predict(K_tr_te, n_bary=n_bary, Y_te=Y_te, n_c_max=n_c_max)
    print(f'Test time: {time() - t0}', flush=True)

    print(f'{(n_tr, n_val, L, h, n_bary, n_c_max)}, mean fgw : {fgw}, topk = {topk}', flush=True)

    return fgw[0], topk, n, n_pred


def exp_finger(n_tr, n_val, L, g, n_bary, n_c_max):

    # Load data
    t0 = time()
    D_tr, D_te = load_dataset_kernel_finger(n_tr-n_val)
    K, Y = D_tr
    K_tr_te, K_te_te, Y_te = D_te
    K_tr_te, K_te_te = K_tr_te[:, :n_val], K_te_te[:n_val, :n_val]
    Y_te = [Y_te[0][: n_val], Y_te[1][: n_val], Y_te[2][: n_val], Y_te[3][: n_val]]
    n = K_tr_te.shape[0]
    print(f'Load time: {time() - t0}', flush=True)

    # Input pre-processing
    t0 = time()
    center, normalize = True, True
    if center:
        K_tr_te = center_gram_matrix(K_tr_te, K, K_tr_te, K)
        K = center_gram_matrix(K)
    if normalize:
        K_tr_te = normalize_gram_matrix(K_tr_te, K, K_te_te)
        K = normalize_gram_matrix(K)

    print(f'Pre-processing time: {time() - t0}', flush=True)

    # Train
    t0 = time()
    clf = IOKREstimator()
    clf.train(K, Y, L, g)
    print(f'Train time: {time() - t0}', flush=True)

    # Predict
    t0 = time()
    n_bary = n_tr
    fgw, topk, n_pred = clf.predict(K_tr_te, n_bary=n_bary, Y_te=Y_te, n_c_max=n_c_max)
    print(f'Test time: {time() - t0}', flush=True)

    print(f'{(n_tr, n_val, L, g, n_bary, n_c_max)}, mean fgw : {fgw}, topk = {topk}', flush=True)

    return fgw[0], topk, n, n_pred
