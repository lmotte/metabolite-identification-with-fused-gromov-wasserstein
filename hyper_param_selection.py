try:
    import numpy as np
    from time import time
    from train_val import exp_finger, exp_gk, exp_gw_onehot, exp_gw_fine, exp_gw_diffuse
    from multiprocessing import Pool
    cluster = False
    method = 'finger'
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, '/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein')
    import numpy as np
    from time import time
    from train_val import exp_finger, exp_gk, exp_gw_onehot, exp_gw_fine, exp_gw_diffuse
    from multiprocessing import Pool
    cluster = True
    method = sys.argv[1]


# Selection of the hyperparameters by taking the ones with the best validation scores

n_tr, n_val = 3000, 600  # 3000 - 600 = 2400 train / 600 val
n_c_max = 1e6  # do not consider test points with more than n_c_max candidates


# Define the grids of hyper-parameters
L_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
g_grid = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
w_grid = [0.01, 0.1, 0.5, 1, 3, 5, 10]
tau_grid = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
h_grid = [3, 5, 10]
n_bary = 5

if method == 'finger':
    param_grid = g_grid
    exp = exp_finger
elif method == 'gw_onehot':
    param_grid = [None]
    exp = exp_gw_onehot
elif method == 'gw_fine':
    param_grid = w_grid
    exp = exp_gw_fine
elif method == 'gw_diffuse':
    param_grid = tau_grid
    exp = exp_gw_diffuse
elif method == 'gk':
    param_grid = h_grid
    exp = exp_gk
else:
    exp, param_grid = None, None

Grid = - np.ones((len(L_grid), len(param_grid), 2))
Sfgw = - np.ones((len(L_grid), len(param_grid)))
Stopk = - np.ones((len(L_grid), len(param_grid), 3))

Larg = []
for i, L in enumerate(L_grid):
    for j, param in enumerate(param_grid):
        Larg.append((n_tr, n_val, L, param, n_bary, n_c_max))
        Grid[i, j, 0] = L
        Grid[i, j, 1] = param

n_pool = len(Larg)

t0 = time()
if __name__ == '__main__':
    with Pool(n_pool) as p:
        R = p.starmap(exp, Larg)

for i, L in enumerate(L_grid):
    for j, param in enumerate(param_grid):
        fgw, topk, n_train, n_pred = R[i * len(param_grid) + j]
        Sfgw[i, j] = fgw
        Stopk[i, j, 0] = topk[0]
        Stopk[i, j, 1] = topk[1]
        Stopk[i, j, 2] = topk[2]

print(f'selection time (with multiprocessing): {time() - t0}')

try:
    path = "Results/"
    np.save(path + method + '_' + f'{n_tr}_{n_val}_{n_bary}_{n_c_max}_Grid.npy', Grid)
    np.save(path + method + '_' + f'{n_tr}_{n_val}_{n_bary}_{n_c_max}_Sfgw.npy', Sfgw)
    np.save(path + method + '_' + f'{n_tr}_{n_val}_{n_bary}_{n_c_max}_Stopk.npy', Stopk)
except FileNotFoundError:
    path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/Results/"
    np.save(path + method + '_' + f'{n_tr}_{n_val}_{n_bary}_{n_c_max}_Grid.npy', Grid)
    np.save(path + method + '_' + f'{n_tr}_{n_val}_{n_bary}_{n_c_max}_Sfgw.npy', Sfgw)
    np.save(path + method + '_' + f'{n_tr}_{n_val}_{n_bary}_{n_c_max}_Stopk.npy', Stopk)
