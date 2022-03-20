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


# Load validation scores and the associated hyperparameters grids
n_tr, n_val = 3000, 600
n_bary = 5
n_c_max = 1e6
try:
    path = "Results/"
    Grid = np.load(path + method + '_' + f'{n_tr}_{n_val}_{n_bary}_{n_c_max}_Grid.npy',)
    Sfgw = np.load(path + method + '_' + f'{n_tr}_{n_val}_{n_bary}_{n_c_max}_Sfgw.npy')
    Stopk = np.load(path + method + '_' + f'{n_tr}_{n_val}_{n_bary}_{n_c_max}_Stopk.npy')
except FileNotFoundError:
    path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/Results/"
    Grid = np.load(path + method + '_' + f'{n_tr}_{n_val}_{n_bary}_{n_c_max}_Grid.npy')
    Sfgw = np.load(path + method + '_' + f'{n_tr}_{n_val}_{n_bary}_{n_c_max}_Sfgw.npy')
    Stopk = np.load(path + method + '_' + f'{n_tr}_{n_val}_{n_bary}_{n_c_max}_Stopk.npy')

# Select the hyperparameters with the best validation scores
R = Stopk
G = Grid
bestvaltop1 = np.max(R[:, :, 0])
bestvaltop10 = np.max(R[:, :, 1])
bestidxtop1 = np.argmax(R[:, :, 0])
bestidxtop10 = np.argmax(R[:, :, 1])

bestidxtop1_un = np.unravel_index(np.argmax(R[:, :, 0]), R[:, :, 0].shape)
bestidxtop10_un = np.unravel_index(np.argmax(R[:, :, 1]), R[:, :, 1].shape)

idx = bestidxtop1_un
bestparam1 = G[idx[0], idx[1], :]
idx = bestidxtop10_un
bestparamtop10 = G[idx[0], idx[1], :]

L, param = bestparam1

# Train and Compute test score
n_tr, n_val = 4145, 1145  # 4145 - 1145 = 3000 train / 1145 test
n_c_max = 1e6  # do not consider test points with more than n_c_max candidates

if method == 'finger':
    exp = exp_finger
elif method == 'gw_onehot':
    exp = exp_gw_onehot
elif method == 'gw_fine':
    exp = exp_gw_fine
elif method == 'gw_diffuse':
    exp = exp_gw_diffuse
elif method == 'gk':
    exp = exp_gk
    param = int(param)
else:
    exp, param = None, None

t0 = time()
R = exp(n_tr, n_val, L, param, n_bary, n_c_max)
fgw, topk, n_train, n_pred = R
top1, top10, top20 = topk
print(f'Test time: {time() - t0}')

try:
    path = "Results/"
    np.save(path + method + '_' + f'{n_tr}_{n_val}_{n_bary}_{n_c_max}_test.npy', R)
except FileNotFoundError:
    path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/Results/"
    np.save(path + method + '_' + f'{n_tr}_{n_val}_{n_bary}_{n_c_max}_test.npy', R)
