import numpy as np
import ot
from Utils.load_data import load_candidate_finger, inchi_to_graph
from Utils.metabolites_utils import gaussian_tani_kernel
from time import time


class IOKREstimator:

    def __init__(self):
        self.n = None
        self.M = None
        self.K = None
        self.k = None
        self.Y = None
        self.Features = None
        self.alpha = 0.5
        self.tau = None
        self.g = None

    def train(self, K, Y, L, g):

        self.n = K.shape[0]
        self.M = np.linalg.inv(K + self.n * L * np.eye(self.n))
        self.K = K
        self.Y = Y
        self.g = g

    def train_candidate_kernel(self, F_c, F_tr):

        # scalar_products = F_tr.toarray().dot(F_c.toarray().T)
        # Y_norms = np.linalg.norm(F_tr.toarray(), axis=1) ** 2
        # Z_norms = np.linalg.norm(F_c.toarray(), axis=1) ** 2
        # nomi = scalar_products
        # K_tr_c = Y_norms.reshape(-1, 1) + Z_norms.reshape(1, -1) - scalar_products
        # K_tr_c = - K_tr_c

        K_tr_c = gaussian_tani_kernel(F_tr.toarray(), F_c.toarray(), g=self.g)

        return K_tr_c

    def predict(self, K_tr_te, Y_te, n_c_max=200):

        n_te = K_tr_te.shape[1]
        A = self.M.dot(K_tr_te)

        # Compute n_te barycenters with n_bary points max for each barycenter
        Y_pred = []
        mean_topk = np.array([0., 0., 0.])
        mean_fgw = np.array([0., 0.])
        n_pred = 0
        error_type = []

        for i in range(n_te):

            # predict weights and take greatest weight graphs
            lambdas = A[:, i]
            idxs = np.argsort(lambdas)
            lambdas = [A[j, i] for j in idxs]

            # load candidate
            try:
                In, Fingers = load_candidate_finger(Y_te[3][i])
            except FileNotFoundError:
                print('FILE NOT FOUND')
                error_type.append('filenotfound')
                continue

            if In == -1:
                error_type.append('In')
                continue

            # compute score
            t0 = time()
            n_c = len(In)
            if n_c > n_c_max:
                error_type.append('too big')
                continue
            if n_c < 1:
                error_type.append('too small')
                continue

            Fingers_selected_train = self.Y[0][idxs]
            K_tr_c = self.train_candidate_kernel(Fingers, Fingers_selected_train)
            score = K_tr_c.T.dot(lambdas)

            print(f'{i}: Computation time gaussian-tani kernel n_c x n_tr = {n_c} x {len(lambdas)}: {time() - t0} s')

            # predict
            idx_sorted = np.argsort(score)
            Inc_sorted = [In[idx] for idx in idx_sorted]

            y_pred = inchi_to_graph(Inc_sorted[-1])
            G_pred = [y_pred[0], y_pred[1][:, :13]]
            G_te = [Y_te[0][i], Y_te[1][i][:, :13]]

            # Compute scores

            # compute fgw and gw
            fgw = self.fgw_distance(G_pred, G_te)
            gw = self.gw_distance(G_pred, G_te)
            print(f'{i}: FGW: {(fgw) / (1 - self.alpha)} / {gw}')
            # print(f'{i}: FGW: {(fgw - self.alpha * gw) / (1 - self.alpha)} / {gw}')
            mean_fgw += np.array([(fgw) / (1 - self.alpha), gw])

            # compute top-k
            inchi_true = Y_te[2][i]
            topk = [inchi_true in Inc_sorted[-1:], inchi_true in Inc_sorted[-10:], inchi_true in Inc_sorted[-20:]]
            topk = np.array(topk).astype(int)
            mean_topk += topk
            n_pred += 1
            print(f'{i}: n_candidate x n_tr = {len(Inc_sorted)} x {len(lambdas)},'
                  f' mean topk = {mean_topk / n_pred * 100}', flush=True)

        mean_fgw = mean_fgw / n_pred
        mean_topk = mean_topk / n_pred * 100
        print(f'n prediction: {n_pred}', flush=True)

        print(error_type)

        return mean_fgw, mean_topk, n_pred

    def post_process(self, C_bary, F_bary, N_edges):

        # most probable atom predicted
        F = np.zeros(F_bary.shape)
        for i in range(F_bary.shape[0]):
            u = np.zeros(F_bary.shape[1])
            a = np.argmax(F_bary[i])
            u[a] = 1
            F[i] = u

        # keep N largest edges
        ind = np.unravel_index(np.argsort(C_bary, axis=None), C_bary.shape)
        ind_max = (ind[0][-2 * N_edges:], ind[1][-2 * N_edges:])
        C = np.zeros(C_bary.shape)
        C[ind_max] = 1

        return C, F

    def fgw_distance(self, G1, G2):

        n1 = len(G1[1])
        n2 = len(G2[1])
        p1 = ot.unif(n1)
        p2 = ot.unif(n2)
        loss_fun = 'square_loss'
        Y_norms = np.linalg.norm(G1[1], axis=1) ** 2
        Z_norms = np.linalg.norm(G2[1], axis=1) ** 2
        scalar_products = G1[1].dot(G2[1].T)
        M = Y_norms.reshape(-1, 1) + Z_norms.reshape(1, -1) - 2 * scalar_products
        d = ot.gromov.fused_gromov_wasserstein2(M, G1[0], G2[0], p1, p2,
                                                loss_fun=loss_fun, alpha=self.alpha)

        return d

    def gw_distance(self, G1, G2):

        n1 = len(G1[1])
        n2 = len(G2[1])
        p1 = ot.unif(n1)
        p2 = ot.unif(n2)
        loss_fun = 'square_loss'
        d = ot.gromov.gromov_wasserstein2(G1[0], G2[0], p1, p2, loss_fun=loss_fun)

        return d
