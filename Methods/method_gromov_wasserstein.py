import numpy as np
import ot
from Utils.load_data import load_candidate_inchi, inchi_to_graph
from time import time
from Utils.diffusion import diffuse


class FgwEstimator:

    def __init__(self):
        self.n = None
        self.M = None
        self.K = None
        self.k = None
        self.Y = None
        self.Features = None
        self.alpha = 0.5
        self.tau = None
        self.w = None
        self.ground_metric = None

    def train(self, K, Y, L):

        self.n = K.shape[0]
        self.M = np.linalg.inv(K + self.n * L * np.eye(self.n))
        self.K = K
        self.Y = Y

    def train_candidate_gwd(self, Y_c, Cs, Features):

        Cs_cand, Ls_cand, _, _ = Y_c
        n_c = len(Cs_cand)
        n_tr = len(Features)
        D = np.zeros((n_tr, n_c))

        for i in range(n_tr):
            for j in range(n_c):

                if self.ground_metric in ['onehot', 'diffuse']:
                    G1 = [Cs[i], Features[i][:, : 13]]
                    G2 = [Cs_cand[j], Ls_cand[j][:, : 13]]
                    d = self.fgw_distance(G1, G2)
                    D[i, j] = d
                elif self.ground_metric in ['fine']:
                    Features[i][:, 13:] = self.w * np.where(np.max(Features[i][:, 13:], axis=0) == 0,
                                                            Features[i][:, 13:],
                                                            Features[i][:, 13:]*1./np.max(Features[i][:, 13:], axis=0))
                    Ls_cand[j][:, 13:] = self.w * np.where(np.max(Ls_cand[j][:, 13:], axis=0) == 0, Ls_cand[j][:, 13:],
                                                           Ls_cand[j][:, 13:]*1./np.max(Ls_cand[j][:, 13:], axis=0))
                    G1 = [Cs[i], Features[i]]
                    G2 = [Cs_cand[j], Ls_cand[j]]
                    d = self.fgw_distance(G1, G2)
                    D[i, j] = d

        # uncomment to check ot convergence
        # a = d[1]['loss']
        # gw = self.gw_distance(G1, G2)
        # print(f'ot convergence: {a}')
        # print(f'fgw / gw: {d[0]} / {self.alpha *  gw}')
        # except:
        #     print(f'n_tr x n_c : {n_tr, n_c}')

        return D

    def predict(self, K_tr_te, n_bary, Y_te, n_c_max=200):

        n_te = K_tr_te.shape[1]
        A = self.M.dot(K_tr_te)

        # Compute n_te barycenters with n_bary points max for each barycenter
        mean_topk = np.array([0., 0., 0.])
        mean_fgw = np.array([0., 0.])
        n_pred = 0
        error_type = []

        for i in range(n_te):

            # predict weights and take greatest weight graphs
            lambdas = A[:, i]
            idxs = np.argsort(lambdas)[-n_bary:]
            lambdas = [A[j, i] for j in idxs]
            Cs = [self.Y[0][idx]for idx in idxs]
            Features = [self.Y[1][idx] for idx in idxs]

            # load candidate
            try:
                In = load_candidate_inchi(Y_te[3][i])
            except FileNotFoundError:
                print('FILE NOT FOUND')
                error_type.append('filenotfound')
                continue

            if In == -1:
                error_type.append('In')
                continue
            try:
                Y_c = [inchi_to_graph(inc) for inc in In]
                Y_c = [[g[0] for g in Y_c], [g[1] for g in Y_c], [], In]
            except TypeError:
                error_type.append('type')
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

            if self.ground_metric in ['diffuse']:
                Y_c_d = diffuse(Y_c, self.tau)
            elif self.ground_metric in ['onehot', 'fine']:
                Y_c_d = Y_c
            D = self.train_candidate_gwd(Y_c_d, Cs, Features)
            score = - D.T.dot(lambdas)

            print(f'{i}: Computation time GW distances n_c x n_tr = {n_c} x {len(lambdas)}: {time() - t0} s')

            # predict
            idx_sorted = np.argsort(score)
            Inc_sorted = [In[idx] for idx in idx_sorted]
            G_pred = [Y_c[0][idx_sorted[-1]], Y_c[1][idx_sorted[-1]][:, :13]]
            G_te = [Y_te[0][i], Y_te[1][i][:, :13]]

            # Compute scores

            # compute fgw and gw
            fgw = self.fgw_distance(G_pred, G_te)
            gw = self.gw_distance(G_pred, G_te)
            # print(f'{i}: FGW: {(fgw - self.alpha * gw) / (1 - self.alpha)} / {gw}')
            print(f'{i}: FGW: {(fgw) / (1 - self.alpha)} / {gw}')
            mean_fgw += np.array([(fgw) / (1 - self.alpha), gw])
            # mean_fgw += np.array([(fgw - self.alpha * gw) / (1 - self.alpha), gw])

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
