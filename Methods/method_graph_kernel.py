import ot
import numpy as np
import scipy as sp
from time import time
from scipy.sparse import csr_matrix
from Utils.load_data import load_candidate_inchi, inchi_to_graph


class GraphKernelEstimator:

    def __init__(self):
        self.n = None
        self.M = None
        self.K = None
        self.k = None
        self.Y = None
        self.Features = None
        self.alpha = 0.5
        self.h = None

    def train(self, K, Y, L):

        self.n = K.shape[0]
        self.M = np.linalg.inv(K + self.n * L * np.eye(self.n))
        self.K = K
        self.Y = Y

    def train_candidate_graphkernel(self, Y_c, Cs, Features):

        Cs_cand, Ls_cand, _, _ = Y_c

        Phi1, Phi2 = self.WLkernel([Cs, Features], Y_c)
        D = Phi1.dot(Phi2.T)
        norms_c = sp.sparse.linalg.norm(Phi2, axis=1) ** 2

        return D, norms_c

    def predict(self, K_tr_te, n_bary, Y_te, n_c_max=200):

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

            D, norms_c = self.train_candidate_graphkernel(Y_c, Cs, Features)
            score = 2 * D.T.dot(lambdas) - norms_c

            print(f'{i}: Computation time GK distances n_c x n_tr = {n_c} x {len(lambdas)}: {time() - t0} s')

            # predict
            idx_sorted = np.argsort(score)
            Inc_sorted = [In[idx] for idx in idx_sorted]
            G_pred = [Y_c[0][idx_sorted[-1]], Y_c[1][idx_sorted[-1]][:, :13]]
            G_te = [Y_te[0][i], Y_te[1][i][:, :13]]

            # Compute scores

            # compute fgw and gw
            fgw = self.fgw_distance(G_pred, G_te)
            gw = self.gw_distance(G_pred, G_te)
            print(f'{i}: FGW: {(fgw - self.alpha * gw) / (1 - self.alpha)} / {gw}')
            mean_fgw += np.array([(fgw - self.alpha * gw) / (1 - self.alpha), gw])

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

    def WL_feature(self, Y, h):

        m = len(Y[0])
        Features_list = []
        Dict_list = []
        level = 0

        # level 0
        Features = []
        for j in range(m):

            C, L = Y[0][j], Y[1][j][:, :13]
            n = L.shape[0]
            d = L.shape[1]
            B = L.dot(np.arange(1, d + 1))
            Features.append(B)
        Features_list.append(Features)
        Dict_list.append(None)
        level += 1

        # level > 0
        while level <= h:

            dico = dict()
            Features = []
            current_label = 1
            for j in range(m):

                C, L = Y[0][j], Y[1][j][:, :13]
                n = L.shape[0]
                B = np.array(Features_list[level - 1][j])

                F = []
                for i in range(n):
                    a = np.sort(B[np.arange(n)[(C[i] > 0)]])
                    a = ''.join(a.astype(int).astype(str))
                    if a not in dico.keys():
                        dico[a] = current_label
                        current_label += 1
                    F.append(dico[a])
                Features.append(F)

            Features_list.append(Features)
            Dict_list.append(dico)
            level += 1

        return Features_list, Dict_list

    def WLkernel(self, Y1, Y2):

        """
        Algorithm 2 in https: // www.jmlr.org / papers / volume12 / shervashidze11a / shervashidze11a.pdf
        """

        h = self.h
        n1 = len(Y1[0])
        n2 = len(Y2[0])
        d = 11
        Y_concat = [Y1[0] + Y2[0], Y1[1] + Y2[1]]
        Features_list, Dict_list = self.WL_feature(Y_concat, h)

        # Compute features
        Phi = []
        for i in range(n1 + n2):
            WL_feature = []

            # level 0
            Features = Features_list[0][i]
            WL_feature += [np.sum((Features == i + 1)) for i in range(d)]

            # level > 0
            for j in range(1, h + 1):
                Features = Features_list[j][i]
                dico = Dict_list[j]
                p = max(list(dico.values()))
                W = [np.sum((np.array(Features) == i + 1)) for i in range(p)]
                WL_feature += W

            Phi.append(WL_feature)

        Phi1 = csr_matrix(Phi[: n1])
        Phi2 = csr_matrix(Phi[n1:])
        print(f'Representation dimension: {Phi1.shape[1]}')

        return Phi1, Phi2

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
