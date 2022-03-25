import numpy as np
from os import listdir
import rdkit
from rdkit import Chem
import pickle as pk
from rdkit.Chem import AllChem
import scipy.io


def build_spec():
    """save spec as numpy arrays
    """

    try:
        path = "Data/"
        L = listdir(path + 'data_new/spectra-2')
    except (FileNotFoundError, OSError):
        path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/Data/"
        L = listdir(path + 'data_new/spectra-2')

    Spectra = []

    for filename in L:

        try:
            try:
                with open('Data/data_new/spectra-2/' + filename) as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                print(f"FILENAME = {filename}")
        except (FileNotFoundError, OSError):
            path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/Data/"
            try:
                with open(path + 'data_new/spectra-2/' + filename) as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                print(f"FILENAME = {filename}")

        I_start = []
        for i, line in enumerate(lines):
            if line[0:10] == '>collision':
                I_start.append(i + 1)

        Spec_x = [float(line.split(' ')[0]) for line in lines[I_start[0]:]]
        Spec_y = [float(line.split(' ')[1][:-1]) for line in lines[I_start[0]:]]
        Spectrum = [Spec_x, Spec_y]
        Spectra.append(Spectrum)

    Spectra = np.array(Spectra)

    try:
        np.save('Data/data_new/spectra.npy', Spectra, allow_pickle=True)
    except (FileNotFoundError, OSError):
        path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/Data/"
        np.save(path + 'data_new/spectra.npy', Spectra, allow_pickle=True)


def load_spec_inchi():
    """load couples (spectrum, inchi)
    """

    L = listdir('Data/data_new/spectra-2')
    Inc = []
    Spectra = []

    for filename in L:

        try:
            try:
                with open('Data/data_new/spectra-2/' + filename) as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                print(f"FILENAME = {filename}")
        except (FileNotFoundError, OSError):
            path = "/tsi/clusterhome/lmotte/Implementation/" \
                   "metabolite-identification-with-fused-gromov-wasserstein/Data/"
            try:
                with open(path + 'data_new/spectra-2/' + filename) as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                print(f"FILENAME = {filename}")

        Inc.append(lines[4][7:-1])

    try:
        Spec_load = np.load('Data/data_new/spectra.npy', allow_pickle=True)
    except (FileNotFoundError, OSError):
        path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/Data/"
        Spec_load = np.load(path + 'data_new/spectra.npy', allow_pickle=True)

    for s in Spec_load:
        Spectra.append(np.array([s[0], s[1]]))

    return Spectra, Inc


def inchi_to_graph(inc):

    m = rdkit.Chem.inchi.MolFromInchi(inc)

    # 1) get molecule graph adjacency
    if m is not None:
        C = Chem.GetAdjacencyMatrix(m)
    else:
        return -1

    # 2) compute atom representations
    Atoms = m.GetAtoms()
    F = []
    U = {'C': 0, 'N': 1, 'O': 2, 'Cl': 3, 'S': 4, 'F': 5, 'P': 6, 'I': 7, 'Br': 8, 'Se': 9, 'Si': 10, 'H': 11, 'B': 12}

    I = np.eye(len(U))

    for atom in Atoms:

        rep = list(I[U[atom.GetSymbol()]])

        Nei = [a.GetSymbol() for a in atom.GetNeighbors()]
        n_attached_H = Nei.count('H')
        rep.append(n_attached_H)

        n_heavy_neigh = len(Nei) - n_attached_H
        rep.append(n_heavy_neigh)

        rep.append(atom.GetFormalCharge())

        rep.append(int(atom.IsInRing()))

        rep.append(int(atom.GetIsAromatic()))

        F.append(rep)

    # compute atoms positions
    AllChem.Compute2DCoords(m)
    P = []
    for c in m.GetConformers():
        P.append(c.GetPositions())

    return C, np.array(F), P


def build_spectrum_graph_dataset():

    # Order output data according to data_GNPS.tct in order to align input kernel and output graph
    try:
        mat = scipy.io.loadmat('Data/data_new/data_GNPS.mat')
    except (FileNotFoundError, OSError):
        path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/Data/"
        mat = scipy.io.loadmat(path + 'data_new/data_GNPS.mat')

    In = [inch[0][0] for inch in mat['inchi']]
    Mf = [mf[0][0] for mf in mat['mf']]

    Cs, Fs, Ps = [], [], []

    for inc in In:
        C, F, P = inchi_to_graph(inc)
        Cs.append(C)
        Fs.append(F)
        # Ps.append(P)

    # save
    # Y = [Cs, Fs, Ps]
    Y = [Cs, Fs, In, Mf]

    try:
        with open('Data/data_new/output_graphs.pickle', 'wb') as handle:
            pk.dump(Y, handle)
    except (FileNotFoundError, OSError):
        path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/Data/"
        with open(path + 'data_new/output_graphs.pickle', 'wb') as handle:
            pk.dump(Y, handle)


# build_spectrum_graph_dataset()


def load_dataset_graph(n):

    try:
        with open('Data/data_new/output_graphs.pickle', 'rb') as handle:
            Y = pk.load(handle)
    except (FileNotFoundError, OSError):
        path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/"

        with open(path + 'Data/data_new/output_graphs.pickle', 'rb') as handle:
            Y = pk.load(handle)

    try:
        Spectra = np.load('Data/data_new/spec_bin.npy')
    except FileNotFoundError:
        path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/"
        Spectra = np.load(path + 'Data/data_new/spec_bin.npy')

    # Divide train/test
    Spectra_tr = Spectra[:n]
    Spectra_te = Spectra[n:]
    Cs, Fs, In, Mf = Y
    Cs_tr = Cs[: n]
    Cs_te = Cs[n:]
    Fs_tr = Fs[: n]
    Fs_te = Fs[n:]
    In_tr = In[: n]
    In_te = In[n:]
    Mf_tr = Mf[: n]
    Mf_te = Mf[n:]
    Y_tr = [Cs_tr, Fs_tr, In_tr, Mf_tr]
    Y_te = [Cs_te, Fs_te, In_te, Mf_te]

    return [Spectra_tr, Y_tr], [Spectra_te, Y_te]


def spectra_to_bins(spec):

    #  define bins from 20 to 2500
    n_bins = 2000
    m = 20
    M = 3000
    bin_size = (M - m) / n_bins
    spec_repr = np.zeros(n_bins)
    for i, x in enumerate(spec[0]):
        bin_idx = int((x - m) / bin_size)
        spec_repr[bin_idx] += spec[1][i]

    return spec_repr


def transform_all_spectrum():

    D_tr, D_te = load_dataset_graph(7000)
    Spectra, Y = D_tr

    S_new = []
    for i, spec in enumerate(Spectra):
        spec_r = spectra_to_bins(spec)
        S_new.append(list(spec_r))

    np.save('Data/data_new/spec_bin.npy', np.array(S_new))


def load_dataset_kernel_graph(n):

    try:
        K = np.loadtxt('Data/data_new/input_kernels/PPKr.txt')
    except (FileNotFoundError, OSError):

        path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/"
        K = np.loadtxt(path + 'Data/data_new/input_kernels/PPKr.txt')

    try:
        with open('Data/data_new/output_graphs.pickle', 'rb') as handle:
            Y = pk.load(handle)
    except (FileNotFoundError, OSError):

        path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/"

        with open(path + 'Data/data_new/output_graphs.pickle', 'rb') as handle:
            Y = pk.load(handle)

    # Divide train/test
    K_tr = K[:n, :n]
    K_tr_te = K[:n, n:]
    K_te_te = K[n:, n:]
    Cs, Fs, In, Mf = Y
    Cs_tr = Cs[: n]
    Cs_te = Cs[n:]
    Fs_tr = Fs[: n]
    Fs_te = Fs[n:]
    In_tr = In[: n]
    In_te = In[n:]
    Mf_tr = Mf[: n]
    Mf_te = Mf[n:]
    Y_tr = [Cs_tr, Fs_tr, In_tr, Mf_tr]
    Y_te = [Cs_te, Fs_te, In_te, Mf_te]

    return [K_tr, Y_tr], [K_tr_te, K_te_te, Y_te]


def load_candidate_inchi(mf):

    try:
        mat = scipy.io.loadmat(f'Data/data_new/candidates/candidate_set_{mf}.mat')
    except FileNotFoundError:
        path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/"
        mat = scipy.io.loadmat(path + f'Data/data_new/candidates/candidate_set_{mf}.mat')

    In = [inch[0][0] for inch in mat['inchi']]

    return In


def load_dataset_kernel_finger(n):

    try:
        K = np.loadtxt('Data/data_new/input_kernels/PPKr.txt')
    except (FileNotFoundError, OSError):
        path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/"
        K = np.loadtxt(path + 'Data/data_new/input_kernels/PPKr.txt')

    try:
        with open('Data/data_new/output_graphs.pickle', 'rb') as handle:
            Y = pk.load(handle)
    except (FileNotFoundError, OSError):
        path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/"

        with open(path + 'Data/data_new/output_graphs.pickle', 'rb') as handle:
            Y = pk.load(handle)

    try:
        mat = scipy.io.loadmat('Data/data_new/data_GNPS.mat')
    except (FileNotFoundError, OSError):
        path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/"
        mat = scipy.io.loadmat(path + 'Data/data_new/data_GNPS.mat')

    Fingers = mat['fp'].T

    # Divide train/test
    Cs, Fs, In, Mf = Y

    K_tr = K[:n, :n]
    K_tr_te = K[:n, n:]
    K_te_te = K[n:, n:]
    Fingers_tr = Fingers[: n]
    # Fingers_te = Fingers[n:]
    Cs_te = Cs[n:]
    Fs_te = Fs[n:]
    In_tr = In[: n]
    In_te = In[n:]
    Mf_tr = Mf[: n]
    Mf_te = Mf[n:]
    Y_tr = [Fingers_tr, In_tr, Mf_tr]
    Y_te = [Cs_te, Fs_te, In_te, Mf_te]

    return [K_tr, Y_tr], [K_tr_te, K_te_te, Y_te]


def load_candidate_finger(mf):

    try:
        mat = scipy.io.loadmat(f'Data/data_new/candidates/candidate_set_{mf}.mat')
    except (FileNotFoundError, OSError):
        path = "/tsi/clusterhome/lmotte/Implementation/metabolite-identification-with-fused-gromov-wasserstein/"
        mat = scipy.io.loadmat(path + f'Data/data_new/candidates/candidate_set_{mf}.mat')

    In = [inch[0][0] for inch in mat['inchi']]
    Fingers = mat['fp'].T

    return In, Fingers
