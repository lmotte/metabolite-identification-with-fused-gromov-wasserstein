# Metabolite identification with fused Gromov-Wasserstein
This repository contains a Python implementation of the supervised graph prediction method proposed in [[1]](#references) for solving the metabolite identification problem.

## Metabolite identification problem

An important problem in metabolomics is to identify the small molecules, called metabolites, that are present in a biological sample. Mass spectrometry is a widespread method to extract distinctive features from a biological sample in the form of a tandem mass (MS/MS) spectrum. The goal of this problem is to predict the molecular structure of a metabolite given its tandem mass spectrum.

**Dataset.** The data are available for download from https://zenodo.org/record/804241#.Yi9bzS_pNhE. It consists in a set of 4138 labeled data used in [[2]](#references) to evaluate the performance of metabolite identification from tandem mass spectra. These data have been extracted and processed in [[3]](#references).  It consists in 4138 MS/MS spectra extracted from the GNPS public spectral library (https://gnps.ucsd.edu/ProteoSAFe/libraries.jsp). The candidate sets have been build with molecular structures from PubChem.

## Quick start

**Load data.**
```python
from Utils.load_data import load_dataset_kernel_graph

n_tr = 3000
n_te = 1148
D_tr, D_te = load_dataset_kernel_graph(n_tr)
K_tr, Y_tr = D_tr
K_tr_te, K_te_te, Y_te = D_te
```

**Input pre-processing (optional).** Centering and normalizing the input kernel improves the statistical performance. 
```python
from Utils.metabolites_utils import center_gram_matrix, normalize_gram_matrix

center, normalize = True, True
if center:
    K_tr_te = center_gram_matrix(K_tr_te, K_tr, K_tr_te, K_tr)
    K_tr = center_gram_matrix(K_tr)
if normalize:
    K_tr_te = normalize_gram_matrix(K_tr_te, K_tr, K_te_te)
    K_tr = normalize_gram_matrix(K_tr)
```

**Train.**
```python
from Methods.method_gromov_wasserstein import FgwEstimator
from Utils.diffusion import diffuse

clf = FgwEstimator()
clf.ground_metric = 'diffuse'
L = 1e-4  # kernel ridge regularization parameter
clf.tau = 0.6  # the bigger tau is the more the neighbor atoms have similar feature. This impacts the FGW's ground metric.
Y_Tr = diffuse(Y_tr, clf.tau)
clf.train(K_tr, Y_tr, L)
```

**Predict and compute the test scores.**
```python
n_bary = 5  # Number of kept alpha_i(x) when predicting
n_c_max = 500   # Do not predict test input with more than n_c_max candidates
fgw, topk, n_pred = clf.predict(K_tr_te, n_bary=n_bary, Y_te=Y_te, n_c_max=n_c_max)
```

You should obtain the following results:

- Mean FGW = 0.14209778 ± 0.0460629.
- Top-1 = 32.6%, Top-10 = 62.7%, Top-20 = 72.5%.
- Number of predictions = 448.


## Reproducing the experiments in [[1]](#references)

[Brogat-Motte et al., 2022 (Section 6.2)](#references) experimentally assess the performance of fused Gromov-Wasserstein barycenter for predicting metabolites from mass spectra. In particular, a comparison of the prediction performance of different ground metrics between atoms used in the fused Gromov-Wasserstein distance between molecules ([Vayer et al., 2020](#references)) is carried out.

These experiments can be reproduced in two steps: 1) hyperparameters selection, 2) test the methods using the selected hyperparameters. It is possible to run directly the step 2).

**1) Run hyperparameters selection.**
```
python hyper_param_selection.py method
```
with method="finger", "gk", "gw_onehot", "gw_fine", "gw_diffuse". 

The results are saved in the folder "Results", where one can find already saved results.

**2) Test the methods with the selected hyperparameters**
```
python test.py method
```
The results are saved in the folder "Results", where one can already find saved results.

The Top-k accuracies obtained on the test data are given in the following table.

<center>

|                      | Top-1 | Top-10 | Top-20 |
|----------------------|:-----:|:------:|:------:|
| WL kernel            | 9.8%  | 29.1%  | 37.4%  |
| Linear fingerprint   | 28.6% | 54.5%  | 59.9%  |
| Gaussian fingerprint | 41.0% | 62.0%  | 67.8%  |
| FGW one-hot          | 12.7% | 37.3%  | 44.2%  |
| FGW fine             | 18.1% | 46.3%  | 53.7%  |
| FGW diffuse          | 27.8% | 52.8%  | 59.6%  |

</center>

## References

[1] Brogat-Motte, L., Flamary, R., Brouard, C., Rousu, J., d'Alché-Buc, F. Learning to Predict Graphs with Fused Gromov-Wasserstein Barycenters. arXiv preprint arXiv:2202.03813, 2022. (http://arxiv.org/abs/2202.03813)

[2] Brouard, C., Shen, H., Dührkop, K., d'Alché-Buc, F., Böcker, S. and Rousu, J.: Fast metabolite identification with Input Output Kernel Regression. In the proceedings of ISMB 2016, Bioinformatics 32(12): i28-i36, 2016. DOI: https://doi.org/10.1093/bioinformatics/btw246

[3] Dührkop, K., Shen, H., Meusel, M., Rousu, J. and Böcker, S.: Searching molecular structure databases with tandem mass spectra using CSI:FingerID. PNAS, 112(41), 12580-12585, 2015. doi:10.1073/pnas.1509788112

[4] Vayer, T., Chapel, L., Flamary, R., Tavenard, R., and Courty, N. Optimal transport for structured data with application on graphs. In International Conference on Machine Learning (ICML), 2019.
