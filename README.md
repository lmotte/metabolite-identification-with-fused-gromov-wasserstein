# Metabolite identification with fused Gromov-Wasserstein
This repository contains a Python implementation of the supervised graph prediction method proposed in http://arxiv.org/abs/2202.03813 for solving the metabolite identification problem.

## Metabolite identification problem

An important problem in metabolomics is to identify the small molecules, called metabolites, that are present in a biological sample. Mass spectrometry is a widespread method to extract distinctive features from a biological sample in the form of a tandem mass (MS/MS) spectrum. The goal of this problem is to predict the molecular structure of a metabolite given its tandem mass spectrum. Labeled data are expensive to obtain. Here we consider a set of 4138 labeled data, that have been extracted and processed in Dührkop et al. (2015), from the GNPS public spectral library.


## Quick start

**Run hyper-parameters selection.**
Code to run the hyper-parameters selection for a given method:
```
python hyper_param_selection.py method
```
with method="finger", "gk", "gw_onehot", "gw_fine", "gw_diffuse". 

The results are saved in the folder "Results".

**Load data.**

**Train.** Python code to run the a training with the 
```
```

**Test.**
```
```


## References
