# DGL Implementation of ARMA

This DGL example implements the GNN model proposed in the paper [Graph Neural Networks with convolutional ARMA filters](https://arxiv.org/abs/1901.01343). For the original implementation, see [here](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.ARMAConv).

Contributor: [xnuohz](https://github.com/xnuohz)

### Requirements
The codebase is implemented in Python 3.6. For version requirement of packages, see below.

```
dgl 0.6a210202
numpy 1.19.5
networkx 2.5
scikit-learn 0.24.1
tqdm 4.56.0
torch 1.7.0s
```

### The graph datasets used in this example

###### Node Classification

The DGL's built-in Cora, Pubmed, Citeseer and PPI datasets. Dataset summary:

| Dataset | #Nodes | #Edges | #Feats | #Classes | #Train Nodes | #Val Nodes | #Test Nodes |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Cora | 2,708 | 10,556 | 1,433 | 7(single label) | 140 | 500 | 1000 |
| Citeseer | 3,327 | 9,228 | 3,703 | 6(single label) | 120 | 500 | 1000 |
| Pubmed | 19,717 | 88,651 | 500 | 3(single label) | 60 | 500 | 1000 |
| PPI | 56,944 | 818,716 | 50 | 121(multi label) | 44906(20 graphs) | 6514(2 graphs) | 5524(2 graphs) |

###### Graph Classification

| Dataset | #Samples | #Classes | #Avg. nodes | #Avg. edges | #Node attr. | Node labels |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Enzymes | 600 | 6 | 32.63 | 62.14 | 18 | no |
| Proteins | 1,113 | 2 | 39.06 | 72.82 | 1 | no |
| D&D | 1,178 | 2 | 284.32 | 715.66 | - | yes |
| MUTAG | 188 | 2 | 17.93 | 19.79 | - | yes |

### Usage

###### Dataset options
```
--dataset          str     The graph dataset name.             Default is 'Cora'.
```

###### GPU options
```
--gpu              int     GPU index.                          Default is -1, using CPU.
```

###### Model options
```
--epochs           int     Number of training epochs.          Default is 2000.
--early-stopping   int     Early stopping rounds.              Default is 100.
--lr               float   Adam optimizer learning rate.       Default is 0.01.
--lamb             float   L2 regularization coefficient.      Default is 0.0005.
--hid-dim          int     Hidden layer dimensionalities.      Default is 16.
--num-stacks       int     Number of K.                        Default is 2.
--num-layers       int     Number of T.                        Default is 1.
--dropout          float   Dropout applied at all layers.      Default is 0.75.
```

###### Examples

The following commands learn a neural network and predict on the test set.
Train an ARMA model which follows the original hyperparameters on different datasets.
```bash
# Cora:
python citation.py --gpu 0

# Citeseer:
python citation.py --gpu 0 --dataset Citeseer --num-stacks 3

# Pubmed:
python citation.py --gpu 0 --dataset Pubmed --dropout 0.25 --num-stacks 1

# PPI:
python ppi.py --gpu 0

# Enzymes
python tu.py --gpu 0

# Proteins
python tu.py --gpu 0 --dataset PROTEINS --num-stacks 4 --num-layers 4

# D&D
python tu.py --gpu 0 --dataset DD --dropout 0 --num-stacks 4 --num-layers 4

# MUTAG
python tu.py --gpu 0 --dataset MUTAG --dropout 0 --num-stacks 4 --num-layers 4
```

### Performance

###### Node Classification

| Dataset | Cora | Citeseer | Pubmed | PPI |
| :-: | :-: | :-: | :-: | :-: |
| Metrics(Table 1.Node classification accuracy/f1) | 83.4±0.6 | 72.5±0.4 | 78.9±0.3 | 90.5±0.3 |
| Metrics(PyG) | 82.3±0.5 | 70.9±1.1 | 78.3±0.8 | - |
| Metrics(DGL) | 80.9±0.6 | 71.6±0.8 | 75.0±4.2 | 73.2±0.1 |

###### Graph Classification

| Dataset | Enzymes | Proteins | D&D | MUTAG |
| :-: | :-: | :-: | :-: | :-: |
| Metrics(Table 3.Graph classification accuracy.) | 60.6±7.2 | 73.7±3.4 | 77.6±2.7 | 91.5±4.2 |
| Metrics(DGL) | 98.2±2.2 | 100.0±0.0 | 100.0±0.0 | 13.0±26 |