""" The main file to train a ARMA model using a full graph """

import argparse
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import networkx as nx
import dgl

from dgl.data import LegacyTUDataset
from dgl.data.utils import split_dataset
from dgl.dataloading import GraphDataLoader
from tqdm import trange
from model import ARMA4GC

def add_degree_feature(dataset):
    min_degree, max_degree = 1e9, 0
    for g, _ in dataset:
        min_degree = min(min_degree, g.in_degrees().min().item())
        max_degree = max(max_degree, g.in_degrees().max().item())
    n_dim = max_degree - min_degree + 1
    for g, _ in dataset:
        n_nodes = g.num_nodes()
        degree_feat = torch.zeros([n_nodes, n_dim])
        degree_feat[:, g.in_degrees() - min_degree] = 1.
        g.ndata['feat'] = torch.cat([g.ndata['feat'], degree_feat], dim=1)
    return dataset

def add_clustering_coefficients_feature(dataset):
    for g, _ in dataset:
        nx_g = dgl.to_networkx(dgl.to_homogeneous(g))
        # MultiDiGraph -> Graph
        nx_g = nx.Graph(nx_g)
        cc = torch.tensor(list(nx.clustering(nx_g).values())).view([-1, 1])
        g.ndata['feat'] = torch.cat([g.ndata['feat'], cc], dim=1)
    return dataset

def add_node_label_feature(dataset):
    # load node labels
    node_labels = dataset._idx_from_zero(np.loadtxt(dataset._file_path("node_labels"), dtype=int))
    one_hot_node_labels = dataset._to_onehot(node_labels)
    # load graph indicator
    indicator = dataset._idx_from_zero(np.genfromtxt(dataset._file_path("graph_indicator"), dtype=int))
    # convert node idx for each graph
    node_idx_list = []
    for idx in range(np.max(indicator) + 1):
        node_idx = np.where(indicator == idx)
        node_idx_list.append(node_idx[0])
    # add node label feature
    for idx, g in zip(node_idx_list, dataset.graph_lists):
        node_labels_tensor = torch.tensor(one_hot_node_labels[idx, :])
        g.ndata['feat'] = torch.cat([g.ndata['feat'], node_labels_tensor], dim=1)
    return dataset

def train(device, model, opt, loss_fn, train_loader):
    model.train()
    epoch_loss = 0
    n_samples = 0

    for g, labels in train_loader:
        g = g.to(device)
        labels = labels.long().to(device)
        logits = model(g, g.ndata['feat'].float())
        loss = loss_fn(logits, labels)
        epoch_loss += loss.data.item() * len(labels)
        n_samples += len(labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return epoch_loss / n_samples

@torch.no_grad()
def evaluate(device, model, valid_loader):
    model.eval()
    acc = 0
    n_samples = 0
    
    for g, labels in valid_loader:
        g = g.to(device)
        labels = labels.long().to(device)
        logits = model(g, g.ndata['feat'].float())
        predictions = logits.argmax(dim=1) + 1
        acc += predictions.eq(labels).sum().item()
        n_samples += len(labels)
    
    return acc / n_samples

def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load from DGL dataset
    dataset = LegacyTUDataset(args.dataset)

    # node degree, clustering coefficients, node labels as additional node features
    dataset = add_degree_feature(dataset)
    dataset = add_clustering_coefficients_feature(dataset)
    dataset = add_node_label_feature(dataset)

    # data split
    train_data, valid_data, test_data = split_dataset(dataset)
    
    # data loader
    train_loader = GraphDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = GraphDataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # check cuda
    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    # retrieve the number of classes and node features
    n_features, n_classes, _ = dataset.statistics()

    # Step 2: Create model =================================================================== #
    model = ARMA4GC(in_dim=n_features,
                    hid_dim=args.hid_dim,
                    out_dim=n_classes,
                    num_stacks=args.num_stacks,
                    num_layers=args.num_layers,
                    activation=nn.ReLU(),
                    dropout=args.dropout).to(device)
    
    best_model = copy.deepcopy(model)

    # Step 3: Create training components ===================================================== #
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lamb)

    # Step 4: training epoches =============================================================== #
    acc = 0
    no_improvement = 0
    epochs = trange(args.epochs, desc='Accuracy & Loss')

    for _ in epochs:
        # Training
        train_loss = train(device, model, opt, loss_fn, train_loader)

        # Validation
        valid_acc = evaluate(device, model, valid_loader)

        # Print out performance
        epochs.set_description(f'Train Loss {train_loss:.4f} | Valid Acc {valid_acc:.4f}')
        
        if valid_acc < acc:
            no_improvement += 1
            if no_improvement == args.early_stopping:
                print('Early stop.')
                break
        else:
            no_improvement = 0
            acc = valid_acc
            best_model = copy.deepcopy(model)

    test_acc = evaluate(device, best_model, test_loader)

    print(f'Test Acc {test_acc:.4f}')
    return test_acc

if __name__ == "__main__":
    """
    ARMA Model Hyperparameters
    """
    parser = argparse.ArgumentParser(description='ARMA GCN')

    # dataset options
    parser.add_argument('--dataset', type=str, default='ENZYMES', help='Name of dataset.')
    # cuda params
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index. Default: -1, using CPU.')
    # training params
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs.')
    parser.add_argument('--early-stopping', type=int, default=50, help='Patient epochs to wait before early stopping.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--lamb', type=float, default=1e-4, help='L2 reg.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    # model params
    parser.add_argument("--hid-dim", type=int, default=32, help='Hidden layer dimensionalities.')
    parser.add_argument("--num-stacks", type=int, default=2, help='Number of K.')
    parser.add_argument("--num-layers", type=int, default=2, help='Number of T.')
    parser.add_argument("--dropout", type=float, default=0.6, help='Dropout applied at all layers.')

    args = parser.parse_args()
    print(args)

    main(args)
    # acc_lists = []

    # for _ in range(50):
    #     acc_lists.append(main(args))

    # acc_lists = list(reversed(sorted(acc_lists)))
    # acc_lists_top = np.array(acc_lists[:10])

    # mean = np.around(np.mean(acc_lists_top, axis=0), decimals=3)
    # std = np.around(np.std(acc_lists_top, axis=0), decimals=3)
    # print('Total acc: ', acc_lists)
    # print('Top 10 acc:', acc_lists_top)
    # print('mean', mean)
    # print('std', std)