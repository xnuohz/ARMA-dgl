""" The main file to train a ARMA model using a full graph """

import argparse
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from dgl.data import PPIDataset
from dgl.dataloading import GraphDataLoader
from tqdm import trange
from sklearn.metrics import f1_score
from model import ARMA4NC

def get_f1(pred, label):
    pred = np.round(pred, 0).astype(np.int16)
    pred = pred.flatten()
    label = label.flatten()
    return f1_score(y_pred=pred, y_true=label, average='micro')

def train(device, model, opt, loss_fn, train_loader):
    model.train()
    epoch_loss = 0
    f1 = []

    for g in train_loader:
        g = g.to(device)
        feat = g.ndata['feat']
        label = g.ndata['label']
        logits = model(g, feat)
        loss = loss_fn(logits, label)
        f1.append(get_f1(logits.detach().cpu().numpy(), label.detach().cpu().numpy()))
        epoch_loss += loss.data.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    return epoch_loss / len(train_loader), np.mean(f1)

@torch.no_grad()
def evaluate(device, model, loss_fn, valid_loader):
    model.eval()
    loss = 0
    f1 = []
    
    for g in valid_loader:
        g = g.to(device)
        feat = g.ndata['feat']
        label = g.ndata['label']
        logits = model(g, feat)
        loss += loss_fn(logits, label)
        f1.append(get_f1(logits.detach().cpu().numpy(), label.detach().cpu().numpy()))
    
    return loss / len(valid_loader), np.mean(f1)

def main(args):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load from DGL dataset
    train_dataset = PPIDataset(mode='train')
    valid_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')

    # data loader
    train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    graph = train_dataset[0]

    # check cuda
    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    # retrieve the number of classes
    n_classes = train_dataset.num_labels

    # Extract node features
    n_features = graph.ndata['feat'].shape[1]

    # Step 2: Create model =================================================================== #
    model = ARMA4NC(in_dim=n_features,
                    hid_dim=args.hid_dim,
                    out_dim=n_classes,
                    num_stacks=args.num_stacks,
                    num_layers=args.num_layers,
                    activation=nn.ReLU(),
                    dropout=args.dropout).to(device)
    
    best_model = copy.deepcopy(model)

    # Step 3: Create training components ===================================================== #
    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lamb)

    # Step 4: training epoches =============================================================== #
    f1 = 0
    no_improvement = 0
    epochs = trange(args.epochs, desc='F1 & Loss')

    for _ in epochs:
        # Training
        train_loss, train_f1 = train(device, model, opt, loss_fn, train_loader)

        # Validation
        valid_loss, valid_f1 = evaluate(device, model, loss_fn, valid_loader)

        # Print out performance
        epochs.set_description(f'Train Loss {train_loss:.4f} | Train F1 {train_f1:.4f} | Valid Loss {valid_loss:.4f} | Valid F1 {valid_f1:.4f}')
        
        if valid_f1 < f1:
            no_improvement += 1
            if no_improvement == args.early_stopping:
                print('Early stop.')
                break
        else:
            no_improvement = 0
            f1 = valid_f1
            best_model = copy.deepcopy(model)

    _, test_f1 = evaluate(device, best_model, loss_fn, test_loader)

    print(f'Test F1 {test_f1:.4f}')
    return test_f1

if __name__ == "__main__":
    """
    ARMA Model Hyperparameters
    """
    parser = argparse.ArgumentParser(description='ARMA GCN')

    # cuda params
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index. Default: -1, using CPU.')
    # training params
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs.')
    parser.add_argument('--early-stopping', type=int, default=100, help='Patient epochs to wait before early stopping.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--lamb', type=float, default=0.0, help='L2 reg.')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size.')
    # model params
    parser.add_argument("--hid-dim", type=int, default=64, help='Hidden layer dimensionalities.')
    parser.add_argument("--num-stacks", type=int, default=3, help='Number of K.')
    parser.add_argument("--num-layers", type=int, default=2, help='Number of T.')
    parser.add_argument("--dropout", type=float, default=0.25, help='Dropout applied at all layers.')

    args = parser.parse_args()
    print(args)

    f1_lists = []

    for _ in range(50):
        f1_lists.append(main(args))
    
    f1_lists = list(reversed(sorted(f1_lists)))
    f1_lists_top = np.array(f1_lists[:10])

    mean = np.around(np.mean(f1_lists_top, axis=0), decimals=3)
    std = np.around(np.std(f1_lists_top, axis=0), decimals=3)
    print('Total acc: ', f1_lists)
    print('Top 10 acc:', f1_lists_top)
    print('mean', mean)
    print('std', std)