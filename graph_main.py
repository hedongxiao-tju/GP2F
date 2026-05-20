import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from torch_geometric.nn import GCNConv      

from data import dataset4graph, normalize_edge, create_graph_data_folder
from torch_geometric.utils import add_self_loops, dropout_adj, to_dense_adj, remove_self_loops, to_dense_batch
from torch_geometric.loader import DataLoader
from util import set_all_seed, GraphEva
from model.backbones.GCN import GCN
from model.pretrain.DGI import DGI, DGI_process
from model.pretrain.GRACE import Model, drop_feature

from model.classifier import LogReg
from model.projector import Projector
from model.prompt.GP2F import consistent_topology_loss_with_fused_sim,CrossViewContrastiveLoss,DualBranchFramework,PromptModule
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def prepare_graph(graph):
    edge_weight = torch.ones(graph.edge_index.size(1))
    edge_index, edge_weight = remove_self_loops(graph.edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
    edge_weight = normalize_edge(edge_index, edge_weight, graph.num_nodes)
    graph.edge_index = edge_index
    graph.edge_weight = edge_weight
    return graph

def pretrain(args, device):
    print("Starting graph pretraining...")
    dataset, input_dim, _ = dataset4graph(args.pretrain_dataset, args.dataset_dir)
    dataset = [prepare_graph(g.clone()) for g in dataset]
    loader = DataLoader(dataset, batch_size=args.pretrain_batch_size, shuffle=True, drop_last=False)

    gnn = GCN(input_dim, args.hidden_dim, args.num_layers, dropout=0.0, jk='last', act=nn.PReLU(), pool_type='mean').to(device)
    if args.pretrain_type == 'DGI':
        model = DGI(gnn, input_dim, args.hidden_dim, 'prelu').to(device)
        loss_func = nn.BCEWithLogitsLoss()
    elif args.pretrain_type == 'GRACE':
        model = Model(gnn, args.hidden_dim, args.hidden_dim, 0.2).to(device)
        loss_func = None
    else:
        raise ValueError(f"Unsupported pretrain_type {args.pretrain_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr, weight_decay=args.pretrain_wd)
    cnt_wait, min_loss, best_epoch = 0, float('inf'), 0
    model_path = f'{args.pretrain_model_path}/{args.pretrain_type}/{args.pretrain_dataset}/lr_{args.pretrain_lr}_weightdecay_{args.pretrain_wd}_hid_dim_{args.hidden_dim}.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with tqdm(total=args.pretrain_epochs, desc='(T)') as pbar:
        for epoch in range(args.pretrain_epochs):
            model.train()
            epoch_loss = 0.0
            for batch in loader:
                batch = batch.to(device)
                edge_index = batch.edge_index
                edge_weight = batch.edge_weight
                optimizer.zero_grad()
                if args.pretrain_type == 'DGI':
                    shuf_x, lbl = DGI_process(batch.num_nodes, batch.x)
                    shuf_x, lbl = shuf_x.to(device), lbl.to(device)
                    logits = model(batch.x, shuf_x, edge_index, None, None, None)
                    loss = loss_func(logits, lbl)
                else:
                    edge_index_1, edge_weight_1 = dropout_adj(edge_index, edge_weight, p=0.2)
                    edge_index_2, edge_weight_2 = dropout_adj(edge_index, edge_weight, p=0.2)
                    x_1, x_2 = drop_feature(batch.x, 0.2), drop_feature(batch.x, 0.2)
                    z1 = model(x_1, edge_index_1, edge_weight_1)
                    z2 = model(x_2, edge_index_2, edge_weight_2)
                    loss = model.loss(z1, z2, batch_size=0)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(loader)
            pbar.set_postfix({'loss': f'{epoch_loss:.4f}'})
            pbar.update()

            if epoch_loss < min_loss:
                min_loss, best_epoch, cnt_wait = epoch_loss, epoch, 0
                torch.save(model.gnn.state_dict(), model_path)
            else:
                cnt_wait += 1
            if cnt_wait == args.patience:
                print('Early stopping!')
                break
    print(f"Pretraining completed. Best epoch: {best_epoch}")
    
def prompt_tuning(args, device):
    print("Starting prompt tuning (graph classification)...")
    dataset, down_dim, num_class = dataset4graph(args.downstream_dataset, args.dataset_dir)
    _, input_dim, _ = dataset4graph(args.pretrain_dataset, args.dataset_dir)
    dataset = [prepare_graph(g.clone()) for g in dataset]
    create_graph_data_folder(args, dataset, num_class)

    accs=[]
    f1s=[]
    rocs=[]
    prcs=[]
    for trail in range(1, args.trails + 1):
        torch.cuda.empty_cache()

        model = GCN(input_dim, args.hidden_dim, args.num_layers, dropout=0.0, jk='last',
                    act=nn.PReLU(), pool_type='mean').to(device)
        model_path = f'{args.pretrain_model_path}/{args.pretrain_type}/{args.pretrain_dataset}/lr_{args.pretrain_lr}_weightdecay_{args.pretrain_wd}_hid_dim_{args.hidden_dim}.pkl'
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        for param in model.parameters():
            param.requires_grad = False

        projector = Projector(down_dim, input_dim).to(device)
        classifier = LogReg(args.hidden_dim, num_class).to(device)
        prompt = PromptModule(args.hidden_dim, args.num_layers, args.r, alpha_init=args.alpha).to(device)
        dual_framework = DualBranchFramework(args.hidden_dim, 'adaptive').to(device)
        contrast_loss_fn = CrossViewContrastiveLoss(tau=args.temp_contrast).to(device)

        optimizer = torch.optim.Adam(
            [{"params": projector.parameters()},
             {"params": classifier.parameters()},
             {"params": prompt.parameters()},
             {"params": dual_framework.parameters()}],
            lr=args.downstream_lr, weight_decay=args.downstream_wd
        )
        

        train_idx = torch.load(f"{args.sample_data_path}/{args.downstream_dataset}/{args.shot}_shot/{trail}/train_idx.pt",
                               map_location='cpu').long()
        train_lbls = torch.load(f"{args.sample_data_path}/{args.downstream_dataset}/{args.shot}_shot/{trail}/train_labels.pt",
                                map_location='cpu').long().squeeze()
        test_idx = torch.load(f"{args.sample_data_path}/{args.downstream_dataset}/{args.shot}_shot/{trail}/test_idx.pt",
                              map_location='cpu').long()
        test_lbls = torch.load(f"{args.sample_data_path}/{args.downstream_dataset}/{args.shot}_shot/{trail}/test_labels.pt",
                               map_location='cpu').long().squeeze()

        train_graphs = [dataset[i] for i in train_idx.tolist()]
        test_graphs = [dataset[i] for i in test_idx.tolist()]
        train_loader = DataLoader(train_graphs, batch_size=args.downstream_batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_graphs, batch_size=args.downstream_batch_size, shuffle=False, drop_last=False)

        min_loss, cnt_wait, best_states = float('inf'), 0, {}
        with tqdm(total=args.downstream_epochs, desc=f'(Trail {trail})') as pbar:
            for epoch in range(args.downstream_epochs):
                projector.train(); classifier.train(); prompt.train(); dual_framework.train(); model.eval()
                epoch_loss = 0.0
                for batch in train_loader:
                    batch = batch.to(device)
                    edge_index = batch.edge_index
                    optimizer.zero_grad()
                    feature = projector(batch.x)
                    frozen_output = model(feature, edge_index)
                    prompt_output = prompt(feature, edge_index, model)

                    target_adj = to_dense_adj(edge_index, max_num_nodes=batch.x.size(0))[0]

                    contrast_loss_fn.build_positive_mask(edge_index, batch.num_nodes)
                    contrast_loss, frozen_sim, tuned_sim = contrast_loss_fn(frozen_output, prompt_output)

                    fused_similarity, fused_output, _ = dual_framework(frozen_output, prompt_output, frozen_sim, tuned_sim)
                    
                    graph_emb = model.pool(fused_output, batch.batch)
                    
                    logits = classifier(graph_emb)
                    
                    cls_loss = F.cross_entropy(logits, batch.y)
                                        
                    adj_loss = consistent_topology_loss_with_fused_sim(fused_similarity, edge_index, target_adj, tau=args.temp_adj, percentile=70)

                    loss = cls_loss + args.lambda_adj * adj_loss + args.lambda_contrast * contrast_loss
                    
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                epoch_loss /= max(len(train_loader), 1)
                pbar.set_postfix({'loss': f'{epoch_loss:.4f}', 'cls': f'{cls_loss.item():.4f}',
                                  'adj': f'{adj_loss.item():.4f}', 'contrast': f'{contrast_loss.item():.4f}'})
                pbar.update()

                if epoch_loss < min_loss:
                    min_loss, cnt_wait = epoch_loss, 0
                    best_states = {
                        'projector': deepcopy(projector.state_dict()),
                        'classifier': deepcopy(classifier.state_dict()),
                        'prompt': deepcopy(prompt.state_dict()),
                        'dual': deepcopy(dual_framework.state_dict()),
                    }
                else:
                    cnt_wait += 1
                if cnt_wait == args.patience:
                    print(f'Early stopping at epoch {epoch}!')
                    break

        projector.load_state_dict(best_states['projector'])
        classifier.load_state_dict(best_states['classifier'])
        prompt.load_state_dict(best_states['prompt'])
        dual_framework.load_state_dict(best_states['dual'])
        projector.eval(); classifier.eval(); prompt.eval(); dual_framework.eval(); model.eval()

        with torch.no_grad():
            logits_list, label_list = [], []
            for batch in test_loader:
                batch = batch.to(device)
                edge_index = batch.edge_index

                feature = projector(batch.x)
                frozen_output = model(feature, edge_index)
                prompt_output = prompt(feature, edge_index, model)
                fused_output = dual_framework.encode(frozen_output, prompt_output)
                graph_emb = model.pool(fused_output, batch.batch)
                logits = classifier(graph_emb)
                logits_list.append(logits.cpu())
                label_list.append(batch.y.cpu())

            logits = torch.cat(logits_list)
            labels = torch.cat(label_list)
            acc, f1, roc, prc = GraphEva(logits, labels, num_class, device)
            print(f"Trail {trail} - Acc: {acc:.4f} | F1: {f1:.4f} | AUROC: {roc:.4f} | AUPRC: {prc:.4f}")
            accs.append(acc); f1s.append(f1); rocs.append(roc); prcs.append(prc)

        del train_loader, test_loader, train_graphs, test_graphs
        torch.cuda.empty_cache()

    return accs, f1s, rocs, prcs


if __name__ == '__main__':
    # 通用超参数
    argparser = argparse.ArgumentParser(description="A simple command-line tool.")
    # dataset and model
    argparser.add_argument('--pretrain_dataset', type=str, default='PROTEINS', help='Dataset for pretraining')
    argparser.add_argument('--downstream_dataset', type=str, default='PROTEINS', help='Dataset for downstream task')
    argparser.add_argument('--model', type=str, default='GCN', help='Model architecture')
    argparser.add_argument('--pretrain_type', type=str, default='GRACE', help='pretrain method')
    argparser.add_argument('--prompt_type', type=str, default='gpf', help='Type of prompt to use')
    # other settings
    argparser.add_argument('--trails', type=int, default=5, help='Number of trials to run')
    argparser.add_argument('--shot', type=int, default=20, help='Number of shots for few-shot learning')
    argparser.add_argument('--device', type=str, default='cuda:1', help='Device to use for computation')
    argparser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    # paths
    argparser.add_argument('--dataset_dir', type=str, default='../../dataset/', help='Directory for datasets')
    argparser.add_argument('--pretrain_model_path', type=str, default='./pretrained_model/', help='Path to save/load the pretrained model')
    argparser.add_argument('--log_path', type=str, default='./result/graph/', help='Path to save logs')
    argparser.add_argument('--sample_data_path', type=str, default='../LP/sample_data', help='Path to save sampled few-shot data')
    # training hyperparameters
    argparser.add_argument('--pretrain_epochs', type=int, default=1000, help='Number of pretraining epochs')
    argparser.add_argument('--downstream_epochs', type=int, default=500, help='Number of downstream training epochs')
    argparser.add_argument('--pretrain_lr', type=float, default=0.0005, help='Learning rate for pretraining') # Cora 0.001
    argparser.add_argument('--downstream_lr', type=float, default=0.01, help='Learning rate for downstream task') # Cora 0.001 CiteSeer(0.005)、PubMed(0.005)、Computers(0.005)、Photo(0.005) Texas 0.05
    argparser.add_argument('--pretrain_wd', type=float, default=1e-5, help='Weight decay for pretraining') # 5e-4
    argparser.add_argument('--downstream_wd', type=float, default=5e-4, help='Weight decay for downstream task') # 5e-4(别动，调小了模型学偏了，adj_loss爆炸)
    argparser.add_argument('--pretrain_batch_size', type=int, default=64, help='Batch size for pretraining')
    argparser.add_argument('--downstream_batch_size', type=int, default=64, help='Batch size for training')
    # model hyperparameters
    argparser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    argparser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model')
    
    argparser.add_argument('--r', type=int, default=32, help='Number of layers in the model')
    argparser.add_argument('--alpha', type=float, default=0.1, help='Alpha for prompt module')
    argparser.add_argument('--temp_adj', type=float, default=0.2, help='Temperature for adjacency loss')
    argparser.add_argument('--temp_contrast', type=float, default=0.2, help='Temperature for contrastive loss')
    argparser.add_argument('--lambda_adj', type=float, default=0.2, help='Weight for adjacency loss')
    argparser.add_argument('--lambda_contrast', type=float, default=0.01, help='Weight for contrastive loss')
    args = argparser.parse_args()
        

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # log all arguments
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    with open(args.log_path + args.downstream_dataset + ".txt", 'a') as f:
        f.write('**'*40+'\n')
        f.write(str(args) + '\n')
    
    test_accs = []
    f1s = []
    rocs = []
    prcs = []

    seed_list = [12345, 23456, 34567, 45678, 56789]

    for seed in seed_list:
        set_all_seed(seed)

        pretrain(args, device)
        accs_seed, f1s_seed, rocs_seed, prcs_seed = prompt_tuning(args, device)  # 拿返回值
        
        test_accs.extend(accs_seed)
        f1s.extend(f1s_seed)
        rocs.extend(rocs_seed)
        prcs.extend(prcs_seed)
              
    mean_test_acc = np.mean(test_accs)
    std_test_acc = np.std(test_accs)    
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)   
    mean_roc = np.mean(rocs)
    std_roc = np.std(rocs)   
    mean_prc = np.mean(prcs)
    std_prc = np.std(prcs)
    print(" Final best | test Accuracy {:.4f} ± {:.4f}(std)".format(mean_test_acc, std_test_acc))   
    print(" Final best | test F1 {:.4f} ± {:.4f}(std)".format(mean_f1, std_f1))   
    print(" Final best | AUROC {:.4f} ± {:.4f}(std)".format(mean_roc, std_roc))   
    print(f"Task completed: {args.pretrain_type} using {args.prompt_type} from {args.pretrain_dataset} to {args.downstream_dataset} at {args.shot}-shot")
    with open(args.log_path + args.downstream_dataset + ".txt", 'a') as f:
        f.write(" Final best | test Accuracy {:.4f} ± {:.4f}(std)".format(mean_test_acc, std_test_acc))
        f.write(" Final best | test F1 {:.4f} ± {:.4f}(std)".format(mean_f1, std_f1))
        f.write(" Final best | AUROC {:.4f} ± {:.4f}(std)".format(mean_roc, std_roc))
        f.write(f"Task completed: {args.pretrain_type} using {args.prompt_type} from {args.pretrain_dataset} to {args.downstream_dataset} at {args.shot}-shot \n\n")
    print('----------------- \n')
