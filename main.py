import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from data import dataset4node, normalize_edge, create_few_data_folder
from torch_geometric.utils import add_self_loops, dropout_adj, to_dense_adj, remove_self_loops
from util import set_all_seed, NodeEva
from model.backbones.GCN import GCN
from model.pretrain.DGI import DGI, DGI_process
from model.pretrain.GRACE import Model, drop_feature
from model.pretrain.graphmae import build_model
from model.classifier import LogReg
from model.projector import Projector
from model.prompt.GP2F import consistent_topology_loss_with_fused_sim,CrossViewContrastiveLoss,DualBranchFramework,PromptModule
import warnings
import time
warnings.filterwarnings("ignore", category=UserWarning)


def print_gpu_memory_detail(component, mem_before, mem_after):
    component_mem = (mem_after - mem_before) / 1024**3
    total_mem = mem_after / 1024**3
    print(f"{component}占用显存{component_mem:.3f}GB，总占用{total_mem:.2f}GB")

def pretrain(args, device):
    print("Starting pretraining...")
    data, input_dim, _ = dataset4node(args.pretrain_dataset, args.dataset_dir)
    
    edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32)    
    edge_index, edge_weight = add_self_loops(data.edge_index, edge_weight)
    edge_weight = normalize_edge(edge_index, edge_weight, data.num_nodes).to(device)
    edge_index = edge_index.to(device)
    data = data.to(device)
    
    # unified model selection, define your model and loss_func
    model = None
    loss_func = None
    if args.pretrain_type == 'DGI':
        gnn = GCN(input_dim, args.hidden_dim, args.num_layers, dropout=0.0, jk='last', act=nn.PReLU()).to(device)
        model = DGI(gnn, input_dim, args.hidden_dim, 'prelu').to(device)
        loss_func = nn.BCEWithLogitsLoss()
    elif args.pretrain_type == 'GRACE':
        # encoder = Encoder(input_dim, args.hidden_dim, nn.PReLU(),
        #               base_model=GCNConv, k=2).to(device)
        gnn = GCN(input_dim, args.hidden_dim, args.num_layers, dropout=0.0, jk='last', act=nn.PReLU()).to(device)
        model = Model(gnn, args.hidden_dim, args.hidden_dim, 0.5).to(device)
    elif args.pretrain_type == 'GraphMAE':
        model = build_model(num_hidden = args.hidden_dim, num_features = input_dim).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.pretrain_lr, weight_decay=args.pretrain_wd)

    # Training loop
    loss = None
    cnt_wait = 0
    min_loss = 1e9
    best_epoch = 0
    with tqdm(total=args.pretrain_epochs, desc='(T)') as pbar:
        for epoch in range(args.pretrain_epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            if args.pretrain_type == 'DGI':
                shuf_x, lbl = DGI_process(data.num_nodes, data.x)
                shuf_x = shuf_x.to(device)
                lbl = lbl.to(device)
                logits = model(data.x, shuf_x, edge_index, None, None, None)
                loss = loss_func(logits, lbl)    
            elif args.pretrain_type == 'GRACE':
                edge_index_1, edge_weight_1 = dropout_adj(edge_index, edge_weight, p=0.2)
                edge_index_2, edge_weight_2 = dropout_adj(edge_index, edge_weight, p=0.2)
                x_1 = drop_feature(data.x, 0.2)
                x_2 = drop_feature(data.x, 0.2)
                z1 = model(x_1, edge_index_1, edge_weight_1)
                z2 = model(x_2, edge_index_2, edge_weight_2)
                loss = model.loss(z1, z2, batch_size=0)
                # z = model(data.x, edge_index, edge_weight)
            elif args.pretrain_type == 'GraphMAE':
                loss, _ = model(data.x, edge_index, edge_weight)
            pbar.set_postfix({'loss': loss})
            pbar.update()
                
            if loss < min_loss:
                min_loss = loss
                best_epoch = epoch
                cnt_wait = 0
                model_save_path = f'{args.pretrain_model_path}/{args.pretrain_type}/{args.pretrain_dataset}/lr_{args.pretrain_lr}_weightdecay_{args.pretrain_wd}_hid_dim_{args.hidden_dim}.pkl'
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  
                if args.pretrain_type == 'GraphMAE':
                    torch.save(model.encoder.state_dict(), model_save_path) 
                else:
                    torch.save(model.gnn.state_dict(), model_save_path)  # 只保存gnn             
            else:
                cnt_wait += 1
                
            if cnt_wait == args.patience:
                print('Early stopping!')  # 早停，可以用于确定epoch数
                break
    
            loss.backward()
            optimizer.step()

    print("Pretraining completed.")


def prompt_tuning(args, device):
    print("Starting prompt tuning...")
    data, down_dim, num_class = dataset4node(args.downstream_dataset, args.dataset_dir)
    _, input_dim, _ = dataset4node(args.pretrain_dataset, args.dataset_dir)
        
    edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32)    
    edge_index, edge_weight =  remove_self_loops(data.edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
    edge_weight = normalize_edge(edge_index, edge_weight, data.num_nodes).to(device)
    edge_index = edge_index.to(device)
    data = data.to(device)

    create_few_data_folder(args, data, num_class)

    test_accs = []
    f1s = []
    rocs = []
    prcs = []
    
    for trail in range(1, args.trails + 1):
        torch.cuda.empty_cache()
        # 加载冻结模型
        model = GCN(input_dim, args.hidden_dim, args.num_layers, dropout=0.0, jk='last', act=nn.PReLU()).to(device)
        if args.pretrain_type == 'GraphMAE':
            pretrain_model = build_model(num_hidden = args.hidden_dim, num_features = input_dim).to(device)
            model = pretrain_model.encoder
        model_save_path = f'{args.pretrain_model_path}/{args.pretrain_type}/{args.pretrain_dataset}/lr_{args.pretrain_lr}_weightdecay_{args.pretrain_wd}_hid_dim_{args.hidden_dim}.pkl'
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        for param in model.parameters():
            param.requires_grad = False
        # 初始化可学习模块
        projector = Projector(down_dim, input_dim).to(device)
        classifier = LogReg(args.hidden_dim, num_class).to(device)
        dual_framework = DualBranchFramework(args.hidden_dim, 'adaptive').to(device)
        contrast_loss_fn = CrossViewContrastiveLoss(tau=args.temp_ctr).to(device)
        prompt = PromptModule(args.hidden_dim, args.num_layers, args.r).to(device)

        # 优化器(只优化可学习部分)
        model_param_group = [
            {"params": projector.parameters()},
            {"params": classifier.parameters()},
            {"params": prompt.parameters()}, 
            {"params": dual_framework.parameters()},
        ]
        down_optim = torch.optim.Adam(model_param_group, lr=args.downstream_lr, weight_decay=args.downstream_wd)
        
        # 加载训练/测试数据
        train_idx = torch.load(f"{args.sample_data_path}/{args.downstream_dataset}/{args.shot}_shot/{trail}/train_idx.pt", weights_only=False).type(torch.long).to(device)
        train_lbls = torch.load(f"{args.sample_data_path}/{args.downstream_dataset}/{args.shot}_shot/{trail}/train_labels.pt", weights_only=False).type(torch.long).squeeze().to(device)
        test_idx = torch.load(f"{args.sample_data_path}/{args.downstream_dataset}/{args.shot}_shot/{trail}/test_idx.pt", weights_only=False).type(torch.long).to(device)
        test_lbls = torch.load(f"{args.sample_data_path}/{args.downstream_dataset}/{args.shot}_shot/{trail}/test_labels.pt", weights_only=False).type(torch.long).squeeze().to(device)
        
        # 训练循环
        min_loss = 1e9
        cnt_wait = 0
        best_epoch = 0
        best_projector_state = None
        best_classifier_state = None
        best_prompt_state = None
        best_dual_framework_state = None
        
        target_adj = to_dense_adj(edge_index)[0]
        

        contrast_loss_fn.build_positive_mask(edge_index, data.num_nodes)

        with tqdm(total=args.downstream_epochs, desc='(T)') as pbar:
            for epoch in range(args.downstream_epochs):
                projector.train()
                classifier.train()
                prompt.train()
                dual_framework.train()
                model.eval()

                down_optim.zero_grad()
                
                feature = projector(data.x)

                if args.pretrain_type == 'GraphMAE':
                    frozen_output = model(feature, edge_index, edge_weight)
                    prompt_output = prompt(feature, edge_index, model, edge_weight)
                else:
                    frozen_output = model(feature, edge_index)   
                    prompt_output = prompt(feature, edge_index, model)

                
                contrast_loss, frozen_sim, tuned_sim = contrast_loss_fn(frozen_output, prompt_output)

                fused_similarity, fused_output, _ = dual_framework(frozen_output, prompt_output, frozen_sim, tuned_sim)
                
                logits = classifier(fused_output)
                
                cls_loss = F.cross_entropy(logits[train_idx], train_lbls)
                
                adj_loss = consistent_topology_loss_with_fused_sim(fused_similarity, edge_index, target_adj, tau=args.temp_adj, percentile=70)

                loss = cls_loss + 0.1 * adj_loss + 0.05 * contrast_loss
                # ===============================================

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'cls_loss': f'{cls_loss.item():.4f}',
                    'adj_loss': f'{adj_loss.item():.4f}',
                    'contrast_loss': f'{contrast_loss.item():.4f}'
                })
                pbar.update()
                
                # 早停
                if loss < min_loss:
                    min_loss = loss
                    best_epoch = epoch
                    cnt_wait = 0
                    best_prompt_state = prompt.state_dict()
                    best_classifier_state = classifier.state_dict()
                    best_projector_state = projector.state_dict()
                    best_dual_framework_state = dual_framework.state_dict()
                else:
                    cnt_wait += 1
                
                if cnt_wait == args.patience:
                    print(f'Early stopping at epoch {epoch}!')
                    break

                loss.backward()
                down_optim.step()

                
        # 评估
        print(f'Loading {best_epoch}th epoch')
        prompt.load_state_dict(best_prompt_state)
        classifier.load_state_dict(best_classifier_state)
        projector.load_state_dict(best_projector_state)
        dual_framework.load_state_dict(best_dual_framework_state)
        
        projector.eval()
        prompt.eval()
        dual_framework.eval()
        classifier.eval()
        
        with torch.no_grad():            
            feature = projector(data.x)

            if args.pretrain_type == 'GraphMAE':
                frozen_output = model(feature, edge_index, edge_weight)
                prompt_output = prompt(feature, edge_index, model, edge_weight)
            else:
                frozen_output = model(feature, edge_index)   
                prompt_output = prompt(feature, edge_index, model)

            fused_output = dual_framework.encode(frozen_output, prompt_output)
            logits = classifier(fused_output)
            
            test_acc, ma_f1, roc, prc = NodeEva(logits, test_idx, data, num_class, device)
            print(f"Trail {trail} - Acc: {test_acc:.4f} | F1: {ma_f1:.4f} | AUROC: {roc:.4f} | AUPRC: {prc:.4f}")

            test_accs.append(test_acc)
            f1s.append(ma_f1)
            rocs.append(roc)
            prcs.append(prc)
        
        # 明确删除引用大的张量/模块
        del target_adj, feature, frozen_output, prompt_output
        del fused_similarity, fused_output, logits
        del cls_loss, adj_loss, contrast_loss, loss
        del train_idx, train_lbls, test_idx, test_lbls
        del model, projector, classifier, dual_framework, contrast_loss_fn, prompt
        del down_optim, model_param_group

    return test_accs, f1s, rocs, prcs

if __name__ == '__main__':
    # 通用超参数
    argparser = argparse.ArgumentParser(description="A simple command-line tool.")
    # dataset and model
    argparser.add_argument('--pretrain_dataset', type=str, default='Cora', help='Dataset for pretraining')
    argparser.add_argument('--downstream_dataset', type=str, default='CS', help='Dataset for downstream task')
    argparser.add_argument('--model', type=str, default='GCN', help='Model architecture')
    argparser.add_argument('--pretrain_type', type=str, default='GRACE', help='pretrain method')
    argparser.add_argument('--prompt_type', type=str, default='gpf', help='Type of prompt to use')
    # other settings
    argparser.add_argument('--trails', type=int, default=5, help='Number of trials to run')
    argparser.add_argument('--shot', type=int, default=1, help='Number of shots for few-shot learning')
    argparser.add_argument('--device', type=str, default='cuda:0', help='Device to use for computation')
    argparser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    # paths
    argparser.add_argument('--dataset_dir', type=str, default='../../dataset/', help='Directory for datasets')
    argparser.add_argument('--pretrain_model_path', type=str, default='./pretrained_model/', help='Path to save/load the pretrained model')
    argparser.add_argument('--log_path', type=str, default='./result/', help='Path to save logs')
    argparser.add_argument('--sample_data_path', type=str, default='./sample_data', help='Path to save sampled few-shot data')
    # training hyperparameters
    argparser.add_argument('--pretrain_epochs', type=int, default=1000, help='Number of pretraining epochs')
    argparser.add_argument('--downstream_epochs', type=int, default=500, help='Number of downstream training epochs')
    argparser.add_argument('--pretrain_lr', type=float, default=0.1, help='Learning rate for pretraining') # Cora 0.001
    argparser.add_argument('--downstream_lr', type=float, default=0.005, help='Learning rate for downstream task') # Cora 0.001 CiteSeer(0.005)、PubMed(0.005)、Computers(0.005)、Photo(0.005) Texas 0.05
    argparser.add_argument('--pretrain_wd', type=float, default=5e-4, help='Weight decay for pretraining') # 5e-4
    argparser.add_argument('--downstream_wd', type=float, default=5e-4, help='Weight decay for downstream task') # 5e-4(别动，调小了模型学偏了，adj_loss爆炸)
    # model hyperparameters
    argparser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    argparser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model')
    
    argparser.add_argument('--r', type=int, default=32, help='Number of layers in the model')
    argparser.add_argument('--temp_ctr', type=float, default=0.5, help='Temperature for contrastive loss')
    argparser.add_argument('--temp_adj', type=float, default=0.05, help='Temperature for adjacency loss')
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
        mean_test_acc = np.mean(accs_seed)
        std_test_acc = np.std(accs_seed)    
        mean_f1 = np.mean(f1s_seed)
        std_f1 = np.std(f1s_seed)   
        mean_roc = np.mean(rocs_seed)
        std_roc = np.std(rocs_seed)   
        mean_prc = np.mean(prcs_seed)
        std_prc = np.std(prcs_seed)
        with open(args.log_path + args.downstream_dataset + ".txt", 'a') as f:
            f.write(" seed {}".format(seed))
            f.write(" Final best | test Accuracy {:.4f} ± {:.4f}(std)".format(mean_test_acc, std_test_acc))
            f.write(" Final best | test F1 {:.4f} ± {:.4f}(std)".format(mean_f1, std_f1))
            f.write(" Final best | AUROC {:.4f} ± {:.4f}(std)".format(mean_roc, std_roc))
            f.write(f"Task completed: {args.pretrain_type} using {args.prompt_type} from {args.pretrain_dataset} to {args.downstream_dataset} at {args.shot}-shot \n\n")
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
