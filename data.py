import os
import torch
from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr, WebKB, Actor, WikipediaNetwork, HeterophilousGraphDataset, Coauthor
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.utils import degree, to_undirected


def dataset4node(dataname, dataset_dir):
    print(f'Dataloader: Loading {dataname}')
    dataset_mapping = {
        'Planetoid': ['PubMed', 'CiteSeer', 'Cora'],
        'Amazon': ['Computers', 'Photo'],
        'Coauthor': ['CS', 'Physics'],  
        'Reddit': ['Reddit'],
        'WikiCS': ['WikiCS'],
        'Flickr': ['Flickr'],
        'WebKB': ['Cornell', 'Wisconsin', 'Texas'],
        'WikipediaNetwork': ['Chameleon', 'Crocodile', 'Squirrel'],
        'Actor': ['Actor'],
        'HeterophilousGraphDataset': ['roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions'],
        'ogb': ['ogbn-arxiv', 'ogbn-products']
    }

    # Load dataset
    if dataname in dataset_mapping['Planetoid']:
        dataset = Planetoid(root=dataset_dir, name=dataname,
                            transform=T.NormalizeFeatures())
    elif dataname in dataset_mapping['Amazon']:
        dataset = Amazon(root=dataset_dir, name=dataname)
    elif dataname in dataset_mapping['Coauthor']: 
        dataset = Coauthor(root=dataset_dir, name=dataname, 
                           transform=T.NormalizeFeatures())
    elif dataname in dataset_mapping['Reddit']:
        dataset = Reddit(root=dataset_dir)
    elif dataname in dataset_mapping['WikiCS']:
        dataset = WikiCS(root=dataset_dir)
    elif dataname in dataset_mapping['Flickr']:
        dataset = Flickr(root=dataset_dir)
    elif dataname in dataset_mapping['WebKB']:
        dataset = WebKB(root=dataset_dir, name=dataname)
    elif dataname in dataset_mapping['Actor']:
        dataset = Actor(root=dataset_dir + '/Actor',
                        transform=T.NormalizeFeatures())
    elif dataname in dataset_mapping['WikipediaNetwork']:
        dataset = WikipediaNetwork(
            root=dataset_dir, name=dataname, transform=T.NormalizeFeatures())
    elif dataname in dataset_mapping['HeterophilousGraphDataset']:
        dataset = HeterophilousGraphDataset(root=dataset_dir, name=dataname)
    elif dataname in dataset_mapping['ogb']:
        dataset = PygNodePropPredDataset(name=dataname, root=dataset_dir)
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        data.train_idx = split_idx['train']
        data.valid_idx = split_idx['valid']
        data.test_idx = split_idx['test']
        data.edge_index = to_undirected(data.edge_index)
        input_dim = data.x.shape[1]
        out_dim = dataset.num_classes
    else:
        raise ValueError(f'Unknown dataset: {dataname}')

    # Unified processing
    if dataname != 'ogbn-arxiv':
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

    print(data)
    print(f'Dataloader: Loading {dataname} success.')
    print(f'—————————————————————————————\n')

    return data, input_dim, out_dim


def dataset4graph(dataname, dataset_dir):
    print(f'Dataloader: Loading {dataname}')
    dataset_mapping = {
        'TUDataset': ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY',
                      'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR', 'DD'],
        'GNNBenchmark': ['MNIST', 'CIFAR10', 'PATTERN', 'CLUSTER'],
        'ogbg': ['ogbg-molhiv', 'ogbg-molpcba', 'ogbg-molsider']
    }

    if dataname in dataset_mapping['TUDataset']:
        root = os.path.join(dataset_dir, 'TUDataset')
        dataset = TUDataset(root=root, name=dataname)
    elif dataname in dataset_mapping['GNNBenchmark']:
        root = os.path.join(dataset_dir, 'GNNBenchmark')
        dataset = GNNBenchmarkDataset(root=root, name=dataname, split='train')
    elif dataname in dataset_mapping['ogbg'] or dataname.startswith('ogbg-'):
        root = os.path.join(dataset_dir, 'OGB')
        dataset = PygGraphPropPredDataset(
            name=dataname, root=root, transform=T.AddSelfLoops())
    else:
        raise ValueError(f'Unknown graph dataset: {dataname}')

    # 处理缺省特征：若 num_features == 0，就给每个图填充常数特征
    if dataset.num_features == 0:
        for data in dataset:
            data.x = torch.ones((data.num_nodes, 1))
        input_dim = 1
    else:
        input_dim = dataset.num_features

    if getattr(dataset, 'num_classes', None):
        out_dim = dataset.num_classes
    else:
        out_dim = dataset.num_tasks

    print(dataset)
    print(f'Dataloader: Loading {dataname} success.')
    print('—————————————————————————————\n')

    return dataset, input_dim, out_dim


def normalize_edge(edge_index, edge_weight, num_nodes):
    row, col = edge_index
    deg = degree(row, num_nodes, dtype=torch.float32)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return norm

#################### Few-shot data preparation ####################


def create_few_data_folder(args, data, output_dim):
    k = args.shot  # shot_num
    task_num = args.trails  # task_num
    for task_index in range(1, task_num + 1):
        k_shot_folder = args.sample_data_path + '/' + \
            args.downstream_dataset + '/' + str(k) + '_shot'
        # print(k_shot_folder)
        os.makedirs(k_shot_folder, exist_ok=True)

        folder = os.path.join(k_shot_folder, str(task_index))
        if not os.path.exists(folder):
            os.makedirs(folder)
            node_sample_and_save(data, k, folder, output_dim)
            print(str(k) + ' shot ' + str(task_index) + ' th is saved!!')


def node_sample_and_save(data, k, folder, num_classes):
    labels = data.y.to('cpu')

    # Random select 90% data as test
    num_test = int(0.9 * data.num_nodes)
    if num_test < 1000:
        num_test = int(0.7 * data.num_nodes)
    perm = torch.randperm(data.num_nodes)
    test_idx = perm[:num_test]
    # rand1 = torch.randperm(data.num_nodes)
    # rand2 = torch.randperm(data.num_nodes)
    # print("Rand1=Rand2:", rand1 == rand2)
    print(torch.rand(1).item())
    print(torch.randperm(5))
    print(torch.randperm(5))
    test_labels = labels[test_idx]

    # The rest are selected as alternative train
    remaining_idx = perm[num_test:]
    remaining_labels = labels[remaining_idx]

    # Select k as train
    train_idx = torch.cat([remaining_idx[remaining_labels == i][:k]
                          for i in range(num_classes)])
    shuffled_indices = torch.randperm(train_idx.size(0))
    train_idx = train_idx[shuffled_indices]
    train_labels = labels[train_idx]

    # 保存文件
    torch.save(train_idx, os.path.join(folder, 'train_idx.pt'))
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
    torch.save(test_idx, os.path.join(folder, 'test_idx.pt'))
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))


def create_graph_data_folder(args, dataset, output_dim):
    k = args.shot
    task_num = args.trails
    for task_index in range(1, task_num + 1):
        k_shot_folder = os.path.join(
            args.sample_data_path, args.downstream_dataset, f'{k}_shot')
        os.makedirs(k_shot_folder, exist_ok=True)

        folder = os.path.join(k_shot_folder, str(task_index))
        if not os.path.exists(folder):
            os.makedirs(folder)
            graph_sample_and_save(dataset, k, folder, output_dim)
            print(str(k) + ' shot ' + str(task_index) + ' th graph task saved!!')


def graph_sample_and_save(dataset, k, folder, num_classes):
    labels = torch.tensor([data.y.item() for data in dataset])

    num_test = int(0.9 * len(dataset))
    perm = torch.randperm(len(dataset))
    test_idx = perm[:num_test]
    test_labels = labels[test_idx]

    torch.save(test_idx, os.path.join(folder, 'test_idx.pt'))
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))

    remaining_idx = perm[num_test:]
    remaining_labels = labels[remaining_idx]

    train_idx = torch.cat([remaining_idx[remaining_labels == i][:k]
                          for i in range(num_classes)])
    shuffled_indices = torch.randperm(train_idx.size(0))
    train_idx = train_idx[shuffled_indices]
    train_labels = labels[train_idx]

    torch.save(train_idx, os.path.join(folder, 'train_idx.pt'))
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
