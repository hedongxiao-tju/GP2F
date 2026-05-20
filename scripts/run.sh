#!/bin/bash

# 学习率范围
pretrain_lr=(0.0005)
downstream_lr=(0.001)
temp_ctr=(0.5)
temp_adj=(0.05)
# 数据集
train_dataset=('Cora')
test_dataset=('Cora')
# few-shot设置
shots=(1)
trails=100
log_path="./result/node/"
# 提示类型，用于日志记录
prompt_type=('ICD')
sample_data_path="./sample_data"                        
echo "Testing prelr=$pretrain_lr, downlr=$downstream_lr, pretrain_dataset=$train_dataset, test_dataset=$test_dataset"
python main.py \
    --pretrain_dataset "$train_dataset" \
    --downstream_dataset "$test_dataset" \
    --prompt_type "$prompt_type" \
    --shot "$shots" \
    --pretrain_lr "$pretrain_lr" \
    --downstream_lr "$downstream_lr" \
    --device "cuda:0" \
    --trails $trails \
    --log_path $log_path \
    --sample_data_path $sample_data_path \
    --temp_ctr $temp_ctr \
    --temp_adj $temp_adj
echo "Completed: prelr=$pretrain_lr, downlr=$downstream_lr, train=$train_dataset, test=$test_dataset"

# 学习率范围
pretrain_lr=(0.0001)
downstream_lr=(0.005)
temp_ctr=(0.2)
temp_adj=(0.1)
# 数据集
train_dataset=('Cora')
test_dataset=('CiteSeer')
# few-shot设置
shots=(1)
trails=100
log_path="./result/node/"
# 提示类型，用于日志记录
prompt_type=('ICD')
sample_data_path="./sample_data"                        
echo "Testing prelr=$pretrain_lr, downlr=$downstream_lr, pretrain_dataset=$train_dataset, test_dataset=$test_dataset"
python main.py \
    --pretrain_dataset "$train_dataset" \
    --downstream_dataset "$test_dataset" \
    --prompt_type "$prompt_type" \
    --shot "$shots" \
    --pretrain_lr "$pretrain_lr" \
    --downstream_lr "$downstream_lr" \
    --device "cuda:0" \
    --trails $trails \
    --log_path $log_path \
    --sample_data_path $sample_data_path \
    --temp_ctr $temp_ctr \
    --temp_adj $temp_adj
echo "Completed: prelr=$pretrain_lr, downlr=$downstream_lr, train=$train_dataset, test=$test_dataset"

# 学习率范围
pretrain_lr=(0.00005)
downstream_lr=(0.01)
temp_ctr=(0.2)
temp_adj=(0.1)
# 数据集
train_dataset=('Cora')
test_dataset=('PubMed')
# few-shot设置
shots=(1)
trails=100
log_path="./result/node/"
# 提示类型，用于日志记录
prompt_type=('ICD')
sample_data_path="./sample_data"                        
echo "Testing prelr=$pretrain_lr, downlr=$downstream_lr, pretrain_dataset=$train_dataset, test_dataset=$test_dataset"
python main.py \
    --pretrain_dataset "$train_dataset" \
    --downstream_dataset "$test_dataset" \
    --prompt_type "$prompt_type" \
    --shot "$shots" \
    --pretrain_lr "$pretrain_lr" \
    --downstream_lr "$downstream_lr" \
    --device "cuda:0" \
    --trails $trails \
    --log_path $log_path \
    --sample_data_path $sample_data_path \
    --temp_ctr $temp_ctr \
    --temp_adj $temp_adj
echo "Completed: prelr=$pretrain_lr, downlr=$downstream_lr, train=$train_dataset, test=$test_dataset"

# 学习率范围
# pretrain_lr=(0.0001)
# downstream_lr=(0.0005)
pretrain_lr=(0.00005)
downstream_lr=(0.0001)
temp_ctr=(0.5)
temp_adj=(0.05)
# 数据集
train_dataset=('Cora')
test_dataset=('Computers')
# few-shot设置
shots=(1)
trails=100
log_path="./result/node/"
# 提示类型，用于日志记录
prompt_type=('ICD')
sample_data_path="./sample_data"                        
echo "Testing prelr=$pretrain_lr, downlr=$downstream_lr, pretrain_dataset=$train_dataset, test_dataset=$test_dataset"
python main.py \
    --pretrain_dataset "$train_dataset" \
    --downstream_dataset "$test_dataset" \
    --prompt_type "$prompt_type" \
    --shot "$shots" \
    --pretrain_lr "$pretrain_lr" \
    --downstream_lr "$downstream_lr" \
    --device "cuda:0" \
    --trails $trails \
    --log_path $log_path \
    --sample_data_path $sample_data_path \
    --temp_ctr $temp_ctr \
    --temp_adj $temp_adj
echo "Completed: prelr=$pretrain_lr, downlr=$downstream_lr, train=$train_dataset, test=$test_dataset"

# 学习率范围
pretrain_lr=(0.00005)
downstream_lr=(0.001)
temp_ctr=(0.5)
temp_adj=(0.05)
# 数据集
train_dataset=('Cora')
test_dataset=('Photo')
# few-shot设置
shots=(1)
trails=100
log_path="./result/node/"
# 提示类型，用于日志记录
prompt_type=('ICD')
sample_data_path="./sample_data"                        
echo "Testing prelr=$pretrain_lr, downlr=$downstream_lr, pretrain_dataset=$train_dataset, test_dataset=$test_dataset"
python main.py \
    --pretrain_dataset "$train_dataset" \
    --downstream_dataset "$test_dataset" \
    --prompt_type "$prompt_type" \
    --shot "$shots" \
    --pretrain_lr "$pretrain_lr" \
    --downstream_lr "$downstream_lr" \
    --device "cuda:0" \
    --trails $trails \
    --log_path $log_path \
    --sample_data_path $sample_data_path \
    --temp_ctr $temp_ctr \
    --temp_adj $temp_adj
echo "Completed: prelr=$pretrain_lr, downlr=$downstream_lr, train=$train_dataset, test=$test_dataset"

# 学习率范围
pretrain_lr=(0.00005)
downstream_lr=(0.001)
temp_ctr=(0.5)
temp_adj=(0.05)
# 数据集
train_dataset=('Cora')
test_dataset=('WikiCS')
# few-shot设置
shots=(1)
trails=100
log_path="./result/node/"
# 提示类型，用于日志记录
prompt_type=('ICD')
sample_data_path="./sample_data"                        
echo "Testing prelr=$pretrain_lr, downlr=$downstream_lr, pretrain_dataset=$train_dataset, test_dataset=$test_dataset"
python main.py \
    --pretrain_dataset "$train_dataset" \
    --downstream_dataset "$test_dataset" \
    --prompt_type "$prompt_type" \
    --shot "$shots" \
    --pretrain_lr "$pretrain_lr" \
    --downstream_lr "$downstream_lr" \
    --device "cuda:0" \
    --trails $trails \
    --log_path $log_path \
    --sample_data_path $sample_data_path \
    --temp_ctr $temp_ctr \
    --temp_adj $temp_adj
echo "Completed: prelr=$pretrain_lr, downlr=$downstream_lr, train=$train_dataset, test=$test_dataset"

# 学习率范围
pretrain_lr=(0.0001)
downstream_lr=(0.0001)
temp_ctr=(0.5)
temp_adj=(0.05)
# 数据集
train_dataset=('Cora')
test_dataset=('CS')
# few-shot设置
shots=(1)
trails=100
log_path="./result/node/"
# 提示类型，用于日志记录
prompt_type=('ICD')
sample_data_path="./sample_data"                        
echo "Testing prelr=$pretrain_lr, downlr=$downstream_lr, pretrain_dataset=$train_dataset, test_dataset=$test_dataset"
python main.py \
    --pretrain_dataset "$train_dataset" \
    --downstream_dataset "$test_dataset" \
    --prompt_type "$prompt_type" \
    --shot "$shots" \
    --pretrain_lr "$pretrain_lr" \
    --downstream_lr "$downstream_lr" \
    --device "cuda:0" \
    --trails $trails \
    --log_path $log_path \
    --sample_data_path $sample_data_path \
    --temp_ctr $temp_ctr \
    --temp_adj $temp_adj
echo "Completed: prelr=$pretrain_lr, downlr=$downstream_lr, train=$train_dataset, test=$test_dataset"