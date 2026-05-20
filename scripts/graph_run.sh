# 学习率范围
pretrain_lr=(0.0005)
downstream_lr=(0.01)
temp_adj=(0.5)
temp_contrast=(0.05)
lambda_adj=(0.5)
lambda_contrast=(2.0)
# 数据集
train_dataset=('PROTEINS')
test_dataset=('PROTEINS')
# few-shot设置
shots=(50)
trails=1
log_path="./result/graph/"
# 提示类型，用于日志记录
prompt_type=('ICD')
sample_data_path="./sample_data"                        
echo "Testing prelr=$pretrain_lr, downlr=$downstream_lr, pretrain_dataset=$train_dataset, test_dataset=$test_dataset"
python graph_main.py \
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
    --temp_adj $temp_adj \
    --lambda_adj $lambda_adj \
    --temp_contrast $temp_contrast \
    --lambda_contrast $lambda_contrast
echo "Completed: prelr=$pretrain_lr, downlr=$downstream_lr, train=$train_dataset, test=$test_dataset"

# 学习率范围
pretrain_lr=(0.005)
downstream_lr=(0.0005)
temp_adj=(0.5)
temp_contrast=(0.5)
lambda_adj=(0.05)
lambda_contrast=(5.0)
pretrain_wd=(0.0001)
downstream_wd=(0)
# 数据集
train_dataset=('PROTEINS')
test_dataset=('MUTAG')
# few-shot设置
shots=(50)
trails=100
log_path="./result/graph/"
# 提示类型，用于日志记录
prompt_type=('ICD')
sample_data_path="./sample_data"                        
echo "Testing prelr=$pretrain_lr, downlr=$downstream_lr, pretrain_dataset=$train_dataset, test_dataset=$test_dataset"
python graph_main.py \
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
    --temp_adj $temp_adj \
    --lambda_adj $lambda_adj \
    --temp_contrast $temp_contrast \
    --lambda_contrast $lambda_contrast \
    --pretrain_wd $pretrain_wd \
    --downstream_wd $dowmstream_wd
echo "Completed: prelr=$pretrain_lr, downlr=$downstream_lr, train=$train_dataset, test=$test_dataset"

# 学习率范围
pretrain_lr=(0.001)
downstream_lr=(0.01)
temp_adj=(0.5)
temp_contrast=(0.05)
lambda_adj=(0.5)
lambda_contrast=(2.0)
# 数据集
train_dataset=('PROTEINS')
test_dataset=('COX2')
# few-shot设置
shots=(50)
trails=100
log_path="./result/graph/"
# 提示类型，用于日志记录
prompt_type=('ICD')
sample_data_path="./sample_data"                        
echo "Testing prelr=$pretrain_lr, downlr=$downstream_lr, pretrain_dataset=$train_dataset, test_dataset=$test_dataset"
python graph_main.py \
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
    --temp_adj $temp_adj \
    --lambda_adj $lambda_adj \
    --temp_contrast $temp_contrast \
    --lambda_contrast $lambda_contrast
echo "Completed: prelr=$pretrain_lr, downlr=$downstream_lr, train=$train_dataset, test=$test_dataset"

# 学习率范围
pretrain_lr=(0.00005)
downstream_lr=(0.05)
temp_adj=(0.2)
temp_contrast=(0.2)
lambda_adj=(0.2)
lambda_contrast=(0.01)
# 数据集
train_dataset=('PROTEINS')
test_dataset=('BZR')
# few-shot设置
shots=(50)
trails=100
log_path="./result/graph/"
# 提示类型，用于日志记录
prompt_type=('ICD')
sample_data_path="./sample_data"                        
echo "Testing prelr=$pretrain_lr, downlr=$downstream_lr, pretrain_dataset=$train_dataset, test_dataset=$test_dataset"
python graph_main.py \
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
    --temp_adj $temp_adj \
    --lambda_adj $lambda_adj \
    --temp_contrast $temp_contrast \
    --lambda_contrast $lambda_contrast
echo "Completed: prelr=$pretrain_lr, downlr=$downstream_lr, train=$train_dataset, test=$test_dataset"

# 学习率范围
pretrain_lr=(0.0001)
downstream_lr=(0.005)
temp_adj=(0.2)
temp_contrast=(0.2)
lambda_adj=(0.2)
lambda_contrast=(0.01)
pretrain_wd=(0.0001)
downstream_wd=(0.0005)
# 数据集
train_dataset=('PROTEINS')
test_dataset=('ENZYMES')
# few-shot设置
shots=(50)
trails=100
log_path="./result/graph/"
# 提示类型，用于日志记录
prompt_type=('ICD')
sample_data_path="./sample_data"                        
echo "Testing prelr=$pretrain_lr, downlr=$downstream_lr, pretrain_dataset=$train_dataset, test_dataset=$test_dataset"
python graph_main.py \
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
    --temp_adj $temp_adj \
    --lambda_adj $lambda_adj \
    --temp_contrast $temp_contrast \
    --lambda_contrast $lambda_contrast \
    --pretrain_wd $pretrain_wd \
    --downstream_wd $dowmstream_wd
echo "Completed: prelr=$pretrain_lr, downlr=$downstream_lr, train=$train_dataset, test=$test_dataset"
# 学习率范围
pretrain_lr=(0.0001)
downstream_lr=(0.05)
temp_adj=(0.2)
temp_contrast=(0.2)
lambda_adj=(0.1)
lambda_contrast=(0.01)
pretrain_wd=(0.0001)
downstream_wd=(0.001)
# 数据集
train_dataset=('PROTEINS')
test_dataset=('DD')
# few-shot设置
shots=(50)
trails=100
log_path="./result/graph/"
# 提示类型，用于日志记录
prompt_type=('ICD')
sample_data_path="./sample_data"                        
echo "Testing prelr=$pretrain_lr, downlr=$downstream_lr, pretrain_dataset=$train_dataset, test_dataset=$test_dataset"
python graph_main.py \
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
    --temp_adj $temp_adj \
    --lambda_adj $lambda_adj \
    --temp_contrast $temp_contrast \
    --lambda_contrast $lambda_contrast \
    --pretrain_wd $pretrain_wd \
    --downstream_wd $dowmstream_wd
echo "Completed: prelr=$pretrain_lr, downlr=$downstream_lr, train=$train_dataset, test=$test_dataset"
