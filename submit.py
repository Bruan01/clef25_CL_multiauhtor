#!/usr/bin/env python
# coding: utf-8

"""
finetune_modernbert_scl.py - 使用监督对比学习训练现代BERT模型(DeBERTa)用于文本分类
"""

import json
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 强制离线模式
# os.environ["HF_DATASETS_OFFLINE"] = "1"   # 禁用数据集下载
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaTokenizer, DebertaModel
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import nltk
import logging
import click
from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

# 配置日志
model_name = "/app/deberta_model"  # 预训练模型名称
# model_name = "microsoft/deberta-base"
max_length = 128  # 最大序列长度
projection_dim = 128  # 对比学习投影维度

# 训练参数
batch_size = 32  # 批量大小
epochs = 5  # 训练轮数
learning_rate = 1e-5  # 学习率
weight_decay = 0.01  # 权重衰减
scl_weight = 0.3  # 对比损失权重
temperature = 0.07  # 对比学习温度参数
num_workers = 4  # 数据加载线程数
log_dir = os.path.join("/app/logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "training.log")

# 创建一个logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建一个文件处理器，将日志保存到文件
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# 创建一个控制台处理器，将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建一个格式化器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 将处理器添加到logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"日志将保存到: {log_file}")
# 设置环境变量

# os.environ['NLTK_DATA'] = "https://mirrors.tuna.tsinghua.edu.cn/nltk_data/"

# 设置NLTK数据路径
nltk.data.path = []
nltk.data.path.append('/app/nltk_data')
logger.info(f"当前NLTK搜索路径：{nltk.data.path}")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

#########################################
#           Step 1: 定义数据集           #
#########################################

class StyleChangeDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=128):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        for filename in os.listdir(data_dir):
            if filename.startswith('problem-') and filename.endswith('.txt'):
                problem_id = filename.split('-')[1].split('.')[0]
                txt_path = os.path.join(data_dir, filename)
                truth_path = os.path.join(data_dir, f'truth-problem-{problem_id}.json')
                
                try:
                    with open(txt_path, 'r', newline='', encoding='utf-8') as f:
                        text = f.read().replace('\n', ' ')
                    sentences = nltk.sent_tokenize(text)
                    
                    with open(truth_path, 'r', encoding='utf-8') as f:
                        truth = json.load(f)
                    changes = truth['changes']
                    
                    # 调整检查条件，处理可能的异常数据
                    if len(sentences) != len(changes) + 1:
                        logger.warning(f"文档 {problem_id} 句子数 {len(sentences)} 与标注数 {len(changes)} 不匹配，已跳过")
                        continue  # 跳过该文档
                    
                    for i in range(len(changes)):
                        sent1 = sentences[i]
                        sent2 = sentences[i + 1]
                        self.samples.append((sent1, sent2, changes[i], problem_id, i))  # 添加问题ID和句对索引用于监督对比学习
                except Exception as e:
                    logger.error(f"处理文件 {filename} 时出错: {e}")
        
        logger.info(f"从 {data_dir} 加载了 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sent1, sent2, label, problem_id, pair_idx = self.samples[idx]
        encoding = self.tokenizer(
            sent1,
            sent2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float),
            'problem_id': problem_id,  # 用于监督对比学习的分组
            'pair_idx': pair_idx       # 用于监督对比学习的分组
        }

#########################################
#        Step 2: 定义损失函数            #
#########################################

class SupervisedContrastiveLoss(nn.Module):
    """
    监督对比损失函数
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # 特征归一化
        features = F.normalize(features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 为了数值稳定性排除自身相似度
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)
        
        # 构建标签匹配矩阵 - 相同标签为正样本对
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T).float()
        
        # 移除对角线上的元素
        mask_pos = mask_pos.masked_fill(mask, 0)
        
        # 计算每行有多少个正样本
        num_positives_per_row = mask_pos.sum(1)
        
        # 避免没有正样本的情况
        denominator = torch.exp(similarity_matrix).sum(dim=1, keepdim=True)
        log_probs = similarity_matrix - torch.log(denominator)
        
        # 只计算正样本对的损失
        if torch.any(num_positives_per_row > 0):
            mean_log_prob_pos = (mask_pos * log_probs).sum(1) / (num_positives_per_row + 1e-8)
            # 对有正样本的样本计算损失
            loss = -mean_log_prob_pos[num_positives_per_row > 0].mean()
        else:
            loss = torch.tensor(0.0).to(features.device)
            
        return loss

#########################################
#           Step 3: 定义模型             #
#########################################

class StyleChangeModelSCL(nn.Module):
    def __init__(self, deberta_model='microsoft/deberta-base', proj_dim=128):
        super().__init__()
        self.deberta = DebertaModel.from_pretrained(deberta_model,force_download=True)
        hidden_size = self.deberta.config.hidden_size
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # 特征投影头，用于对比学习
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

    def forward(self, input_ids, attention_mask, return_features=False):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # 使用[CLS]标记
        
        # 获取分类logits
        logits = self.classifier(pooled_output)
        
        if return_features:
            # 投影特征用于对比学习
            features = self.projection(pooled_output)
            return logits.squeeze(-1), features
        else:
            return logits.squeeze(-1)

#########################################
#        Step 4: 训练和评估函数          #
#########################################

def train_epoch(model, train_loader, optimizer, criterion_cls, criterion_scl, device, scl_weight=0.1):
    model.train()
    total_loss = 0
    total_scl_loss = 0
    total_cls_loss = 0
    
    progress_bar = tqdm(train_loader, desc="训练中")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播，获取logits和特征
        logits, features = model(input_ids, attention_mask, return_features=True)
        
        # 计算分类损失
        cls_loss = criterion_cls(logits, labels)
        
        # 计算对比损失
        scl_loss = criterion_scl(features, labels)
        
        # 组合损失
        loss = cls_loss + scl_weight * scl_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_scl_loss += scl_loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls_loss': f'{cls_loss.item():.4f}',
            'scl_loss': f'{scl_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_scl_loss = total_scl_loss / len(train_loader)
    
    return avg_loss, avg_cls_loss, avg_scl_loss

def evaluate(model, val_loader, criterion_cls, device):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="评估中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            logits = model(input_ids, attention_mask)
            
            # 计算损失
            loss = criterion_cls(logits, labels)
            val_loss += loss.item()
            
            # 计算预测
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    avg_loss = val_loss / len(val_loader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1, 
        'precision': precision,
        'recall': recall,
        'preds': all_preds,
        'labels': all_labels
    }

#########################################
#        Step 5: 预测函数                #
#########################################
def predict(test_df, model, tokenizer, device, max_length=128):
    model.eval()
    predictions = {}
    
    for _, row in test_df.iterrows():
        problem_id = os.path.basename(row['file']).split('-')[1].split('.')[0]
        text = ' '.join(row['paragraphs'])  # 合并段落为文本
        
        sentences = nltk.sent_tokenize(text)
        changes = []
        
        if len(sentences) >= 2:
            for i in range(len(sentences) - 1):
                sent1, sent2 = sentences[i], sentences[i+1]
                encoding = tokenizer(
                    sent1, sent2,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                    # truncation_strategy='only_first'
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                with torch.no_grad():
                    output = model(input_ids, attention_mask)
                    prob = torch.sigmoid(output).item()
                    changes.append(1 if prob > 0.5 else 0)
        
        predictions[problem_id] = changes
    return predictions

def save_predictions(predictions, output_dir, subtask):
    """Save predictions to separate folders for each subtask"""
    subtask_dir = os.path.join(output_dir, subtask)
    os.makedirs(subtask_dir, exist_ok=True)
    for problem_id, changes in predictions.items():
        with open(f'{subtask_dir}/solution-problem-{problem_id}.json', 'w') as f:
            json.dump({"changes": changes}, f)

#########################################
#           Step 6: 主函数               #
#########################################
@click.command()
@click.option('--dataset', default='multi-author-writing-style-analysis-2025/multi-author-writing-spot-check-20250503-training', help='The dataset to run predictions on (can point to a local directory).')
@click.option('--output', default=Path(get_output_directory(str(Path(__file__).parent)), help='The file where predictions should be written to.'))
@click.option('--model-dir', default='/app/model', help='Directory containing model files')
def main(dataset, output, model_dir):
    tokenizer = DebertaTokenizer.from_pretrained(model_name,force_download=True)
    model = StyleChangeModelSCL(model_name, proj_dim=projection_dim).to(device)
    tira = Client()
    
    # Load input data
    input_df = tira.pd.inputs(dataset, formats=["multi-author-writing-style-analysis-problems"])
    
    # Process each subtask separately
    subtasks = ["easy","medium", "hard"]
    for subtask in subtasks:
        logger.info(f"Processing subtask: {subtask}")
        
        # Load model for this subtask
        model_path = os.path.join(model_dir, f"{subtask}.pth") 
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Filter data for current subtask
        subtask_df = input_df[input_df["task"] == subtask]
        
        # Make predictions
        predictions = predict(subtask_df, model, tokenizer, device)
        
        # Save predictions to subtask-specific folder
        save_predictions(predictions, output, subtask)
        logger.info(f"Predictions for {subtask} saved to {os.path.join(output, subtask)}")

if __name__ == "__main__":
    main()
