import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torch
import numpy as np
import random
import evaluate
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    default_data_collator
)

# ================= 設定區域 =================
data_root = r'C:\SHIPS\VIT2\dataset_recognition'
train_label_file = os.path.join(data_root, 'train_labels.txt')
val_label_file = os.path.join(data_root, 'val_labels.txt')

# 輸出路徑 (通用修復版)
output_dir = r'C:\SHIPS\VIT2\output_model_universal'

# 參數設定
BATCH_SIZE = 2        # Base 模型比較大，Batch 改小避免記憶體不足
EPOCHS = 30           # 30 輪足夠讓 Base 模型適應
LEARNING_RATE = 1e-5  # [關鍵] 低學習率，保持穩定
# ===========================================

def load_data(label_file):
    try:
        df = pd.read_csv(label_file, sep='\t', header=None, names=['file_name', 'text'], dtype=str)
        df = df.dropna()
        df['file_name'] = df['file_name'].apply(lambda x: os.path.join(data_root, x))
        return df
    except Exception as e:
        print(f"讀取失敗: {e}")
        return pd.DataFrame(columns=['file_name', 'text'])

class ShipDataset(Dataset):
    def __init__(self, df, processor, augment=False):
        self.df = df
        self.processor = processor
        self.augment = augment 

    def apply_augmentation(self, image):
        """
        [關鍵修復] 這是實作文件要求的圖像增強策略
        如果少了這個，模型就無法學會「腦補」被遮擋的字。
        """
        rand_val = random.random()
        
        # 1. 15% 機率模糊 (模擬對焦不準)
        if rand_val < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
        
        # 2. 15% 機率遮擋 (模擬小畫家塗改/障礙物) 
        # 這是讓模型學會認出 "Or...ed" -> "Orsted" 的關鍵
        elif rand_val < 0.30:
            img_np = np.array(image)
            h, w, _ = img_np.shape
            # 隨機產生一個遮罩大小
            mask_size = random.randint(8, 20) 
            x = random.randint(0, max(0, w - mask_size))
            y = random.randint(0, max(0, h - mask_size))
            # 將該區域塗黑
            img_np[y:y+mask_size, x:x+mask_size] = 0 
            image = Image.fromarray(img_np)
            
        # 剩下 70% 的時間保持原圖，讓模型看清楚標準答案

        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'].iloc[idx]
        text = self.df['text'].iloc[idx]
        
        try:
            image = Image.open(file_name).convert("RGB")
            # [關鍵] 只有在訓練模式 (augment=True) 才會啟動上述的破壞性增強
            if self.augment:
                image = self.apply_augmentation(image)
        except:
            # 萬一讀不到圖，給一張白紙避免程式崩潰
            image = Image.new('RGB', (384, 384), (255, 255, 255))
            
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        labels = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=32,
            truncation=True
        ).input_ids
        
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]

        return {
            "pixel_values": pixel_values.squeeze(), 
            "labels": torch.tensor(labels)
        }