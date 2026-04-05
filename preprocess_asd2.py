import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from pyts.image import GramianAngularField

# ================= 配置 =================
SOURCE_DIR = "AU-ASD-TD"
OUTPUT_DIR = "AU-ASD-TD-GASF"
IMAGE_SIZE = 64

AU_COLS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
    'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
    'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
]

REQUIRED_FILES = ['A1.csv', 'A2.csv', 'C.csv', 'D.csv']

def interpolate_seq(data, target_size):
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
    interpolated = F.interpolate(data_tensor, size=target_size, mode='linear', align_corners=False)
    return interpolated.squeeze(0).transpose(0, 1).numpy()

def preprocess():
    transformer = GramianAngularField(image_size=IMAGE_SIZE, method='summation')
    scaler = MinMaxScaler(feature_range=(-1, 1))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for group in ["ASD", "TD"]:
        group_src = os.path.join(SOURCE_DIR, group)
        group_out = os.path.join(OUTPUT_DIR, group)
        os.makedirs(group_out, exist_ok=True)

        subjects = [s for s in os.listdir(group_src) if os.path.isdir(os.path.join(group_src, s))]
        print(f"📦 正在处理 {group} 组，共 {len(subjects)} 个样本...")

        for sub in tqdm(subjects):
            sub_path = os.path.join(group_src, sub)
            subject_tensors = []
            skip_subject = False
            
            for file_name in REQUIRED_FILES:
                file_path = os.path.join(sub_path, file_name)
                if not os.path.exists(file_path):
                    print(f"⚠️ 缺失文件: {file_path}")
                    skip_subject = True
                    break
                
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
                valid_df = df[(df['success'] == 1) & (df['confidence'] >= 0.8)]
                
                if len(valid_df) == 0:
                    subject_tensors.append(torch.zeros((len(AU_COLS), IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32))
                    continue
                
                data = valid_df[AU_COLS].values.astype(np.float32)
                data = interpolate_seq(data, IMAGE_SIZE)
                seq_normalized = scaler.fit_transform(data)
                
                try:
                    images = transformer.fit_transform(seq_normalized.T)
                    subject_tensors.append(torch.tensor(images, dtype=torch.float32))
                except Exception as e:
                    print(f"\n⚠️ 转换失败 ({sub} - {file_name}): {e}")
                    subject_tensors.append(torch.zeros((len(AU_COLS), IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32))

            if not skip_subject:
                final_tensor = torch.cat(subject_tensors, dim=0) 
                save_path = os.path.join(group_out, f"{sub}.pt")
                torch.save(final_tensor, save_path)

    print(f"✅ 预处理完成！GASF 数据已保存在: {OUTPUT_DIR}")

if __name__ == '__main__':
    preprocess()