import os
import random
import glob

# ================= 設定區域 =================
# 您的裁切圖資料夾
data_root = r'C:\SHIPS\VIT2\dataset_recognition'
# 輸出檔名
train_label_file = os.path.join(data_root, 'train_labels.txt')
val_label_file = os.path.join(data_root, 'val_labels.txt')
split_ratio = 0.8  # 80% 訓練
# ===========================================

def parse_label_from_filename(filename):
    """
    檔名解析邏輯：
    1. 去除副檔名 (.jpg)
    2. 去除重複編號 (_1, _2...)
    3. 將底線 (_) 轉回空白鍵 ( ) -> 針對英文船名
    """
    basename = os.path.splitext(filename)[0]
    
    # 1. 處理重複編號 (如果結尾是 _數字，就切掉)
    # 例如: ORSTED_1 -> ORSTED
    if '_' in basename and basename.split('_')[-1].isdigit():
        basename = '_'.join(basename.split('_')[:-1])
    
    # 2. 處理空白鍵 (將底線轉回空白)
    # 例如: MP_EURUS -> MP EURUS
    # 注意：如果您的船名本身就包含底線，請註解掉下面這行
    label = basename.replace('_', ' ')
    
    return label

def main():
    print("===== 開始製作標籤索引 =====")
    
    # 支援的圖片格式
    valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_images = []
    
    for ext in valid_exts:
        all_images.extend(glob.glob(os.path.join(data_root, ext)))
    
    if len(all_images) == 0:
        print(f"錯誤：在 {data_root} 找不到任何圖片！")
        return

    # 隨機打亂
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * split_ratio)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]
    
    print(f"總共找到 {len(all_images)} 張圖片。")
    print(f"訓練集: {len(train_imgs)} 張")
    print(f"驗證集: {len(val_imgs)} 張")

    # 寫入 txt 檔案
    # 格式: 檔名 <TAB> 真實文字
    
    def write_labels(file_path, image_list):
        with open(file_path, 'w', encoding='utf-8') as f:
            for img_path in image_list:
                filename = os.path.basename(img_path)
                label = parse_label_from_filename(filename)
                # 寫入相對路徑與標籤，中間用 Tab 分隔
                f.write(f"{filename}\t{label}\n")
    
    write_labels(train_label_file, train_imgs)
    write_labels(val_label_file, val_imgs)

    print("-" * 30)
    print("處理完成！已生成以下兩個檔案：")
    print(f"1. {train_label_file}")
    print(f"2. {val_label_file}")
    print("-" * 30)
    print("檢查範例 (前 3 筆)：")
    with open(train_label_file, 'r', encoding='utf-8') as f:
        for i in range(3):
            print(f.readline().strip())
    print("=" * 30)

if __name__ == '__main__':
    main()