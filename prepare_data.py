import os
import shutil
import glob
import cv2
import numpy as np # 需要安裝 numpy

# ================= 路徑設定 =================
images_root_path = r'C:\SHIPS\VIT2\20251125_船名船號'
txt_path = r'C:\SHIPS\VIT2\annotations'
output_rec_path = r'C:\SHIPS\VIT2\dataset_recognition'
# ===========================================

def cv2_imread_win(file_path):
    """
    專門解決 Windows 下 OpenCV 無法讀取中文路徑的問題
    """
    try:
        # 使用 numpy 先讀取成二進制，再用 opencv 解碼
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    except Exception as e:
        print(f"讀取錯誤: {e}")
        return None

def cv2_imwrite_win(file_path, img):
    """
    專門解決 Windows 下 OpenCV 無法寫入中文路徑的問題
    """
    try:
        cv2.imencode(os.path.splitext(file_path)[1], img)[1].tofile(file_path)
        return True
    except Exception as e:
        print(f"寫入錯誤: {e}")
        return False

def main():
    print("===== 啟動中文路徑支援模式 =====")
    
    # 1. 建立圖片索引
    image_map = {}
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    for root, dirs, files in os.walk(images_root_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_exts:
                basename = os.path.splitext(file)[0]
                image_map[basename] = os.path.join(root, file)
    print(f"-> 索引建立完成，共有 {len(image_map)} 張圖片。")

    # 2. 建立輸出資料夾
    if not os.path.exists(output_rec_path):
        os.makedirs(output_rec_path)

    # 3. 讀取類別 (從您的截圖看到您的 classes.txt 包含很多雜項，這裡自動讀取)
    classes_file = os.path.join(txt_path, 'classes.txt')
    classes = []
    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        print(f"-> 使用 classes.txt 定義的 {len(classes)} 個類別")
    else:
        print("-> 警告：找不到 classes.txt")
        return

    # 4. 處理
    txt_files = glob.glob(os.path.join(txt_path, '*.txt'))
    txt_files = [f for f in txt_files if 'classes.txt' not in f]

    print(f"-> 開始處理 {len(txt_files)} 個標註檔...")
    success_count = 0

    for txt_file in txt_files:
        basename = os.path.basename(txt_file).replace('.txt', '')
        
        if basename not in image_map:
            print(f"  [X] 跳過：找不到圖片 {basename}")
            continue
        
        img_src = image_map[basename]
        
        # [修正點] 使用支援中文的讀取函式
        img = cv2_imread_win(img_src)
        
        if img is None:
            print(f"  [X] 失敗：無法讀取圖片 {img_src}")
            continue
        
        h_img, w_img, _ = img.shape

        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            try:
                cls_id = int(parts[0])
                if cls_id >= len(classes): continue
                cls_name = classes[cls_id]
                
                n_x, n_y, n_w, n_h = map(float, parts[1:5])
                x_center, y_center = n_x * w_img, n_y * h_img
                box_w, box_h = n_w * w_img, n_h * h_img
                
                xmin = int(x_center - box_w/2)
                ymin = int(y_center - box_h/2)
                xmax = int(x_center + box_w/2)
                ymax = int(y_center + box_h/2)

                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w_img, xmax), min(h_img, ymax)

                crop_img = img[ymin:ymax, xmin:xmax]
                
                if crop_img.size > 0:
                    save_name = f"{basename}_{cls_name}_{i}.jpg"
                    save_path = os.path.join(output_rec_path, save_name)
                    
                    # [修正點] 使用支援中文的寫入函式
                    cv2_imwrite_win(save_path, crop_img)
                    success_count += 1
            
            except Exception as e:
                print(f"  [!] 錯誤: {e}")

    print("\n" + "="*30)
    print(f"處理完成！共裁切出 {success_count} 張圖片。")
    print(f"請打開資料夾: {output_rec_path}")
    print("="*30)

if __name__ == '__main__':
    main()