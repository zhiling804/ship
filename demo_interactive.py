import cv2
import torch
import numpy as np
import os
import glob
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ================= 設定區域 =================
model_path = r'C:\SHIPS\VIT2\output_model_universal'

image_folder = r'C:/SHIPS\VIT2\step2_ship_crops'
# ===========================================

def cv2_imread_win(file_path):
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    except Exception:
        return None

def main():
    # 1. 載入模型
    print(f"正在載入模型: {model_path} ...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
        model.eval()
        print(f"模型載入成功！")
    except Exception as e:
        print(f"載入失敗: {e}")
        return

    # 2. 搜尋所有圖片
    valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    if os.path.isdir(image_folder):
        for ext in valid_exts:
            images.extend(glob.glob(os.path.join(image_folder, ext)))
    else:
        # 如果使用者指到單張圖，就只測那張
        images = [image_folder]

    if not images:
        print("找不到圖片！")
        return

    print(f"-> 共找到 {len(images)} 張圖片。")
    print("\n" + "="*50)
    print("【操作說明】")
    print("1. 畫框 + Enter   : 辨識選取區域")
    print("2. 不畫框 + Enter : 跳至「下一張」圖片")
    print("3. Esc            : 退出程式")
    print("="*50 + "\n")

    # 3. 開始迴圈 (一張一張開)
    for i, img_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] 開啟: {os.path.basename(img_path)}")
        
        img = cv2_imread_win(img_path)
        if img is None: continue

        # 縮放顯示 (避免爆出螢幕)
        scale_percent = 100
        if img.shape[1] > 1280:
            scale_percent = 1280 / img.shape[1] * 100
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            img_display = cv2.resize(img, (width, height))
        else:
            img_display = img.copy()

        # 單張圖片內的迴圈
        while True:
            # 開啟選取視窗
            window_name = f"Image {i+1}/{len(images)} (Enter to Next, Draw to OCR)"
            rect = cv2.selectROI(window_name, img_display, showCrosshair=True)
            cv2.destroyWindow(window_name)

            # 如果使用者沒畫框就按 Enter，rect 會是 (0,0,0,0)
            # 這時候我們就跳出內部迴圈，進入下一張圖
            if rect == (0, 0, 0, 0):
                print("-> 下一張...")
                break 

            # 還原座標並裁切
            x, y, w, h = rect
            if scale_percent != 100:
                x, y, w, h = int(x*100/scale_percent), int(y*100/scale_percent), int(w*100/scale_percent), int(h*100/scale_percent)

            crop_img = img[y:y+h, x:x+w]
            if crop_img.size == 0: continue

            # 辨識
            pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            pixel_values = processor(pil_img, return_tensors="pt").pixel_values.to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, num_beams=5, max_length=32)
            
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"辨識結果: {text}")

            # 顯示結果
            cv2.imshow(f"Result: {text}", crop_img)
            key = cv2.waitKey(0) # 等待按鍵
            cv2.destroyAllWindows()

            # 如果在結果視窗按 Esc，直接結束整個程式
            if key == 27: 
                print("程式結束。")
                return

    print("已瀏覽完所有圖片！")

if __name__ == '__main__':
    main()