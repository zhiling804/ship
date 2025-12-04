import os
import glob
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ================= 設定區域 =================
# 指向剛剛練好的「通用修復版」模型
model_path = r'C:\SHIPS\VIT2\output_model_universal'
image_folder = r'C:\SHIPS\VIT2\dataset_recognition'
# ===========================================

def main():
    print(f"正在載入通用修復版模型: {model_path} ...")
    try:
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
    except Exception as e:
        print(f"載入失敗: {e}")
        return

    print("模型載入成功！開始測試...")
    print("-" * 50)
    print(f"{'檔名':<30} | {'預測結果':<30}")
    print("-" * 50)

    # 抓取所有 jpg 圖片
    images = glob.glob(os.path.join(image_folder, "*.jpg"))
    model.eval()

    for img_path in images:
        try:
            image = Image.open(img_path).convert("RGB")
            
            # 預處理
            pixel_values = processor(image, return_tensors="pt").pixel_values
            
            # 預測 (使用 Beam Search 增加準確度)
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, num_beams=5, max_length=32)
            
            # 解碼
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            filename = os.path.basename(img_path)
            print(f"{filename:<30} | {generated_text:<30}")

        except Exception as e:
            print(f"錯誤: {e}")

    print("-" * 50)
    print("測試結束！")

if __name__ == '__main__':
    main()