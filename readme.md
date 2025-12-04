資料清理 (data_cleaning.py)：對原始影像進行亮度過濾和重複影像過濾，輸出乾淨的影像。

船隻偵測 (ship_detection.py)：在乾淨影像中偵測船隻，並裁切出船隻圖片，同時移除連續重複的截圖。

OCR 資料準備 (prepare_data.py)：利用 YOLO 標註檔，從原始圖片中精確裁切出被標註的文字區域，作為 OCR 的訓練輸入。

標籤生成 (prepare_ocr_labels.py)：解析裁切圖的檔名來獲取真實標籤，並將資料集分成訓練集和驗證集標籤檔案。

模型訓練 (train_universal_fix.py)：使用 TrOCR 模型，讀取標籤檔案和裁切圖，並透過圖像增強 (模糊和遮擋) 來訓練 OCR 模型。

模型應用 (predict_universal.py / demo_interactive.py)：使用訓練好的模型進行批次測試或互動式測試

已訓練好的模型連結:https://drive.google.com/file/d/1Llp9eqlKmdqn2C4LxE1AWLCWA11zWY4_/view

 直接執行demo_interactive.py
