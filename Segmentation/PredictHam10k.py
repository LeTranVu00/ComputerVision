import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import os
import glob

# ==========================================
# 1. CẤU HÌNH
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'best_unet_ham10000_v2.pth' # Tên file model của bạn
MODEL_PATH = os.path.join(CURRENT_DIR, MODEL_NAME)

INPUT_FOLDER = os.path.join(CURRENT_DIR, 'input_images')
OUTPUT_FOLDER = os.path.join(CURRENT_DIR, 'output_results')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256

# ==========================================
# 2. CÁC HÀM XỬ LÝ
# ==========================================
# Hàm nạp model đã huấn luyện
def load_model(path):
    print(f"⏳ Đang nạp model từ: {path}")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    if not os.path.exists(path):
        print(f"❌ LỖI: Không tìm thấy file model tại {path}")
        return None
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Hàm dự đoán mask cho một ảnh
def predict_image(model, img_path):
    image = cv2.imread(img_path)
    if image is None: return None, None
    
    original_img = image.copy()
    original_h, original_w = image.shape[:2]
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    augmented = transform(image=image_rgb)
    input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob_mask = torch.sigmoid(output)
        pred_mask = (prob_mask > 0.5).float().squeeze().cpu().numpy()
    
    pred_mask_resized = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    return original_img, pred_mask_resized

# Hàm tạo ảnh chồng lớp (ảnh kết quả)
def create_overlay(original_img, mask):
    # Tạo overlay để lưu và hiển thị
    colored_mask = np.zeros_like(original_img)
    colored_mask[:, :, 1] = 255 # Màu xanh lá
    
    overlay = original_img.copy()
    overlay[mask > 0] = cv2.addWeighted(original_img[mask > 0], 0.6, colored_mask[mask > 0], 0.4, 0)
    
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    return overlay

# ==========================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"⚠️ Đã tạo thư mục '{INPUT_FOLDER}'. Hãy bỏ ảnh vào đó rồi chạy lại.")
        exit()
        
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Lấy danh sách ảnh
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    
    if not image_paths:
        print("❌ Không thấy ảnh nào!")
        exit()

    model = load_model(MODEL_PATH)
    
    if model is not None:
        print(f"🚀 Bắt đầu xử lý {len(image_paths)} ảnh...")
        print("💡 HƯỚNG DẪN: Nhấn phím bất kỳ để qua ảnh tiếp theo. Nhấn 'q' để thoát.")

        for i, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            print(f"[{i+1}/{len(image_paths)}] Đang xử lý: {filename}")
            
            # 1. Dự đoán
            img, mask = predict_image(model, img_path)
            
            if img is not None:
                # 2. Tạo kết quả chồng lớp
                overlay = create_overlay(img, mask)
                
                # 3. Lưu xuống ổ cứng
                save_path = os.path.join(OUTPUT_FOLDER, f"result_{filename}")
                cv2.imwrite(save_path, overlay)
            
                # Resize về cùng kích thước nhỏ (ví dụ 300x300) để ghép cho đẹp
                view_size = (300, 300)
                v_orig = cv2.resize(img, view_size)
                
                # Mask cần chuyển từ xám sang màu để ghép
                v_mask = cv2.resize(mask, view_size)
                v_mask = (v_mask * 255).astype(np.uint8) # Chuyển 0-1 thành 0-255
                v_mask = cv2.cvtColor(v_mask, cv2.COLOR_GRAY2BGR) 
                
                v_over = cv2.resize(overlay, view_size)
                
                # Ghép 3 ảnh nằm ngang (Horizontal Stack)
                combined_view = np.hstack([v_orig, v_mask, v_over])
                
                # Thêm chữ chú thích
                cv2.putText(combined_view, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(combined_view, "AI Mask", (310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(combined_view, "Result", (610, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # 5. Hiện cửa sổ
                cv2.imshow("Skin Lesion Analysis (Nhan phim bat ky de tiep tuc, 'q' de thoat)", combined_view)
                
               
                key = cv2.waitKey(0) 
                if key == ord('q'): 
                    print("🛑 Đã dừng chương trình.")
                    break

        cv2.destroyAllWindows()
        print("\n✅ Hoàn tất!")