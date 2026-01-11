import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from PIL import Image

# --- CẤU HÌNH ---
MODEL_PATH = 'unet_isic_epoch_15.pth'  # Tên file model của bạn
IMAGE_PATH = 'R.jpg'          # Tên file ảnh muốn kiểm tra
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. DỰNG LẠI KIẾN TRÚC MODEL ---
# Phải khai báo giống hệt lúc train thì mới nạp được weight
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

# --- 2. NẠP WEIGHT TỪ FILE .PTH ---
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("✅ Đã nạp model thành công!")
except FileNotFoundError:
    print(f"❌ Không tìm thấy file {MODEL_PATH}. Kiểm tra lại tên file nhé!")
    exit()

# --- 3. HÀM XỬ LÝ ẢNH & DỰ ĐOÁN ---
def predict_image(image_path):
    # Đọc ảnh
    image = Image.open(image_path).convert("RGB")
    original_size = image.size # Lưu kích thước thật để sau này resize mask lại cho khớp
    
    # Tiền xử lý (Giống hệt lúc train)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0) # Thêm dimension batch: [1, 3, 256, 256]
    input_tensor = input_tensor.to(DEVICE)

    # Dự đoán
    with torch.no_grad(): # Không tính đạo hàm để tiết kiệm RAM
        output = model(input_tensor)
        
        # Chuyển về xác suất (Sigmoid) rồi thành nhị phân 0-1
        prob_mask = torch.sigmoid(output)
        pred_mask = (prob_mask > 0.5).float()

    # Chuyển từ Tensor về ảnh numpy để hiển thị
    pred_mask = pred_mask.squeeze().cpu().numpy() # [256, 256]
    pred_mask = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
    return image, pred_mask

print(f"🔍 Đang xử lý ảnh: {IMAGE_PATH}...")
try:
    original_img, mask = predict_image(IMAGE_PATH)
    plt.figure(figsize=(12, 6))
    
    # Ảnh gốc
    plt.subplot(1, 2, 1)
    plt.title("Ảnh gốc")
    plt.imshow(original_img)
    plt.axis('off')

    # Kết quả dự đoán
    plt.subplot(1, 2, 2)
    plt.title("AI Phân vùng (Segmentation)")
    plt.imshow(original_img)
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')

    plt.show()
    print("🎉 Xong! Hãy xem cửa sổ hình ảnh hiện lên.")
    
except FileNotFoundError:
    print(f"❌ Không tìm thấy ảnh {IMAGE_PATH}. Hãy tải 1 tấm ảnh về và đổi tên cho đúng.")