import torch
import segmentation_models_pytorch as smp # Cần thư viện này để dựng lại khung xương

# 1. Cấu hình thiết bị
device = torch.device('cpu') # Chạy trên CPU để xuất file cho tiện

# 2. DỰNG LẠI KIẾN TRÚC MODEL (Bước quan trọng nhất bị thiếu)
# Phải khai báo y hệt lúc train: backbone resnet34, classes=1
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None, # Không cần weight imagenet vì ta sắp nạp weight của mình
    in_channels=3,
    classes=1
)

# 3. Nạp trọng số (Weights) từ file .pth vào khung xương
try:
    # Load file pth (chứa dictionary)
    state_dict = torch.load('unet_isic_epoch_15.pth', map_location=device)
    # Nạp vào model
    model.load_state_dict(state_dict)
    # Chuyển sang chế độ đánh giá (quan trọng để tắt Dropout/BatchNorm động)
    model.eval() 
    print("✅ Đã nạp weights thành công!")
except Exception as e:
    print(f"❌ Lỗi nạp model: {e}")
    exit()

# 4. Tạo dữ liệu giả (Dummy input)
# Input đúng cú pháp: (Batch_size, Channels, Height, Width)
dummy_input = torch.randn(1, 3, 256, 256).to(device)

# 5. Xuất ra ONNX
onnx_path = "unet_model.onnx"

try:
    torch.onnx.export(
        model,               # Model đã nạp weight
        dummy_input,         # Input giả
        onnx_path,           # Tên file xuất
        verbose=False,
        input_names=['input_image'],  # Tên biến đầu vào (để vẽ đồ thị cho đẹp)
        output_names=['output_mask'], # Tên biến đầu ra
        opset_version=11     # Phiên bản ONNX (11 là bản ổn định nhất)
    )
    print(f"🎉 Đã tạo file {onnx_path} thành công! Hãy upload lên Netron.app để xem.")
except Exception as e:
    print(f"❌ Lỗi xuất ONNX: {e}")