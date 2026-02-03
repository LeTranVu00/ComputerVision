from torchinfo import summary
import segmentation_models_pytorch as smp

# 1. Load model
model = smp.Unet(encoder_name="resnet34", classes=1)

# 2. Hiển thị tóm tắt (Chỉ hiện đến cấp độ khối, không hiện từng lớp Conv nhỏ)
summary(model, input_size=(1, 3, 256, 256), depth=3, 
        col_names=["input_size", "output_size", "num_params"])