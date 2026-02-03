import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import os
import glob
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

# ==========================================
# 1. CẤU HÌNH (CONFIG)
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- ĐƯỜNG DẪN (Sửa lại cho đúng máy bạn) ---
SEG_MODEL_PATH = r"C:\Users\Huy\PythonMining\ImageProcessing\Detection&Segmentation\Segmentation\best_unet_ham10000_v2.pth"
DET_MODEL_PATH = r"C:\Users\Huy\PythonMining\ImageProcessing\Detection&Segmentation\Detection\best_model.pth"

INPUT_FOLDER = os.path.join(CURRENT_DIR, 'input_images')
OUTPUT_FOLDER = os.path.join(CURRENT_DIR, 'output_combined')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- THÔNG SỐ ---
DET_THRESHOLD = 0.2      # Ngưỡng nhạy 20%
NUM_CLASSES_DET = 8     # 7 bệnh + 1 nền
SEG_IMG_SIZE = 256       # Size model segmentation
TARGET_HEIGHT = 500      # Chiều cao ảnh đầu ra

# ==========================================
# 2. MODULE DETECTION (Đơn giản hóa)
# ==========================================
def load_detection_model(path):
    print(f"⏳ Loading Detection Model...")
    if not os.path.exists(path): return None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES_DET)
    try:
        checkpoint = torch.load(path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
        else: model.load_state_dict(checkpoint)
        model.to(DEVICE).eval()
        return model
    except: return None

def predict_detection_top1(model, image_rgb):
    """Chỉ lấy 1 box có điểm cao nhất"""
    if model is None: return []
    
    transform = T.Compose([T.ToTensor()]) 
    input_tensor = transform(image_rgb).to(DEVICE).unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_tensor)[0]

    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    if len(scores) == 0: return []
    
    # 1. Tìm điểm cao nhất
    best_idx = np.argmax(scores)
    
    # 2. Kiểm tra ngưỡng
    if scores[best_idx] < DET_THRESHOLD:
        return []
        
    # 3. Trả về đúng 1 box
    return boxes[best_idx : best_idx+1]

def draw_box_only(original_img, boxes):
    """Chỉ vẽ khung đỏ, không chữ"""
    img_draw = original_img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        # Vẽ khung màu ĐỎ, nét dày 2
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img_draw 

# ==========================================
# 3. MODULE SEGMENTATION (Giữ nguyên)
# ==========================================
def load_segmentation_model(path):
    print(f"⏳ Loading Segmentation Model...")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE).eval()
            return model
        except: return None
    return None

def predict_segmentation(model, image_rgb, original_size):
    transform = A.Compose([
        A.Resize(SEG_IMG_SIZE, SEG_IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    input_tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        mask = (torch.sigmoid(output) > 0.5).float().squeeze().cpu().numpy()
    return cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

def draw_segmentation(original_img, mask):
    overlay = original_img.copy()
    colored_mask = np.zeros_like(original_img)
    colored_mask[:, :, 1] = 255 # Xanh lá
    overlay[mask > 0] = cv2.addWeighted(original_img[mask > 0], 0.6, colored_mask[mask > 0], 0.4, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay

# ==========================================
# 4. HÀM HIỂN THỊ
# ==========================================
def resize_maintain_aspect(image, target_height):
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)

def add_header(img, text, bg_color=(0,0,0)):
    h, w = img.shape[:2]
    header = np.zeros((40, w, 3), dtype=np.uint8)
    header[:] = bg_color
    cv2.putText(header, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return np.vstack([header, img])

# ==========================================
# 5. MAIN PIPELINE
# ==========================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    
    seg_model = load_segmentation_model(SEG_MODEL_PATH)
    det_model = load_detection_model(DET_MODEL_PATH)
    
    image_paths = glob.glob(os.path.join(INPUT_FOLDER, "*.*"))
    print(f"\n🚀 Đang xử lý {len(image_paths)} ảnh...")

    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        print(f"[{idx+1}/{len(image_paths)}] : {filename}")
        
        original_img = cv2.imread(img_path)
        if original_img is None: continue
        
        image_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        h, w = original_img.shape[:2]
        
        # 1. Segmentation
        if seg_model:
            mask = predict_segmentation(seg_model, image_rgb, (h, w))
            img_seg = draw_segmentation(original_img, mask)
        else:
            img_seg = original_img.copy()
            
        # 2. Detection 
        if det_model:
            boxes = predict_detection_top1(det_model, image_rgb)
            img_det = draw_box_only(original_img, boxes)
        else:
            img_det = original_img.copy()

        # 3. Resize & Ghép
        final_orig = resize_maintain_aspect(original_img, TARGET_HEIGHT)
        final_det  = resize_maintain_aspect(img_det, TARGET_HEIGHT)
        final_seg  = resize_maintain_aspect(img_seg, TARGET_HEIGHT)
        
        final_orig = add_header(final_orig, "Input")
        final_det  = add_header(final_det, "Detection", bg_color=(0, 0, 150)) # Đầu đỏ
        final_seg  = add_header(final_seg, "Segmentation", bg_color=(0, 100, 0)) # Đầu xanh
        
        combined = np.hstack([final_orig, final_det, final_seg])
        
        # 4. Lưu
        save_path = os.path.join(OUTPUT_FOLDER, f"Result_{filename}")
        cv2.imwrite(save_path, combined)
        
        # 5. Hiển thị
        MAX_W = 1400
        if combined.shape[1] > MAX_W:
            scale = MAX_W / combined.shape[1]
            h_view = int(combined.shape[0] * scale)
            combined_view = cv2.resize(combined, (MAX_W, h_view))
        else:
            combined_view = combined
            
        cv2.imshow("Result", combined_view)
        if cv2.waitKey(0) == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"\n✅ Đã xong! File lưu tại: {OUTPUT_FOLDER}")