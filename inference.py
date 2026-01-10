import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# =============================
# CONFIG
# =============================
MODEL_PATH = "faster_rcnn_isic2018.pth"
IMAGE_PATH = "test.jpg"
SCORE_THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# LOAD MODEL
# =============================
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

num_classes = 2  # background + lesion
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =============================
# LOAD & PREPROCESS IMAGE
# =============================
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_tensor = torch.tensor(image / 255.0, dtype=torch.float32)
image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

# =============================
# INFERENCE
# =============================
with torch.no_grad():
    prediction = model(image_tensor)[0]

# =============================
# VISUALIZE
# =============================
for box, score in zip(prediction["boxes"], prediction["scores"]):
    if score < SCORE_THRESHOLD:
        continue

    x1, y1, x2, y2 = box.int().tolist()
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(
        image,
        f"{score:.2f}",
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1
    )

plt.figure(figsize=(6,6))
plt.imshow(image)
plt.axis("off")
plt.show()
