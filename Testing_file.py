"""Importing libraries"""
import time
import torch
import torchvision
import numpy as np
import cv2
from torchvision.transforms.functional import to_tensor
start_time = time.time()
from torchvision.models.detection import fasterrcnn_resnet50_fpn
#Trained model_path and test_image_path
model_path = 'fasterrcnn_camouflage_ssim_value4.pth'
image_path = '3.jpg'
#Detection threshold (user_input)
try:
    score_threshold = float(input("Enter the detection threshold: "))
except ValueError:
    print("Invalid input")
    score_threshold = 0.4
#Load image
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_float = img_rgb.astype(np.float32) / 255.0
#Convert to tensor
image_tensor = to_tensor(img_float).unsqueeze(0)
original_img = img_rgb.copy()
#Load model
num_classes = 2
model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                           num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
#Inference
with torch.no_grad():
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
#Draw bounding box
def draw_boxes(image, boxes, scores, threshold=0.5):
    for i, box in enumerate(boxes):
        if scores[i] < threshold:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"camouflage: {scores[i]:.2f}"
        cv2.putText(image, label, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return image
#Output
output = outputs[0]
boxes = output['boxes'].cpu().numpy()
scores = output['scores'].cpu().numpy()
vis_img = draw_boxes(original_img, boxes, scores, threshold=score_threshold)
cv2.imwrite('output_image.jpg', cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
end_time = time.time()
print(f'Total time is {end_time-start_time} seconds')
