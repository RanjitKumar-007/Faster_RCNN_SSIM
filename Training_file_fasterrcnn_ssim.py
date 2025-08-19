"""Importing libraries"""
import os
import json
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.nn.functional as F
class CocoFasterRCNNDataset(Dataset):
    """Dataset class"""
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)
        self.image_map = {img['id']: img for img in self.coco['images']}
        self.image_list = list(self.image_map.values())
        self.annotations = {}
        for ann in self.coco['annotations']:
            self.annotations.setdefault(ann['image_id'], []).append(ann)
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        img_info = self.image_list[idx]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        image_id = img_info['id']
        anns = self.annotations.get(image_id, [])
        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([image_id]),
            'area': torch.tensor(areas, dtype=torch.float32),
            'iscrowd': torch.tensor(iscrowd, dtype=torch.uint8)
        }
        if self.transforms:
            img = self.transforms(img)
        return img, target
def ssim_loss(img1, img2, window_size=11, C1=0.01 ** 2, C2=0.03 ** 2):
    """Compute SSIM loss between two image tensors"""
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, window_size//2) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, window_size//2) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size//2) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1 - ssim_map.mean()
# Main training script
if __name__ == '__main__':
    train_dir = 'dataset/train'
    valid_dir = 'dataset/valid'
    train_dataset = CocoFasterRCNNDataset(
        root=train_dir,
        annotation_file=os.path.join(train_dir, '_annotations.fixed.json'),
        transforms=ToTensor()
    )
    valid_dataset = CocoFasterRCNNDataset(
        root=valid_dir,
        annotation_file=os.path.join(valid_dir, '_annotations.fixed.json'),
        transforms=ToTensor()
    )
    train_loader = DataLoader(train_dataset, batch_size=2,
                              shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_dataset, batch_size=2,
                              shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    num_classes = 2
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                               num_classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    num_epochs = 10
    lambda_ssim = 0.4
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            detection_loss = sum(loss for loss in loss_dict.values())
            ssim_total = 0.0
            with torch.no_grad():
                for img, tgt in zip(images, targets):
                    for box in tgt['boxes']:
                        x1, y1, x2, y2 = box.int()
                        crop_gt = img[:, y1:y2, x1:x2].unsqueeze(0)
                        crop_pred = F.interpolate(crop_gt, scale_factor=0.9,
                                                  mode='bilinear', align_corners=False)
                        crop_pred = F.interpolate(crop_pred, size=(crop_gt.shape[2],
                                                                   crop_gt.shape[3]),
                                                  mode='bilinear', align_corners=False)
                        if crop_gt.numel() > 0:
                            ssim_total += ssim_loss(crop_gt, crop_pred)
            total_loss_step = detection_loss + lambda_ssim * ssim_total
            optimizer.zero_grad()
            total_loss_step.backward()
            optimizer.step()
            total_loss += total_loss_step.item()
        print(f"\n[Epoch {epoch+1}/{num_epochs}] Training Loss: {total_loss:.4f}")
        # Validation
        model.eval()
        print(f"[Epoch {epoch+1}] Validation Prediction Summary:")
        with torch.no_grad():
            for i, (images, targets) in enumerate(valid_loader):
                images = [img.to(device) for img in images]
                outputs = model(images)
                for j, output in enumerate(outputs):
                    boxes = output['boxes'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    print(f"  Val sample {i*len(images)+j+1}: {len(boxes)} boxes, top score = {scores[0]:.2f}"
                          if len(scores) > 0 else "  No detections")
    torch.save(model.state_dict(), 'fasterrcnn_camouflage_ssim.pth')
    print("Model saved with SSIM loss")
