

from ultralytics import YOLO
import torch

# Load models (segmentation model + another pretrained)
m1 = YOLO('yolo11n-seg.pt').model
m2 = YOLO('best.pt').model

# Combine class names
names1 = list(m1.names.values()) if isinstance(m1.names, dict) else list(m1.names)
names2 = list(m2.names.values()) if isinstance(m2.names, dict) else list(m2.names)
combined_classes = sorted(set(names1) | set(names2))

# Create combined model using the segmentation checkpoint architecture (keep its structure)
combined = YOLO('yolo11n-seg.pt').model

# Update metadata only
combined.names = {i: name for i, name in enumerate(combined_classes)}
# If the model records nc on a detection layer, update that value as well
if hasattr(combined, 'nc'):
    combined.nc = len(combined_classes)
# Many YOLO variants store nc on the last model block
try:
    combined.model[-1].nc = len(combined_classes)
except Exception:
    pass

# Load weights permissively (do not overwrite incompatible shapes)
w_combined = combined.state_dict()
w1 = m1.state_dict()
w2 = m2.state_dict()

# Merge weights conservatively: prefer weights from m1, add m2 when shapes match
for k in list(w_combined.keys()):
    if k in w1 and w1[k].shape == w_combined[k].shape:
        w_combined[k] = w1[k].clone()
    if k in w2 and w2[k].shape == w_combined[k].shape:
        # simple average if both match
        if k in w1 and w1[k].shape == w2[k].shape:
            w_combined[k] = (w1[k].float() + w2[k].float()) / 2.0
        else:
            w_combined[k] = w2[k].clone()

merge= combined.load_state_dict(w_combined, strict=False)

y=YOLO(merge)
y.save('Y')

# IMPORTANT: now fine-tune the combined model on a dataset that includes all combined_classes
# Use ultralytics training script/command (example):
# combined_yolo = YOLO(combined)  # wrap model again if required by your ultralytics version
# combined_yolo.train(data='data_combined.yaml', epochs=50, imgsz=640)