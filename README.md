# Object Detection with Faster R-CNN and ResNet50

This project demonstrates **real-time-style object detection** using a pre-trained **Faster R-CNN** model with a **ResNet50 + FPN backbone**.

The model is pre-trained on the **Microsoft COCO dataset** and can detect multiple objects in a single image, returning:

- Object class
- Confidence score
- Bounding box coordinates

The project is implemented using **PyTorch** and **Torchvision** and is designed to run in **Google Colab**, with GPU acceleration when available.

---

# 📌 Project Overview

Unlike image classification, where a model predicts what an entire image contains, **object detection** identifies:

1. **What objects are present**
2. **Where those objects are located**

For example, an image containing a person and a car could produce:

```text
Person → 97.4%
Bounding Box → [120, 80, 420, 600]

Car → 91.8%
Bounding Box → [450, 300, 900, 550]
```

The model therefore provides both **classification** and **localization**.

---

# 🧠 Object Detection Pipeline

The overall workflow is:

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ▼
Faster R-CNN
     │
     ├── ResNet50 Backbone
     │
     ├── Feature Pyramid Network
     │
     ├── Region Proposal Network
     │
     └── ROI Detection
     │
     ▼
Object Predictions
     │
     ├── Bounding Boxes
     ├── Class Labels
     └── Confidence Scores
     │
     ▼
Filter by Confidence
     │
     ▼
Display Detected Objects
```

---

# 🛠️ Technologies Used

This project uses:

- **Python**
- **PyTorch**
- **Torchvision**
- **Faster R-CNN**
- **ResNet50**
- **Feature Pyramid Network (FPN)**
- **Microsoft COCO**
- **PIL**
- **NumPy**
- **Matplotlib**
- **Google Colab**
- **CUDA / GPU acceleration**

---

# 📦 Model

The project uses the following pre-trained model:

```text
Faster R-CNN
      +
ResNet50
      +
Feature Pyramid Network (FPN)
```

The model is loaded directly from Torchvision:

```python
model = models.detection.fasterrcnn_resnet50_fpn(
    weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)
```

The model is then moved to the available device:

```python
model = model.to(device)
```

and switched to evaluation mode:

```python
model.eval()
```

---

# 🧠 Why Faster R-CNN?

Faster R-CNN is an object detection architecture designed to identify multiple objects within an image while also determining their locations.

Unlike a simple classifier:

```text
Image
  ↓
Class
```

an object detector produces:

```text
Image
  ↓
┌──────────────────────────────────┐
│ Object 1 → Class + Bounding Box │
│ Object 2 → Class + Bounding Box │
│ Object 3 → Class + Bounding Box │
└──────────────────────────────────┘
```

This makes it suitable for images containing multiple objects.

---

# 🏗️ Model Architecture

The model used in this project can be represented as:

```text
                 Input Image
                      │
                      ▼
                  ResNet50
                      │
                      ▼
           Feature Pyramid Network
                      │
                      ▼
        Region Proposal Network
                      │
                      ▼
            Region of Interest
                   Detection
                      │
                      ▼
              Object Predictions
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Boxes        Labels      Scores
```

---

# 🚀 Step-by-Step Implementation

## Step 1: Environment Setup

The notebook begins by importing the required libraries:

```python
import torch
import torchvision

from torchvision import transforms, models
from torchvision.datasets import VOCDetection

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
import numpy as np
```

---

# Step 2: GPU Check

The notebook automatically checks whether CUDA is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")
```

If a GPU is available:

```text
Using device: cuda
```

Otherwise:

```text
Using device: cpu
```

The model and input tensors are moved to the selected device.

---

# Step 3: Load the Pre-Trained Faster R-CNN Model

The project uses the default pre-trained Faster R-CNN weights provided by Torchvision:

```python
model = models.detection.fasterrcnn_resnet50_fpn(
    weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)
```

The model is then prepared for inference:

```python
model = model.to(device)

model.eval()
```

### Important

This notebook does **not train or fine-tune the detector**.

It uses an already-trained Faster R-CNN model and performs inference on uploaded images.

---

# Step 4: COCO Object Classes

The model uses the COCO object-category mapping.

The notebook defines:

```python
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__',
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'N/A',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'N/A',
    'backpack',
    'umbrella',
    'N/A',
    'N/A',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'N/A',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'N/A',
    'dining table',
    'N/A',
    'N/A',
    'toilet',
    'N/A',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'N/A',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
]
```

The model therefore uses the COCO object-category indexing to translate numerical class predictions into human-readable object names.

---

# Step 5: Image Preprocessing

The notebook uses a simple preprocessing function:

```python
def preprocess_image(image):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    return transform(image)
```

The important operation here is:

```python
transforms.ToTensor()
```

which converts a PIL image from pixel values in the range:

```text
0–255
```

into a PyTorch tensor with values approximately in:

```text
0.0–1.0
```

---

# Step 6: Upload an Image

The project is designed for Google Colab.

The notebook uses:

```python
from google.colab import files
import io
```

The user is prompted to upload an image:

```python
print("Upload an image for object detection:")

uploaded = files.upload()
```

Multiple uploaded images can be processed because the code iterates over:

```python
uploaded.keys()
```

---

# Step 7: Confidence Threshold

The notebook uses a confidence threshold of:

```python
score_threshold = 0.5
```

This means that only detections with a confidence score of **50% or higher** are displayed.

For example:

```text
Dog → 92%       ✓ Display
Car → 78%       ✓ Display
Chair → 43%     ✗ Ignore
```

The predictions are filtered using:

```python
valid_indices = np.where(
    scores >= score_threshold
)[0]
```

---

# Step 8: Run Object Detection

Each uploaded image is opened and converted to RGB:

```python
raw_image = Image.open(
    io.BytesIO(uploaded[filename])
).convert("RGB")
```

It is then converted into a tensor:

```python
input_tensor = (
    preprocess_image(raw_image)
    .unsqueeze(0)
    .to(device)
)
```

The `unsqueeze(0)` adds the batch dimension.

The resulting input has the conceptual shape:

```text
[1, 3, H, W]
```

where:

- `1` = batch size
- `3` = RGB channels
- `H` = image height
- `W` = image width

---

# Step 9: Model Inference

The model performs inference without calculating gradients:

```python
with torch.no_grad():

    predictions = model(
        input_tensor
    )[0]
```

The prediction dictionary contains three important outputs:

```text
boxes
labels
scores
```

---

# 📦 Bounding Boxes

The `boxes` output contains bounding-box coordinates:

```text
[xmin, ymin, xmax, ymax]
```

The notebook converts them to NumPy:

```python
boxes = predictions[
    'boxes'
].cpu().numpy()
```

Each bounding box describes where an object is located in the image.

For example:

```text
[120, 80, 420, 600]
```

means approximately:

```text
left   = 120
top    = 80
right  = 420
bottom = 600
```

---

# 🏷️ Object Labels

The model returns numerical class IDs:

```python
labels = predictions[
    'labels'
].cpu().numpy()
```

These numerical IDs are converted into object names using:

```python
class_name = COCO_INSTANCE_CATEGORY_NAMES[
    label_idx
]
```

For example:

```text
17 → cat
18 → dog
19 → horse
```

according to the mapping used by the notebook.

---

# 📊 Confidence Scores

The model also returns a confidence score for every detection:

```python
scores = predictions[
    'scores'
].cpu().numpy()
```

Scores are represented as values between:

```text
0.0
```

and:

```text
1.0
```

For example:

```text
0.95 → 95%
0.82 → 82%
0.61 → 61%
```

The notebook converts the score to a percentage when displaying it:

```python
caption = (
    f"{class_name}: "
    f"{score*100:.1f}%"
)
```

---

# 🖼️ Drawing Bounding Boxes

For every valid detection, the notebook calculates:

```python
xmin, ymin, xmax, ymax = box

width = xmax - xmin
height = ymax - ymin
```

It then creates a rectangle:

```python
rect = patches.Rectangle(
    (xmin, ymin),
    width,
    height,
    linewidth=2,
    edgecolor='red',
    facecolor='none'
)
```

The rectangle is added to the image:

```python
ax.add_patch(rect)
```

The result is an image with bounding boxes surrounding detected objects.

---

# 🏷️ Displaying Object Names

Each bounding box receives a label containing:

```text
Class Name
+
Confidence Percentage
```

For example:

```text
person: 97.3%
```

The label is displayed above the corresponding bounding box.

---

# 📋 Detection Summary

In addition to displaying the image, the notebook prints a textual summary:

```python
print(
    f"\n--- Detected Objects in {filename} ---"
)

for idx in valid_indices:

    print(
        f"• "
        f"{COCO_INSTANCE_CATEGORY_NAMES[labels[idx]]}"
        f" "
        f"({scores[idx]*100:.2f}%) "
        f"at "
        f"[{boxes[idx].astype(int)}]"
    )
```

The output has the general format:

```text
--- Detected Objects in image.jpg ---

• person (97.32%) at [120 80 420 600]
• car (91.45%) at [450 300 900 550]
• bicycle (84.21%) at [200 250 500 600]
```

---

# 🔬 Understanding Object Detection Output

For every detected object, the model provides:

```text
┌──────────────────────────┐
│ Object Detection         │
├──────────────────────────┤
│ Class                    │
│ Confidence Score         │
│ Bounding Box             │
└──────────────────────────┘
```

For example:

```text
Class:
person

Confidence:
97.32%

Bounding Box:
[xmin, ymin, xmax, ymax]
```

This is fundamentally different from a standard image classifier.

---

# 🆚 Classification vs Object Detection

| Feature | Image Classification | Object Detection |
|---|---|---|
| Predicts object class | ✓ | ✓ |
| Finds object location | ✗ | ✓ |
| Bounding boxes | ✗ | ✓ |
| Multiple objects | Limited | ✓ |
| Confidence score | ✓ | ✓ |
| Example output | `dog` | `dog + bounding box` |

### Classification

```text
Image
  ↓
Dog
```

### Object Detection

```text
Image
  ↓
┌────────────────────────────┐
│        ┌──────────┐        │
│        │   DOG    │        │
│        └──────────┘        │
│                            │
│  ┌─────────┐               │
│  │ PERSON  │               │
│  └─────────┘               │
└────────────────────────────┘
```

---

# 🧠 Faster R-CNN Components

The model used in this project consists of several important components.

## ResNet50 Backbone

ResNet50 extracts visual features from the input image.

```text
Image
  ↓
ResNet50
  ↓
Visual Features
```

---

## Feature Pyramid Network

The FPN allows the detector to work with objects at different scales by constructing multi-scale feature representations.

```text
ResNet50 Features
       │
       ▼
Feature Pyramid Network
       │
       ├── Small-scale features
       ├── Medium-scale features
       └── Large-scale features
```

---

## Region Proposal Network

The Region Proposal Network identifies candidate regions that may contain objects.

```text
Feature Maps
     ↓
Region Proposal Network
     ↓
Candidate Object Regions
```

---

## ROI Detection

The proposed regions are then classified and assigned bounding boxes.

```text
Candidate Regions
       ↓
Classification
       +
Bounding Box Regression
       ↓
Final Detections
```

---

# 📈 Confidence Threshold

The project uses:

```python
score_threshold = 0.5
```

The threshold can be changed depending on the desired behavior.

### Lower Threshold

For example:

```python
score_threshold = 0.3
```

This allows more detections but may also display more false positives.

### Higher Threshold

For example:

```python
score_threshold = 0.8
```

This displays only higher-confidence detections but may remove some valid objects.

---

# ⚡ GPU Acceleration

The notebook automatically uses CUDA when available:

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
```

The model is moved to the selected device:

```python
model = model.to(device)
```

and each input image is also moved to the same device:

```python
input_tensor = input_tensor.to(device)
```

This is important because the model and input tensors must be located on the same device during inference.

---

# 🗂️ Project Workflow

The complete project workflow is:

```text
          Upload Image
               │
               ▼
        Convert to RGB
               │
               ▼
         ToTensor()
               │
               ▼
          Add Batch
            Dimension
               │
               ▼
       Faster R-CNN
       ResNet50 + FPN
               │
               ▼
       Model Predictions
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
     Boxes   Labels   Scores
       │       │        │
       └───────┼────────┘
               ▼
      Confidence Filtering
          score ≥ 0.5
               │
               ▼
       Draw Bounding Boxes
               │
               ▼
     Display Detection Image
               │
               ▼
       Print Detection Summary
```

---

# 📚 COCO Object Categories

The pre-trained detector supports the COCO category mapping used by the notebook, including objects such as:

```text
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
suitcase
bottle
cup
fork
knife
bowl
banana
apple
pizza
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
keyboard
cell phone
microwave
oven
sink
refrigerator
book
clock
vase
scissors
teddy bear
toothbrush
```

The complete category mapping is defined directly in the notebook through:

```python
COCO_INSTANCE_CATEGORY_NAMES
```

---

# 📦 Requirements

The project requires:

```text
Python
PyTorch
Torchvision
Matplotlib
Pillow
NumPy
```

For example:

```bash
pip install torch torchvision matplotlib pillow numpy
```

The notebook is intended to be run in **Google Colab**, where the required libraries are commonly available.

---

# ▶️ Running the Project

## Option 1: Google Colab

1. Open the notebook in Google Colab.
2. Enable GPU acceleration if available.
3. Run the imports.
4. Check the selected device.
5. Load the pre-trained Faster R-CNN model.
6. Define the COCO class names.
7. Upload an image.
8. Run object detection.
9. Adjust the confidence threshold if necessary.
10. View the detected objects and bounding boxes.

---

# 📁 Suggested Project Structure

A simple repository structure could be:

```text
Object-Detector/
│
├── Object_Detector_Model_(MS_COCO_80_categories)_
│   Pre_trained_Faster_R_CNN_with_ResNet_50.ipynb
│
├── README.md
│
└── images/
    └── example.jpg
```

---

# 🚀 Possible Improvements

The current notebook focuses on straightforward inference using the pre-trained model.

Possible extensions include:

- Add Non-Maximum Suppression visualization
- Allow the user to change the confidence threshold interactively
- Process video files
- Process webcam frames
- Save annotated images
- Detect objects in batches
- Display detection statistics
- Measure inference time
- Add FPS calculation for video detection
- Build a simple web interface
- Fine-tune Faster R-CNN on a custom dataset
- Train on custom object categories
- Export the model for deployment

---

# 🧪 Custom Dataset Training

The current notebook does **not** train Faster R-CNN on a custom dataset.

Instead, it uses:

```text
Pre-trained Faster R-CNN
        +
COCO-trained weights
        ↓
Direct Inference
```

A future version could replace the COCO-trained detector with a model fine-tuned on a custom object-detection dataset.

That would require:

```text
Custom Images
      +
Bounding Box Annotations
      ↓
Custom Dataset
      ↓
Faster R-CNN Fine-Tuning
      ↓
Custom Object Detector
```

---

# ⚠️ Important Notes

### Pre-trained Model

This project uses the default Torchvision pre-trained weights:

```python
models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
```

The notebook does not perform model training.

### COCO Classes

The detector is limited to the object categories represented by its pre-trained COCO model.

It cannot automatically recognize arbitrary custom objects that are not represented in those categories.

### Confidence Threshold

The displayed detections are limited to:

```text
confidence ≥ 50%
```

This is only a filtering threshold; it does not retrain or modify the underlying model.

---

# 📝 Summary

This project demonstrates how to use a **pre-trained Faster R-CNN object detector with a ResNet50 backbone and Feature Pyramid Network** to detect multiple objects in an image.

The complete process is:

```text
Input Image
    ↓
Preprocessing
    ↓
Faster R-CNN
    ↓
ResNet50 + FPN
    ↓
Object Detection
    ↓
Bounding Boxes
    +
Class Labels
    +
Confidence Scores
    ↓
Confidence Filtering
    ↓
Annotated Image
```

The main difference between this project and the previous image-classification projects is that the model does not simply answer:

```text
"What is in this image?"
```

Instead, it answers:

```text
"What objects are in this image,
where are they,
and how confident is the model?"
```

This makes Faster R-CNN suitable for applications where **object localization** is required in addition to object recognition.

---

# 📌 Key Concepts Demonstrated

This project demonstrates:

- Object Detection
- Faster R-CNN
- ResNet50
- Feature Pyramid Networks
- Transfer Learning
- Pre-trained COCO models
- Bounding Boxes
- Object Classification
- Confidence Scores
- Confidence Thresholding
- PyTorch
- Torchvision
- CUDA / GPU inference
- Image preprocessing
- Google Colab
- Visualization with Matplotlib
- Multi-object detection
