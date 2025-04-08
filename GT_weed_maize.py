import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

# Define constants
IMG_SIZE = (416, 416)
NUM_CLASSES = 2
CLASS_NAMES = ['maize', 'weed']
DATA_DIR = 'Annotated-Maize-Weed-Images'  # Update with your dataset path

# Dummy model that doesn't do real predictions
def create_dummy_model():
    """Create a simple dummy model for demonstration purposes."""
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.Conv2D(8, 3, padding='same', activation='relu')(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    class_output = layers.Dense(NUM_CLASSES, activation='softmax', name='class_output')(x)
    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox_output')(x)
    
    model = models.Model(inputs=inputs, outputs=[class_output, bbox_output])
    model.compile(optimizer='adam', loss='mse')
    
    return model

def visualize_ground_truth(image_path):
    """
    Display an image with colored ground truth bounding boxes:
    - Green for maize (class 0)
    - Red for weed (class 1)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_height, original_width = img.shape[:2]
    
    # Get annotation file path
    txt_path = image_path.replace('.jpg', '.txt')
    if not os.path.exists(txt_path):
        print(f"Annotation file not found: {txt_path}")
        return
    
    # Read ground truth annotations (
    true_boxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert normalized coordinates to pixel values
                x1 = int((x_center - width/2) * original_width)
                y1 = int((y_center - height/2) * original_height)
                x2 = int((x_center + width/2) * original_width)
                y2 = int((y_center + height/2) * original_height)
                
                true_boxes.append({
                    'class_id': class_id,
                    'class_name': CLASS_NAMES[class_id],
                    'box': [x1, y1, x2, y2]
                })
    
    # Visualize results
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    
    # Plot ground truth boxes with different colors based on class
    for obj in true_boxes:
        x1, y1, x2, y2 = obj['box']
        class_id = obj['class_id']
        class_name = obj['class_name']
        
        # Green for maize, red for weed
        color = 'g' if class_id == 0 else 'r'
        
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, 
            linewidth=2, edgecolor=color, facecolor='none'
        )
        plt.gca().add_patch(rect)
        plt.text(
            x1, y1-10, 
            f"{class_name}",
            color=color, fontsize=12, backgroundcolor='white'
        )
    
    plt.axis('off')
    plt.title('Maize (Green), Weed (Red)')
    plt.tight_layout()
    plt.savefig('ground_truth_visualization.png')
    plt.show()

def main():
    # Create dummy model
    dummy_model = create_dummy_model()
    
    # Visualize ground truth on sample images
    print("Visualizing ground truth bounding boxes...")
    
    # Get the first 5 images from directory (or adjust as needed)
    image_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')][:5]
    
    for img_file in image_files:
        # img_path = os.path.join(DATA_DIR, img_file)
        img_path = "Annotated-Maize-Weed-Images\C00020611.jpg"
        print(f"Processing {img_path}")
        visualize_ground_truth(img_path)

if __name__ == "__main__":
    main()
