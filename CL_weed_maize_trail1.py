import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, applications, optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define constants
IMG_SIZE = (416, 416)  # Standard input size for many object detection models
BATCH_SIZE = 8
EPOCHS = 50
NUM_CLASSES = 2  # maize and weed
CLASS_NAMES = ['maize', 'weed']
DATA_DIR = r'E:\Project_2\Annotated-Maize-Weed-Images'
MODEL_SAVE_PATH = 'maize_weed_detection_model.h5'

# Create a custom CNN model for object detection
def create_detection_model():
    """
    Create a custom object detection model based on a CNN backbone
    with additional detection heads for bounding box prediction.
    """
    # Base model (feature extractor)
    base_model = applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Build detection head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    
    # Output layers:
    # - Class prediction (softmax for classification)
    # - Bounding box regression (linear activation for coordinates)
    class_output = layers.Dense(NUM_CLASSES, activation='softmax', name='class_output')(x)
    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox_output')(x)  # x, y, w, h (normalized)
    
    # Create the model
    model = models.Model(
        inputs=base_model.input, 
        outputs=[class_output, bbox_output]
    )
    
    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss={
            'class_output': 'categorical_crossentropy',
            'bbox_output': 'mse'
        },
        loss_weights={
            'class_output': 1.0,
            'bbox_output': 5.0  # Give more weight to bounding box accuracy
        },
        metrics={
            'class_output': 'accuracy',
            'bbox_output': 'mse'
        }
    )
    
    return model

def load_dataset():
    """
    Load the dataset from the specified directory.
    Returns X (images), y_class (class labels), y_bbox (bounding boxes)
    """
    X = []
    y_class = []
    y_bbox = []
    
    # List all image files
    image_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} images.")
    
    # Process each image and its annotations
    for img_file in image_files:
        img_path = os.path.join(DATA_DIR, img_file)
        txt_path = os.path.join(DATA_DIR, img_file.replace('.jpg', '.txt'))
        json_path = os.path.join(DATA_DIR, img_file.replace('.jpg', '.json'))
        
        # Skip if annotation files don't exist
        if not (os.path.exists(txt_path) and os.path.exists(json_path)):
            print(f"Skipping {img_file} - missing annotation files")
            continue
        
        # Load and resize image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_file}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, IMG_SIZE)
        
        # Normalize pixel values
        img_normalized = img_resized / 255.0
        
        # Read TXT annotations (YOLO format)
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        # Process each annotation line (we'll only use the first annotation for simplicity)
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Create one-hot encoded class vector
            class_vector = np.zeros(NUM_CLASSES)
            class_vector[class_id] = 1
            
            # Bounding box coordinates
            bbox = np.array([x_center, y_center, width, height])
            
            # Add to dataset
            X.append(img_normalized)
            y_class.append(class_vector)
            y_bbox.append(bbox)
            
            # For simplicity, we'll only use one annotation per image
            # In a real system, you'd process all annotations for multi-object detection
            break
    
    # Convert to numpy arrays
    X = np.array(X)
    y_class = np.array(y_class)
    y_bbox = np.array(y_bbox)
    
    print(f"Dataset loaded: {len(X)} images with annotations")
    return X, y_class, y_bbox

def train_model():
    """
    Train the object detection model
    """
    # Load dataset
    X, y_class, y_bbox = load_dataset()
    
    # Split into training and validation sets
    X_train, X_val, y_class_train, y_class_val, y_bbox_train, y_bbox_val = train_test_split(
        X, y_class, y_bbox, test_size=0.2, random_state=42
    )
    
    # Create model
    model = create_detection_model()
    print(model.summary())
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH, 
            save_best_only=True,
            monitor='val_loss'
        ),
        EarlyStopping(
            patience=10,
            monitor='val_loss'
        ),
        ReduceLROnPlateau(
            factor=0.1,
            patience=5,
            monitor='val_loss',
            min_lr=1e-6
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train,
        {
            'class_output': y_class_train,
            'bbox_output': y_bbox_train
        },
        validation_data=(
            X_val,
            {
                'class_output': y_class_val,
                'bbox_output': y_bbox_val
            }
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['class_output_accuracy'])
    plt.plot(history.history['val_class_output_accuracy'])
    plt.title('Class Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['bbox_output_mse'])
    plt.plot(history.history['val_bbox_output_mse'])
    plt.title('Bounding Box MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    return model

def predict_and_visualize(model, image_path):
    """
    Make predictions and visualize results with bounding boxes
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Store original dimensions
    original_height, original_width = img.shape[:2]
    
    # Resize and normalize
    img_resized = cv2.resize(img, IMG_SIZE)
    img_normalized = img_resized / 255.0
    
    # Make prediction
    class_pred, bbox_pred = model.predict(np.expand_dims(img_normalized, axis=0))
    
    # Get predicted class
    class_id = np.argmax(class_pred[0])
    class_name = CLASS_NAMES[class_id]
    confidence = class_pred[0][class_id]
    
    # Get predicted bounding box
    # The model outputs normalized coordinates (0-1)
    x_center, y_center, width, height = bbox_pred[0]
    
    # Convert to pixel coordinates in original image
    x1 = int((x_center - width/2) * original_width)
    y1 = int((y_center - height/2) * original_height)
    x2 = int((x_center + width/2) * original_width)
    y2 = int((y_center + height/2) * original_height)
    
    # Ensure coordinates are within image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(original_width, x2)
    y2 = min(original_height, y2)
    
    # Display image with bounding box
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    
    # Create a rectangle patch
    rect = patches.Rectangle(
        (x1, y1), x2-x1, y2-y1, 
        linewidth=2, edgecolor='r', facecolor='none'
    )
    
    # Add the patch to the Axes
    plt.gca().add_patch(rect)
    
    # Add label with class name and confidence
    plt.text(
        x1, y1-10, 
        f"{class_name}: {confidence:.2f}",
        color='red', fontsize=12, backgroundcolor='white'
    )
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()

def process_multiple_objects(model, image_path):
    """
    Process an image and detect multiple objects
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Store original dimensions
    original_height, original_width = img.shape[:2]
    
    # Load annotations to compare with predictions
    txt_path = image_path.replace('.jpg', '.txt')
    json_path = image_path.replace('.jpg', '.json')
    
    # Read ground truth annotations
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
                    'class': CLASS_NAMES[class_id],
                    'box': [x1, y1, x2, y2]
                })
    
    # Implement a simple sliding window approach for multi-object detection
    window_sizes = [(416, 416), (512, 512), (640, 640)]
    stride = 200
    detections = []
    
    for size in window_sizes:
        for y in range(0, original_height - size[1] + 1, stride):
            for x in range(0, original_width - size[0] + 1, stride):
                # Extract window
                window = img[y:y+size[1], x:x+size[0]]
                window_resized = cv2.resize(window, IMG_SIZE)
                window_normalized = window_resized / 255.0
                
                # Make prediction
                class_pred, bbox_pred = model.predict(np.expand_dims(window_normalized, axis=0))
                
                # Get predicted class
                class_id = np.argmax(class_pred[0])
                confidence = class_pred[0][class_id]
                
                # Skip low confidence detections
                if confidence < 0.5:
                    continue
                
                # Get predicted bounding box
                wx, wy, ww, wh = bbox_pred[0]
                
                # Convert to pixel coordinates in window
                wx1 = int((wx - ww/2) * size[0])
                wy1 = int((wy - wh/2) * size[1])
                wx2 = int((wx + ww/2) * size[0])
                wy2 = int((wy + wh/2) * size[1])
                
                # Convert to coordinates in original image
                x1 = x + wx1
                y1 = y + wy1
                x2 = x + wx2
                y2 = y + wy2
                
                # Add to detections
                detections.append({
                    'class': CLASS_NAMES[class_id],
                    'confidence': float(confidence),
                    'box': [x1, y1, x2, y2]
                })
    
    # Non-maximum suppression to remove overlapping detections
    def calculate_iou(box1, box2):
        # Calculate intersection over union
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # Apply NMS
    final_detections = []
    while detections:
        current = detections.pop(0)
        final_detections.append(current)
        
        i = 0
        while i < len(detections):
            if detections[i]['class'] == current['class'] and calculate_iou(current['box'], detections[i]['box']) > 0.5:
                detections.pop(i)
            else:
                i += 1
    
    # Visualize results
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    
    # Plot ground truth boxes in green
    for obj in true_boxes:
        x1, y1, x2, y2 = obj['box']
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, 
            linewidth=2, edgecolor='g', facecolor='none'
        )
        plt.gca().add_patch(rect)
        plt.text(
            x1, y1-10, 
            f"GT: {obj['class']}",
            color='green', fontsize=10, backgroundcolor='white'
        )
    
    # Plot predicted boxes in red
    for obj in final_detections:
        x1, y1, x2, y2 = obj['box']
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, 
            linewidth=2, edgecolor='r', facecolor='none'
        )
        plt.gca().add_patch(rect)
        plt.text(
            x1, y1-10, 
            f"{obj['class']}: {obj['confidence']:.2f}",
            color='red', fontsize=10, backgroundcolor='white'
        )
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('multi_object_detection_result.png')
    plt.show()

def main():
    # Train the model
    print("Training the model...")
    model = train_model()
    
    # Test on a single image
    print("Testing on a sample image...")
    sample_image = os.path.join(DATA_DIR, os.listdir(DATA_DIR)[0])
    process_multiple_objects(model, sample_image)
    
    print("Process completed!")

if __name__ == "__main__":
    main()