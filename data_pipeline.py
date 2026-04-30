import os
import cv2
import numpy as np
import tensorflow as tf

def preprocess_frame(frame, face_cascade_path=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'):
    """
    Detects face, crops, resizes to 48x48 grayscale, applies histogram equalization,
    and normalizes pixel values to [0, 1].
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48))
    
    processed_faces = []
    face_coords = []
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_eq = cv2.equalizeHist(face_resized)
        face_norm = face_eq.astype('float32') / 255.0
        face_tensor = np.expand_dims(face_norm, axis=-1) # (48, 48, 1)
        
        processed_faces.append(face_tensor)
        face_coords.append((x, y, w, h))
        
    return processed_faces, face_coords

def random_occlusion(image):
    """
    Applies random occlusion patches to the image as data augmentation.
    Image is expected to be a numpy array.
    """
    img = np.array(image)
    h, w = img.shape[:2]
    
    # 50% chance to apply occlusion
    if np.random.rand() < 0.5:
        return img
        
    patch_size = np.random.randint(4, 12)
    x = np.random.randint(0, w - patch_size)
    y = np.random.randint(0, h - patch_size)
    
    # fill with random noise or gray
    if img.dtype == np.float32 or img.dtype == np.float64:
        img[y:y+patch_size, x:x+patch_size] = np.random.uniform(0, 1)
    else:
        img[y:y+patch_size, x:x+patch_size] = np.random.randint(0, 255)
    
    return img

def get_data_generators(train_dir, test_dir, batch_size=64):
    """
    Sets up tf.keras ImageDataGenerators with the requested augmentations.
    """
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=12,
        brightness_range=[0.8, 1.2],
        preprocessing_function=random_occlusion
    )
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, test_generator
