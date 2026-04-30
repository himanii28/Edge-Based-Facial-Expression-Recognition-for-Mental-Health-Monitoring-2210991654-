import os
import tensorflow as tf
from data_pipeline import get_data_generators
from model import build_lightweight_fer_model, compile_model

def main():
    base_dir = r'd:\hrp'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    
    print("Initializing data generators...")
    train_gen, test_gen = get_data_generators(train_dir, test_dir, batch_size=64)
    
    print("Building model...")
    model = build_lightweight_fer_model(num_classes=7)
    model = compile_model(model)
    model.summary()
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(base_dir, 'best_fer_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            verbose=1,
            min_lr=1e-5
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=12,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=10,
        callbacks=callbacks
    )
    
    print("Training complete. Best model saved as 'best_fer_model.h5'.")

if __name__ == "__main__":
    main()
