import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque
from data_pipeline import preprocess_frame

def main():
    model_path = r'd:\hrp\best_fer_model.h5'
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have trained the model by running train.py first.")
        return

    # ImageDataGenerator flow_from_directory returns classes in alphabetical order:
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Emotion index mappings for Distress logic
    idx_sad = emotion_labels.index('sad')
    idx_fear = emotion_labels.index('fear')
    idx_angry = emotion_labels.index('angry')
    idx_happy = emotion_labels.index('happy')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 25) # Target 20-25 FPS
    
    # Temporal Smoothing
    alpha = 0.2
    p_bar_t_1 = np.zeros(7)
    
    # Rolling 5-minute window for Distress Trend Indicator
    # 5 minutes = 300 seconds. We'll store 1 average score per second.
    rolling_distress = deque(maxlen=300)
    
    last_tick = time.time()
    current_second_scores = []
    
    print("Starting real-time inference... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        faces_tensors, face_coords = preprocess_frame(frame)
        
        for tensor, (x, y, w, h) in zip(faces_tensors, face_coords):
            # Predict probabilities
            pred = model.predict(np.expand_dims(tensor, axis=0), verbose=0)[0]
            
            # 1. Exponential Smoothing
            p_bar_t = alpha * pred + (1 - alpha) * p_bar_t_1
            p_bar_t_1 = p_bar_t
            
            # 2. Distress Trend Indicator Logic
            distress_score = (0.35 * p_bar_t[idx_sad] + 
                              0.30 * p_bar_t[idx_fear] + 
                              0.20 * p_bar_t[idx_angry] - 
                              0.15 * p_bar_t[idx_happy])
            distress_score = np.clip(distress_score, 0.0, 1.0)
            
            # Visualization
            dominant_idx = np.argmax(p_bar_t)
            emotion = emotion_labels[dominant_idx]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({p_bar_t[dominant_idx]:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Distress: {distress_score:.2f}", (x, y+h+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            current_second_scores.append(distress_score)
            
        # 3. Aggregate score over 1-second ticks
        current_time = time.time()
        if current_time - last_tick >= 1.0:
            if current_second_scores:
                rolling_distress.append(np.mean(current_second_scores))
            else:
                rolling_distress.append(0.0) # No faces detected
                
            current_second_scores = []
            last_tick = current_time
            
        # 4. Check for sustained elevation in the 5 min window (300 seconds)
        warning_msg = ""
        # Only evaluate if we have a full window or a reasonable amount of time has passed
        if len(rolling_distress) >= 10: 
            avg_window_distress = np.mean(rolling_distress)
            if avg_window_distress > 0.4: # Arbitrary threshold for sustained risk
                warning_msg = "RISK SIGNAL: Sustained distress elevated!"
                
        if warning_msg:
            cv2.putText(frame, warning_msg, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            
            # Log the signal
            with open("risk_log.txt", "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {warning_msg} (Score: {avg_window_distress:.2f})\n")

        cv2.imshow('Edge-Based Mental Health Monitor', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
