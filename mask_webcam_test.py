import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('final_model.keras')

# Define your labels (change if different in your model)
labels = ['Mask', 'No Mask']

# Set input size (update based on your model's training input shape)
IMG_SIZE = 150  # e.g., 150x150 — change if yours is different

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Select region of interest (or use full frame)
    roi = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)

    # Predict
    pred = model.predict(roi)
    class_idx = np.argmax(pred)
    label = labels[class_idx]
    confidence = pred[0][class_idx]

    # Display result
    cv2.putText(frame, f'{label} ({confidence*100:.2f}%)', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == 'Mask' else (0, 0, 255), 2)
    cv2.imshow("Face Mask Detection", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
