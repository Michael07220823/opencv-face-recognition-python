import os
import cv2
import numpy as np
from Face import prepare_training_data, predict, classification_label, face_recognizer


print("[FACE] Preparing data...")
faces, labels = prepare_training_data("training-data")
print("[FACE] Data prepared...")

# Print total faces and labels
print("[FACE] Total faces: ", len(faces))
print("[FACE] Total labels: ", len(labels))


# Train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))


# Load test images
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")
test_img3 = cv2.imread("test-data/test3.jpg")

# predict image.
print("[FACE] Predicting images...")
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)
print("[FACE] Prediction complete")

# Display both images
cv2.imshow(classification_label[1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(classification_label[2], cv2.resize(predicted_img2, (400, 500)))
cv2.imshow(classification_label[3], cv2.resize(predicted_img3, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save model
face_recognizer.write("model/model.yaml")