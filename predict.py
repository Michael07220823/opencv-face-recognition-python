import cv2
from Face import predict, classification_label, face_recognizer

# Load LBPH model.
face_recognizer.read("model/model.yaml")

# Load test image.
read_image = cv2.imread("test-data/test5.jpg")

# Predict image.
predict_img = predict(read_image)

cv2.imshow(classification_label[3], cv2.resize(predict_img, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()