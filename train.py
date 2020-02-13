import cv2
import os
import numpy as np
from Face import prepare_training_data, predict


#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["unknow", "Ramiz Raja", "Elvis Presley", "Michael"]

print("Preparing data...")
faces, labels = prepare_training_data("training-data")

print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))



print("Predicting images...")

#load test images
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")
test_img3 = cv2.imread("test-data/test3.jpg")

#perform a prediction
predicted_img1 = predict(test_img1, face_recognizer, subjects)
predicted_img2 = predict(test_img2, face_recognizer, subjects)
predicted_img3 = predict(test_img3, face_recognizer, subjects)
print("Prediction complete")

#display both images
cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.imshow(subjects[3], cv2.resize(predicted_img3, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save model
face_recognizer.write("model/model.yaml")