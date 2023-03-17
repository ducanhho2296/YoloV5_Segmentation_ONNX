import cv2

# Load the image
img = cv2.imread("C:/Users/NovoAIUser/Documents/GitHub/FastAPI-Real-time-data-streaming/demo.jpg")

cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
cv2.imshow('Select ROI', img)
rect = cv2.selectROI('Select ROI', img, False)
cv2.destroyWindow('Select ROI')

# Extract the ROI from the image
x, y, w, h = rect
roi = img[y:y+h, x:x+w]

