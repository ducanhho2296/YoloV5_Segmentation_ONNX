import cv2

# Load the image
img = cv2.imread("img_path")
cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
cv2.imshow('Select ROI', img)
rect = cv2.selectROI('Select ROI', img, False)
cv2.destroyWindow('Select ROI')

# Extract the ROI from the image
x, y, w, h = rect

roi = img[y:y+h, x:x+w]

# Calculate the average brightness of the ROI
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
avg_brightness = cv2.mean(gray_roi)[0]

# Adjust the exposure of the entire image based on the average brightness of the ROI
exposure_compensation = int(round((128 - avg_brightness) / 16))
adjusted_img = cv2.add(img, exposure_compensation)

# Display the original image and the adjusted image side by side
cv2.imshow('Original Image', img)
cv2.imshow('Adjusted Image', adjusted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()