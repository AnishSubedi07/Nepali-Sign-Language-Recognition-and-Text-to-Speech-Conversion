import os
import cv2

# Define directories
grayscale_directory = "Pictures/picture_grayscale"
original_directory = "Pictures/picture_original"

# Create base directories if they do not exist
if not os.path.exists(grayscale_directory):
    os.makedirs(grayscale_directory)
if not os.path.exists(original_directory):
    os.makedirs(original_directory)

# Initialize camera
cap = cv2.VideoCapture(0)

min_value = 70
capture_images = False
capture_number = 0

# Variable to increment file names
x = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (20, 50), (350, 350), (255, 0, 0), 1)
    cv2.imshow("Frame", frame)

    roi = frame[50:350, 20:350]

    original_image = roi.copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    blur = cv2.bilateralFilter(blur, 3, 75, 75)
    th3 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    _, roi = cv2.threshold(th3, min_value, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow("ROI", roi)

    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF == 27:  # ESC key
        break

    if not capture_images:
        for number in range(10):
            if interrupt & 0xFF == ord(str(number)):
                capture_images = True
                capture_number = number  # Capture the number pressed
                break
    else:
        # Combine key press number with the x variable (as numbers, not strings)
        file_name_prefix = capture_number + x  # Treat as numbers

        # Save grayscale image
        grayscale_file_path = f"{grayscale_directory}/{file_name_prefix}.jpg"
        cv2.imwrite(grayscale_file_path, roi)

        # Save original (non-processed) image
        original_file_path = f"{original_directory}/{file_name_prefix}.jpg"
        cv2.imwrite(original_file_path, original_image)

        # Reset capture_images to False after saving one image
        capture_images = False

cap.release()
cv2.destroyAllWindows()
