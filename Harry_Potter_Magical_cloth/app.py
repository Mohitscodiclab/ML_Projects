import cv2
import numpy as np
import time

# Initialize camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Get camera properties for proper window sizing
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create windows
cv2.namedWindow("Invisible Cloak Setup", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Invisible Cloak Setup", frame_width, frame_height)
cv2.namedWindow("Invisible Cloak Effect", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Invisible Cloak Effect", frame_width, frame_height)

# Step 1: Capture background for 3 seconds
print("Capturing background... Please keep the camera still for 3 seconds.")
background_captured = False
background = None
start_time = time.time()

while time.time() - start_time < 3:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    # Display countdown
    remaining = max(0, 3 - int(time.time() - start_time))
    cv2.putText(frame, f"Capturing background: {remaining}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Invisible Cloak Setup", frame)
    cv2.waitKey(1)

# Capture the background frame
ret, background = cap.read()
if not ret:
    print("Error: Failed to capture background")
    cap.release()
    cv2.destroyAllWindows()
    exit()

print("Background captured successfully!")

# Step 2: Capture cloak color
print("\nPlease show your colored cloth to the camera.")
print("Position it in the center of the frame and press 'c' to capture its color.")

cloak_color_captured = False
cloak_color_hsv = None

while not cloak_color_captured:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    # Create a copy for display
    display_frame = frame.copy()
    
    # Draw a rectangle in the center to guide the user
    h, w, _ = frame.shape
    rect_size = min(w, h) // 5
    x1, y1 = (w - rect_size) // 2, (h - rect_size) // 2
    x2, y2 = x1 + rect_size, y1 + rect_size
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(display_frame, "Place cloth here and press 'c'", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("Invisible Cloak Setup", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Extract the region of interest (center rectangle)
        roi = frame[y1:y2, x1:x2]
        
        # Convert to HSV and calculate average color
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        cloak_color_hsv = np.mean(hsv_roi, axis=(0, 1))
        
        print(f"Captured cloak color (HSV): {cloak_color_hsv}")
        cloak_color_captured = True
    elif key == ord('q'):
        print("Quitting...")
        cap.release()
        cv2.destroyAllWindows()
        exit()

print("\nStarting invisibility effect! Press 'q' to quit.")

# Step 3: Main loop for invisibility effect
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    # Convert current frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color range for the cloak (with tolerance)
    lower_bound = np.array([
        max(0, cloak_color_hsv[0] - 15), 
        max(0, cloak_color_hsv[1] - 40), 
        max(0, cloak_color_hsv[2] - 40)
    ])
    upper_bound = np.array([
        min(179, cloak_color_hsv[0] + 15), 
        min(255, cloak_color_hsv[1] + 40), 
        min(255, cloak_color_hsv[2] + 40)
    ])
    
    # Create mask for the cloak color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    # Create inverse mask
    mask_inv = cv2.bitwise_not(mask)
    
    # Apply the mask to the current frame and background
    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=mask_inv)
    
    # Combine the two areas
    result = cv2.add(cloak_area, non_cloak_area)
    
    # Display the result
    cv2.imshow("Invisible Cloak Effect", result)
    
    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Program ended. Goodbye!")