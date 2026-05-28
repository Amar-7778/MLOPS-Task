import os
import cv2

# Change working directory
os.chdir(r"D:\Academic Projects\Task 3\day 8")

# Video path
cap = cv2.VideoCapture(
    r"C:\Users\Amarnath\Downloads\1100011002.avi\1100011002.avi"
)

# Get video FPS
framerate = int(cap.get(cv2.CAP_PROP_FPS))

# Variables
framecount = 0
count = 0

# Approximate 750 images for 2-minute video
# Around 6 to 7 images per second

interval = 1
while True:
    success, frame = cap.read()

    # Stop if video ends
    if not success:
        break

    # Resize frame
    frame = cv2.resize(frame, (1280, 720))

    framecount += 1

    # Save frames
    if framecount >= interval:
        framecount = 0

        filename = f"frames{count}.jpg"
        cv2.imwrite(filename, frame)

        print(f"Saved: {filename}")
        count += 1

    # Press q to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"\nTotal Images Saved: {count}")