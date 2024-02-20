import cv2
import os

folder_name = r"C:\Hammy\Kuliah\9. Semester 5\PKL\shintavr\saved"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

total_images = 5
counter = 1
name = str(input("Write your name: "))

cam = cv2.VideoCapture(0)  # Access the camera

while True:
    ret, frame = cam.read()  # Read each frame from the camera stream
    frame_copy = frame.copy()  # Copy frame
    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )  # Convert mode BGR to GRAY (black and white)

    cv2.imshow("Face Detect Video", frame)  # Window to display the result

    key = cv2.waitKey(1)

    if key == ord("c"):  # Wait for 'c' key to be pressed
        cv2.imwrite(f"{folder_name}/{name}_{counter}.jpg", frame_copy)

        counter += 1
        if counter > total_images:
            print(f"[INFO] {total_images} IMAGES CAPTURED!")  # Info about the process
            break

    elif key == ord("q"):  # Exit with 'q' key
        break

cam.release()  # Release camera access
cv2.destroyAllWindows()
