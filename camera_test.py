import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Laptop camera not detected")
    exit()

print("Laptop camera started successfully")
print("Press Q to quit")

while True:
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Cannot read laptop camera")
        break

    cv2.imshow("SmileInsight - Laptop Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
