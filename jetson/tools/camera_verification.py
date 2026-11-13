
import cv2


def main():
    # CSI Camera Mode 3: 1640x1232 @ 30 FPS
    pipeline = (
        'nvarguscamerasrc sensor-id=0 ! '
        'video/x-raw(memory:NVMM),'
        'width=1640,height=1232,framerate=30/1 ! '
        'nvvidconv ! '
        'video/x-raw,format=BGRx ! '
        'videoconvert ! '
        'video/x-raw,format=BGR ! '
        'appsink drop=1 max-buffers=1'
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    print("Camera opened:", cap.isOpened())

    if not cap.isOpened():
        raise RuntimeError("Failed to open CSI camera")

    try:
        ret, frame = cap.read()

        print("Frame captured:", ret)

        if not ret:
            raise RuntimeError("Failed to capture camera frame")

        print("Frame shape:", frame.shape)

        output_path = "camera_test.jpg"

        if cv2.imwrite(output_path, frame):
            print("Saved:", output_path)
        else:
            raise RuntimeError("Failed to save camera image")

    finally:
        cap.release()


if __name__ == "__main__":
    main()
