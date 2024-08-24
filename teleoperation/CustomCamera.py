import cv2
import numpy as np
import traitlets

class CustomCamera(traitlets.HasTraits):
    image = traitlets.Bytes()

    def __init__(self, device_id=0, width=640, height=480, fps=30, **kwargs):
        super().__init__(**kwargs)
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

    def update(self):
        """Capture a new image from the camera and update the `image` trait."""
        ret, frame = self.cap.read()
        if ret:
            try:
                _, jpeg = cv2.imencode('.jpg', frame)
                self.image = jpeg.tobytes()
            except cv2.error as e:
                print(f"Failed to encode image: {e}")
        else:
            print("Failed to capture image")

    def release(self):
        """Release the camera resource."""
        self.cap.release()

    def __del__(self):
        """Destructor to ensure resources are released when object is deleted."""
        self.release()

def main():
    camera = CustomCamera()

    while True:
        camera.update()
        if camera.image:
            # Convert bytes to numpy array and then to OpenCV image format
            image_np = cv2.imdecode(np.frombuffer(camera.image, np.uint8), cv2.IMREAD_COLOR)
            
            # Display the image
            cv2.imshow('Live Feed', image_np)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
