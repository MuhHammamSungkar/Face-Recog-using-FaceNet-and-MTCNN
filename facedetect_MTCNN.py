import cv2
from facenet_pytorch import MTCNN
from types import MethodType


def detect_box(self, img, save_path=None):
    # Detect faces
    batch_boxes, _, _ = self.detect(img, landmarks=True)
    # Select faces
    if not self.keep_all:
        batch_boxes, _, _ = self.select_boxes(
            batch_boxes, None, None, img, method=self.selection_method
        )
    # Extract faces
    faces = self.extract(img, batch_boxes, save_path)
    return batch_boxes, faces


# Load MTCNN model
mtcnn = MTCNN(
    image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60
)
mtcnn.detect_box = MethodType(detect_box, mtcnn)


def detect(cam=0):
    vdo = cv2.VideoCapture(cam)
    while vdo.grab():
        _, img0 = vdo.retrieve()
        batch_boxes, _ = mtcnn.detect_box(img0)

        if batch_boxes is not None:
            for box in batch_boxes:
                x, y, x2, y2 = [int(coord) for coord in box]
                cv2.rectangle(img0, (x, y), (x2, y2), (0, 0, 255), 2)

        # Display the output
        cv2.imshow("Face Detection", img0)
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    detect(0)
