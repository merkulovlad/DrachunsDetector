
import cv2

class FrameSource:
    def __init__(self, source=0, desired_fps=None):
        self.cap = cv2.VideoCapture(source)
        if desired_fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, desired_fps)
        self.source = source

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        if self.cap:
            self.cap.release()
