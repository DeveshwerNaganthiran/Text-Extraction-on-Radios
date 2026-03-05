import cv2
import time

def check(n=8):
    for i in range(n):
        # Use DirectShow backend on Windows for better compatibility
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        opened = cap.isOpened()
        print(f"Index {i}: opened={opened}")
        if opened:
            ret, _ = cap.read()
            print(f"  read_ok={ret}")
            cap.release()
        time.sleep(0.2)

if __name__ == '__main__':
    check(6)
