import subprocess
import sys

def install_and_import_opencv():
    try:
        import cv2
        print(f"OpenCV is already installed. Current version: {cv2.__version__}")
    except ModuleNotFoundError:
        print("OpenCV is not installed. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
        import cv2
        print(f"OpenCV successfully installed. Current version: {cv2.__version__}")

if __name__ == "__main__":
    install_and_import_opencv()
