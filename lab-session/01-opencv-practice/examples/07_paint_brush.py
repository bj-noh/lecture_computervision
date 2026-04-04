from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    canvas = np.full((480, 640, 3), 255, dtype=np.uint8)
    drawing = False
    last_point = None

    def paint(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal drawing, last_point
        del flags, param

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing and last_point is not None:
            cv2.line(canvas, last_point, (x, y), (40, 40, 220), 6, cv2.LINE_AA)
            last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            last_point = None

    cv2.namedWindow("Paint Brush")
    cv2.setMouseCallback("Paint Brush", paint)

    print("Drag left mouse button to paint | Press 'c' to clear | Press 's' to save | Press 'q' to quit")

    while True:
        preview = canvas.copy()
        cv2.putText(
            preview,
            "Drag to paint | c: clear | s: save | q: quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Paint Brush", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            canvas[:] = 255
        elif key == ord("s"):
            output_path = OUTPUT_DIR / "paint_brush_canvas.png"
            cv2.imwrite(str(output_path), canvas)
            print(f"Saved drawing to: {output_path}")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
