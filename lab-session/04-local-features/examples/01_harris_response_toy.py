from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    image = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    uy = np.array([[-1, 0, 1]], dtype=np.float32).T
    ux = np.array([[-1, 0, 1]], dtype=np.float32)
    g = np.outer(cv2.getGaussianKernel(3, 1), cv2.getGaussianKernel(3, 1))

    dy = cv2.filter2D(image, cv2.CV_32F, uy)
    dx = cv2.filter2D(image, cv2.CV_32F, ux)

    dyy = dy * dy
    dxx = dx * dx
    dyx = dy * dx

    gdyy = cv2.filter2D(dyy, cv2.CV_32F, g)
    gdxx = cv2.filter2D(dxx, cv2.CV_32F, g)
    gdyx = cv2.filter2D(dyx, cv2.CV_32F, g)

    response = (gdyy * gdxx - gdyx * gdyx) - 0.04 * (gdyy + gdxx) ** 2

    marked = image.copy()
    for y in range(1, response.shape[0] - 1):
        for x in range(1, response.shape[1] - 1):
            neighborhood = response[y - 1 : y + 2, x - 1 : x + 2]
            if response[y, x] > 0.1 and np.count_nonzero(response[y, x] > neighborhood) >= 8:
                marked[y, x] = 9.0

    response_norm = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    response_color = cv2.applyColorMap(cv2.resize(response_norm, (320, 320), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_JET)
    pattern = cv2.resize(np.uint8(marked / marked.max() * 255), (320, 320), interpolation=cv2.INTER_NEAREST)
    pattern = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)

    canvas = np.hstack([pattern, response_color])
    output_path = OUTPUT_DIR / "harris_toy_response.png"
    cv2.imwrite(str(output_path), canvas)

    cv2.imshow("Toy Pattern | Harris Response", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    np.set_printoptions(precision=2, suppress=True)
    print(response)
    print(f"Saved toy Harris response to: {output_path}")


if __name__ == "__main__":
    main()
