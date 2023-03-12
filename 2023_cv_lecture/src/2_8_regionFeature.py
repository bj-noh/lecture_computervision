import cv2
import numpy as np
import skimage
import time


def main():
    
    # Image Load
    origin_image = skimage.data.horse()

    image = 255 - np.uint8(origin_image) * 255

    cv2.imshow('Horse', image)

    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    image2 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image2, contours, -1, (255, 0, 255), 2)
    cv2.imshow('Horse with contour', image2)

    contour = contours[0]
    m = cv2.moments(contour)
    area = cv2.contourArea(contour)

    cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
    
    perimeter = cv2.arcLength(contour, True)
    roundness = (4.0 * np.pi * area) / (perimeter * perimeter)

    print('Area = ', area, '\nCenter Point = (', cx, ', ', cy, ')', '\nPerimeter = ', perimeter, '\nRoundness = ', roundness)

    image3 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    contour_approx = cv2.approxPolyDP(contour, 8, True)
    cv2.drawContours(image3, [contour_approx], -1, (0, 255, 0), 2)

    hull = cv2.convexHull(contour)
    hull = hull.reshape(1, hull.shape[0], hull.shape[2])

    cv2.drawContours(image3, hull, -1, (0, 0, 255), 2)

    cv2.imshow('Horse with line segmenets and convex hull', image3)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
