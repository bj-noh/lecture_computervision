import cv2
import numpy as np
import skimage


def main():
    
    # Image Load
    image = skimage.data.coffee()
    cv2.imshow('Coffee image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    slic1 = skimage.segmentation.slic(image, compactness = 20, n_segments = 600)
    sp_image1 = skimage.segmentation.mark_boundaries(image, slic1)
    sp_image1 = np.uint8(sp_image1 * 255.0)

    slic2 = skimage.segmentation.slic(image, compactness = 40, n_segments = 600)
    sp_image2 = skimage.segmentation.mark_boundaries(image, slic2)
    sp_image2 = np.uint8(sp_image2 * 255.0)


    cv2.imshow('Super pixel (compar 40)', cv2.cvtColor(sp_image1, cv2.COLOR_RGB2BGR))
    cv2.imshow('Super pixel (compar 80)', cv2.cvtColor(sp_image2, cv2.COLOR_RGB2BGR))
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
