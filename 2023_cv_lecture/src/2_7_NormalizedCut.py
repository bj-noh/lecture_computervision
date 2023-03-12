import cv2
import numpy as np
import skimage
import time


def main():
    # Image Load
    image = skimage.data.coffee()
    cv2.imshow('Coffee image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    start = time.time()

    slic = skimage.segmentation.slic(image, compactness = 20, n_segments = 600, start_label = 1)

    g = skimage.graph.rag_mean_color(image, slic, mode = 'similarity')
    ncut = skimage.graph.cut_normalized(slic, g)
    print(image.shape, 'Coffee 영상 분할 처리 시간: ', time.time() - start , '초')

    marking = skimage.segmentation.mark_boundaries(image, ncut)
    ncut_image = np.uint8(marking * 255.0)

    cv2.imshow('Normalized cut', cv2.cvtColor(ncut_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
