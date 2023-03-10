import cv2
import sys


def main(): 

    path = './data/dog1.jpg' # Edit your image path
    src = cv2.imread(path)

    # ROI extraction
    x, y, w, h = cv2.selectROI(src)

    # BGR -> YCrCb
    src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

    # crop: area which user selected
    crop = src_ycrcb[y:y+h, x:x+w]

    channels = [1, 2]  
    cr_bins = 128      # cr을 표현하는 범위.
    cb_bins = 128
    histSize = [cr_bins, cb_bins]
    cr_range = [0, 256]
    cb_range = [0, 256]
    ranges = cr_range + cb_range

    # Histogram calculation
    hist = cv2.calcHist([crop], channels, None, histSize, ranges)
    hist_norm = cv2.normalize(cv2.log(hist + 1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Histogram back-projection
    backproj = cv2.calcBackProject([src_ycrcb], channels, hist, ranges, 1)

    # Masking
    dst = cv2.copyTo(src, backproj)

    cv2.imshow('backprj', backproj)
    cv2.imshow('hist_norm', hist_norm)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()

