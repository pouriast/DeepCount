# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import cv2
import matplotlib.pyplot as plt
import configuration as config
import os


def wateshed(visImg, binaryImg, imageID):

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this distance map
    D = ndimage.distance_transform_edt(binaryImg)
    localMax = peak_local_max(D, indices=False, min_distance=config.watershed_Thr, labels=binaryImg)

    # plt.imshow(-D, cmap=plt.cm.gray, interpolation='nearest')

    # perform a connected component analysis on the local peaks, using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=binaryImg)
    # print("[INFO] {} unique watershed segments found".format(len(np.unique(labels)) - 1))

    earNum = 1
    # loop over the unique labels returned by the Watershed algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background' so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw it on the mask
        gray = cv2.cvtColor(visImg, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # loop over the contours again
        for (cntsNum, cc) in enumerate(cnts):
            # compute the area and the perimeter of the contour
            area = cv2.contourArea(cc)
            perimeter = cv2.arcLength(cc, True)
            # print "Contour #%d -- area: %.2f, perimeter: %.2f" % (earNum + 1, area, perimeter)

            # === remove small blobs
            if area >= config.NOISE_Thr:
                try:
                    # draw the contour on the image
                    cv2.drawContours(visImg, [cc], -1, (193, 182, 255), 1)

                    # compute the center of the contour and draw the contour number
                    M = cv2.moments(cc)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(visImg, "#%d" % (earNum + 1), (cX - 20, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (133, 21, 199), 2)
                    earNum += 1
                except:
                    print("area error")

    resultID = imageID + "_EarNo_ {} .jpg".format(earNum)
    saveDir = os.path.join(config.SAVE_RESULTS_PATH, resultID)

    cv2.imwrite(saveDir, visImg)

    return earNum
