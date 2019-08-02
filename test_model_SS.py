# === import packages
import Models, LoadBatches
import os
import cv2
import numpy as np
import pandas as pd
import configuration as config
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from imutils import paths
from watershed import wateshed
from filesNumericalSort import *
from imgPortrait import imgRotate


def test_model_SS():

    n_classes = config.NUM_CLASSES
    model_name = config.MODEL_NAME
    images_path = config.TEST_IMAGES_PATH
    input_width = config.INPUT_WIDTH
    input_height = config.INPUT_HEIGHT
    epoch_number = config.EPOCHS_TEST
    save_weights_path = config.SAVE_WEIGHTS_PATH


    # === load Unet model
    print("[INFO] loading pre-trained network...")
    modelFns = {'vgg_unet': Models.VGGUnet.VGGUnet}
    modelFN = modelFns[model_name]

    m = modelFN(n_classes, input_height=input_height, input_width=input_width)

    m.load_weights(os.path.join(save_weights_path, ("ex." + str(epoch_number))))

    m.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    output_height = m.outputHeight
    output_width = m.outputWidth

    # === load test images
    test_images_path = list(paths.list_images(images_path))
    imageIDs = list()
    NumEars_no_watershed = []
    NumEars_with_watershed = []

    colors = []
    for color_id in range(n_classes):
        if color_id == 0:
            colors.append((0, 0, 0))
        elif color_id == 1:
            colors.append((1, 1, 1))

    for imgPath in sorted(test_images_path, key=numericalSort):
        # outName = imgPath.replace(images_path, result_path)
        imageID = (imgPath.split(os.path.sep)[-1]).split(".")[0]

        print("[INFO] Processing image ... " + imageID)

        # ===== SUPERPIXEL ANALYSIS ====#
        img_orig = cv2.imread(imgPath)

        # === portrait to landscape transformation
        if img_orig.shape[0] > img_orig.shape[1]:
            img_orig = imgRotate(img_orig)

        img = img_orig.copy()

        img_xyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)

        img_float = cv2.normalize(img_xyz.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        # === super segment the image
        superPxl_segments = slic(img_float, n_segments=config.SLIC_SEGMENT,
                                 compactness=config.SLIC_COMPACT, sigma=config.SLIC_SIGMA,
                                 enforce_connectivity=True, convert2lab=True)
        segment_label = np.unique(superPxl_segments)
        # === create an empty mask to copy final results
        img_blobs = np.zeros(img_float.shape[:2], dtype="uint8")

        # plt.imshow(mark_boundaries(img, superPxl_segments))

        for k, region in enumerate(regionprops(superPxl_segments), 1):
            minr, minc, maxr, maxc = region.bbox
            seg_rect = img[minr:maxr, minc:maxc]
            mask = np.zeros(img_float.shape[:2], dtype="uint8")
            mask[superPxl_segments == segment_label[k]] = 255
            mask2 = mask[minr:maxr, minc:maxc]
            img_superPxl = cv2.bitwise_and(seg_rect, seg_rect, mask=mask2)

            # === new mean value calculation instead of fixed used threshold
            X = LoadBatches.getImageArr(img_superPxl, input_width, input_height)
            pr = m.predict(np.array([X]))[0]
            pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

            seg_img = np.zeros((output_height, output_width, 3))
            for c in range(n_classes):
                seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
                seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
                seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

            # === accumulate the result puzzle
            seg_img_resize_down = cv2.resize(seg_img, (img_superPxl.shape[1], img_superPxl.shape[0]),
                                             cv2.INTER_NEAREST).astype("uint8")

            seg_img_resize_down = cv2.cvtColor(seg_img_resize_down, cv2.COLOR_BGR2GRAY)
            thr, superPxl_blob_bw = cv2.threshold(seg_img_resize_down, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            mask_img_seg = np.zeros(img.shape[:2], dtype="uint8")
            mask_img_seg[minr:maxr, minc:maxc] = superPxl_blob_bw

            img_blobs = cv2.bitwise_or(img_blobs, mask_img_seg)

        earNum = 0
        # === watershed analysis
        watershed_numEars = wateshed(img_orig, img_blobs, imageID)


        print("[INFO] Image ID: {}; Num Orig: {}; Num watershed: {}"
              .format(imageID, earNum, watershed_numEars))

        # === store ear numbers and ids into array
        NumEars_no_watershed.append(earNum)
        NumEars_with_watershed.append(watershed_numEars)
        imageIDs.append(imageID)

    # Create a Pandas dataframe and save mean wavelength file
    save_excel_Dir = os.path.join(config.SAVE_RESULTS_PATH, 'Ear_Numbers.xlsx')
    EarCnts_watershed = np.array(NumEars_with_watershed)
    imageIDs = np.array(imageIDs)
    EarCnts_orig = np.array(NumEars_no_watershed)

    df_EarCount = pd.DataFrame({'image ID': imageIDs,
                                'Ear_Num_watershed': EarCnts_watershed,
                                'Ear_Num_orig': EarCnts_orig})

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer1 = pd.ExcelWriter(save_excel_Dir)
    # Convert the dataframe to an XlsxWriter Excel object.
    df_EarCount.to_excel(writer1, sheet_name='Sheet1', index=False, header=False)
    # Close the Pandas Excel writer and output the Excel file.
    writer1.save()
    writer1.close()
