import os
import re

import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def order_points(pts):
    points = np.zeros((4, 2), dtype="int")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    points[0] = pts[np.argmin(s)]
    points[1] = pts[np.argmin(diff)]
    points[2] = pts[np.argmax(s)]
    points[3] = pts[np.argmax(diff)]

    return points


def get_corners(image):
    #rect = None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    #erosion = cv2.erode(blur, np.ones((9, 9), np.uint8), iterations=2)
    #dilation = cv2.dilate(erosion, np.ones((9, 9), np.uint8), iterations=2)

   # _, thresh = cv2.threshold(
   #     blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 85, 1)

    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    c = max(contours, key=cv2.contourArea)

    #x, y, w, h = cv2.boundingRect(c)
    #bbox = x, y, w, h

    #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    accuracy = 0.04*cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, accuracy, True)

    corners_img = image.copy()
    if len(approx) == 4:
        #cv2.drawContours(image, [approx], 0,  (0, 255, 0), 2)
        approx = np.reshape(approx, (4, 2))
        rect = order_points(approx)

        for i, corner in enumerate(rect):

            x_corner = int(corner[0])
            y_corner = int(corner[1])

            cv2.circle(corners_img, (x_corner, y_corner), 1, (255, 0, 0), 2)
            cv2.putText(corners_img, f'{i}', (x_corner, y_corner),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=(255, 0, 0))

        #cv2.imshow("corners_img", corners_img)
        #cv2.waitKey(0)
        return rect

#path da modificare
im_path = "D:\\CECILIA\\Desktop\\Vision and Cognitive Systems\\progetto-vision\\Project material\\images"
im_list = glob.glob(f'{im_path}/*.png', recursive=False)
output_path = "D:\\CECILIA\\Desktop\\Vision and Cognitive Systems\\progetto-vision\\painting_rect"
#file_path = "D:\\CECILIA\\Desktop\\Vision and Cognitive Systems\\progetto-vision\\Project material\\000 - 014 correct_bb\\labels"
#file_list = glob.glob(f'{im_path}/*.txt', recursive=False)

'''file_list = []
im_list = []

for file in os.listdir(im_path):
    if file.endswith('.txt'):
        file_list.append(file)
    else:
        im_list.append(file)'''
#file_list.sort(key=natural_keys)
im_list.sort(key=natural_keys)


for image in im_list:
    im_name = os.path.splitext(os.path.basename(image))[0]

    file = open(im_path + '\\' + im_name + ".txt", 'r')
    txt = file.readlines()

    im = cv2.imread(image)
    im_h, im_w, _ = im.shape

    for i in range(len(txt)): #per ogni riga del file, quindi per ogni bounding box

        coordinates = txt[i].split()

        if coordinates[0] == '0': #considero solo i quadri, quindi id=0

            #prendo le coordinate dal file txt e le rendo non normalizzate
            ncx, ncy, nw, nh = float(coordinates[1]), float(coordinates[2]), float(coordinates[3]), float(coordinates[4])

            w = nw * im_w
            h = nh * im_h

            cx = ncx * im_w
            cy = ncy * im_h
            #cx, cy = float(coordinates[1]) * im.shape[1], float(coordinates[2]) * im.shape[0]
            # calcolo l'angolo in alto a sinistra

            x1 = int(cx - (float(w) / 2))
            y1 = int(cy - (float(h) / 2))
            x2 = int(x1 + float(w))
            y2 = int(y1 + float(h))

            # taglio l'immagine tenendo solo la bounding box
            crop = im[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]

            #cv2.imshow('crop', crop)
            #cv2.waitKey(0)
            #src, bbox = get_corners(crop, draw=True)
            try:
                src = get_corners(crop)
                #print('src: ', src)
                (tl, tr, br, bl) = src
            except:
                continue
            #print(src)
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="int")
            ret, _ = cv2.findHomography(src, dst, cv2.RANSAC)

            #x, y, w, h = bbox
            #dst = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            #ret = cv2.getPerspectiveTransform(src, dst)

            warped = cv2.warpPerspective(crop, ret, (maxWidth, maxHeight))

            cv2.imwrite(
                f'{output_path}/{im_name}-rect{i}-.png', warped)
            print(f'{im_name}-rect{i}-.png')
            cv2.imshow('warped', warped)
            cv2.waitKey(0)