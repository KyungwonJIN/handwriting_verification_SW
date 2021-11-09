import numpy as np
import cv2
import matplotlib as plt
import math

def getChar(img, img_size):
    res = []
    rectangle = []

    # ## 이미지 히스토그램 mean 값을 threshold로 받아서 이진화
    # ret, thresh_gray = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    #
    # ## 이진화 된 글자를 팽창 시켜 하나의 contour로 뽑을 수 있게 함
    # kernel = np.ones((5, 5), np.uint8)
    # thresh_gray = cv2.erode(thresh_gray, kernel, iterations=50)

    ## 경계선을 검출함
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    ## 검출된 경계선 중 최외각 경계선을 제외하고 글자 부분의 경계선만 남김
    for contour in contours:
        area = cv2.contourArea(contour)
        rect = cv2.boundingRect(contour)
        x, y, width, height = rect
        radius = 0.25 * (width + height)

        area_condition = (img_size / 2 >= area)
        fill_condition = (abs(1 - (area / (math.pi * math.pow(radius, 2.0)))) <= 0.3)
        if area_condition and fill_condition:
        # if area_condition:
            res.append(((int(x + radius), int(y + radius)), int(1 * radius)))
            rectangle.append(((x, y), (x + width, y + height)))
    tmp = 0

    temp_list = []
    for i in range(len(rectangle)):
        if rectangle[i][1][0] - rectangle[i][0][0] > tmp:
            # max_rect = rectangle[i]
            temp_list.append(rectangle[i])
            # print(max_rect)
            tmp = rectangle[i][1][0] - rectangle[i][0][0]
    rectangle = temp_list.pop()
    # rectangle = max_rect
    return rectangle
def crop_image(image_from_pixmap):
    image_name = 'kyung1_re'
    img = cv2.imread('./image/' + image_name + '.jpg')
    img = image_from_pixmap
    pr_img = img.copy()
    img_size = image_from_pixmap.shape[0]*image_from_pixmap.shape[1]
    # img_size = img.shape[0]*img.shape[1]
    # ravel()을 통해 img를 1차원 배열로 펼치고 histogram으로 그림
    # x, y = np.histogram(img.ravel())
    # # y, x, _ = plt.hist(img.ravel())
    # mean_histo = np.mean(y)
    # print(mean_histo)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rectxy = getChar(img_gray, img_size)
    print(rectxy)
    rect_x=rectxy[0]
    rect_y=rectxy[1]
    print(rect_x,rect_y)
    # cv2.rectangle(pr_img, rect_x, rect_y, (255,0,0), 1)

    # _, pr_img = cv2.threshold(pr_img, _, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    pr_img = cv2.erode(pr_img, kernel, iterations = 2)
    cropped_img = pr_img[rect_x[1]:rect_y[1], rect_x[0]:rect_y[0]]

    # plt.imshow(cropped_img,'gray')
    # plt.show()
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('./image/' + image_name + '_crop.jpg', cropped_img)
    return cropped_img
    '''
    contour 제거하고 어차피 mean값 뽑아서 이진화면 되면
    그 상태로 그냥 글자 값 읽어서 tight crop 할 수 있었을텐데
    '''
