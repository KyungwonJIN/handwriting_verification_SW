import numpy as np
import cv2
import matplotlib as plt
import random
"""
경로 및 파일 이름 설정 부분
"""
def resize_image(image_path, count_num):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = img.shape[:2]
    # height_bigger = int(height-width)/2
    # width_bigger = int(width-heigth)/2

    if height > width:
        square_img = cv2.copyMakeBorder(img, 0, 0, int((height - width) / 2), int((height - width) / 2),
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255))
    elif width < height:
        square_img = cv2.copyMakeBorder(img, int((width - height) / 2), int((width - height) / 2), 0, 0,
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255))
    else:
        square_img = img

    # resize_img = cv2.resize(img, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
    # print(height , ",", width)
    resize_img = cv2.resize(square_img, dsize=(96, 96), interpolation=cv2.INTER_AREA)

    # cv2.imshow("letter", resize_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    final_image = cv2.copyMakeBorder(resize_img, 8, 8, 8, 8,
                                     cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # savename = './image/' + image_name + '.jpg'
    # savename = "single_test/02100001_1.jpg"
    # k = random.randrange(1,1000)
    k= count_num
    # image_path_final = './image/connect/cropped_'+str(k)+'.jpg'
    image_path_final = 'cropped_'+str(k)+'.jpg'

    cv2.imwrite('./image/connect/'+image_path_final, final_image)
    # cv2.waitKey(0)
    return image_path_final