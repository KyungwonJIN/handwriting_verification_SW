import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic

import time
from random import *
## import myutil
from siamese_predict import *
from svm_model_load import *
from crop_image import *
from resize_image_square import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
CUDA_VISIBLE_DEVICES = 0
K.clear_session()


def make_randf():
    a = randrange(1000)
    print(f'{a:.4f}')
    return a

main_ui = uic.loadUiType('main.ui')[0]

class OptionWindow(QDialog):
    def __init__(self, parent):
        super(OptionWindow, self).__init__(parent)
        option_ui = './option.ui'
        uic.loadUi(option_ui, self)
        self.setWindowTitle('Image processing')
        self.setWindowIcon(QIcon('brush.jpg'))
        self.show()

        # 변수 설정
        self.binary_resize_crop_path = './image/connect/_crop.jpg'
        self.image_path2=''
        self.thresh = [130, 255]  # [min, max]
        self.horizontalSlider_threshold.setValue(self.thresh[0])
        self.roi_coord = []
        self.count_num = 0
        self.label1=False
        self.label2=False
        self.label3=False
        # 사진 삽입
        self.pushButton_image_select1.clicked.connect(self._showImage)
        self.pushButton_change_image.clicked.connect(self._showImage)
        self.pushButton_deleteImage_both.clicked.connect(self.deleteImage)
        self.pushButton_crop_image.clicked.connect(self._showImage)
        self.pushButton_save.clicked.connect(self._showImage)

        # 슬라이더 기능
        self.horizontalSlider_threshold.valueChanged.connect(self.horizontalSlider_show)
    def horizontalSlider_show(self):
        self.thresh[0] = self.horizontalSlider_threshold.value()
        self.label_maxThr.setText(f'{self.thresh[0]}')

    def getChar(self, img_path):
        res = []
        rectangle = []
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img_size = img.shape[0] * img.shape[1]
        ## 이진화 된 글자를 팽창 시켜 하나의 contour로 뽑을 수 있게 함
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.erode(img, kernel, iterations=50)

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
                res.append(((int(x + radius), int(y + radius)), int(1 * radius)))
                rectangle.append(((x, y), (x + width, y + height)))
        tmp = 0
        temp_list = []
        for i in range(len(rectangle)):
            if rectangle[i][1][0] - rectangle[i][0][0] > tmp:
                temp_list.append(rectangle[i])
                tmp = rectangle[i][1][0] - rectangle[i][0][0]
        rectangle = temp_list.pop()
        return rectangle


    def _showImage(self):
        sender = self.sender()
        print('sender = ', sender)
        if sender is self.pushButton_image_select1:
            self.image_path2 = \
            QFileDialog.getOpenFileNames(self, 'Get Images', './image/original_image', 'Image Files(*.png *.jpg *.bmp)')[0]
            if self.image_path2:
                self.label1=True
                self.image_path2 = self.image_path2[0]
                self.qPixmapVar = QPixmap()
                self.qPixmapVar.load(self.image_path2)
                self.Image_before.setScaledContents(True)
                self.Image_before.setPixmap(self.qPixmapVar)
                # self.I1_path = self.image_path

            else:
                QMessageBox.warning(self, "QMessageBox", "이미지 선택 안됨", QMessageBox.Ok)

        elif sender is self.pushButton_change_image:
            if self.label1 and self.image_path2:
                self.label2 = True
                self.img = cv2.imread(self.image_path2)
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                ret, self.img_thresh = cv2.threshold(self.img, self.thresh[0], 255, cv2.THRESH_BINARY)
                h, w, c = self.img_thresh.shape
                self.qImg = QImage(self.img_thresh.data, w, h, w * c, QImage.Format_RGB888)
                self.pixmap = QPixmap.fromImage(self.qImg)
                self.pixmap.save('./image/connect/pixmapsave.jpg')
                self.ori_img = cv2.imread('./image/connect/pixmapsave.jpg', cv2.IMREAD_GRAYSCALE)
                self.Image_after.setPixmap(self.pixmap)
            else:
                QMessageBox.warning(self, "QMessageBox", "이미지를 먼저 선택해야 합니다", QMessageBox.Ok)

        elif sender is self.pushButton_crop_image:
            if self.label2 and self.image_path2:
                self.label3 = True
                ''' 여기서 부터 crop resize'''
                self.count_num += 1

                img_path = './image/connect/pixmapsave.jpg'

                rectxy = self.getChar(img_path)
                rect_x = rectxy[0]
                rect_y = rectxy[1]

                self.image_cropped = cv2.imread(img_path)
                cropped_img = self.image_cropped[rect_x[1]:rect_y[1], rect_x[0]:rect_y[0]]
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(self.binary_resize_crop_path, cropped_img)
                rand_num= make_randf()
                self.final_image_path = resize_image(self.binary_resize_crop_path, rand_num)
                self.final_image = QPixmap()
                print(self.final_image_path)
                self.path_final = './image/connect/'+self.final_image_path
                print(self.path_final)
                self.final_image.load(self.path_final)
                self.Image_after_2.setPixmap(self.final_image)
            else:
                QMessageBox.warning(self, "QMessageBox", "이진화를 먼저 해야합니다", QMessageBox.Ok)
        elif sender is self.pushButton_save:
            if self.label3 and self.image_path2:
                self.final_image.save('./image/final_image/'+self.final_image_path)
            else:
                QMessageBox.warning(self, "QMessageBox", "crop&resize 먼저 진행해야 합니다", QMessageBox.Ok)




    def deleteImage(self):
        self.image_path2=''
        self.label1=False
        self.label2=False
        self.label3=False
        self.Image_before.clear()
        self.Image_after.clear()
        self.Image_after_2.clear()



class MyApp(QMainWindow, main_ui):
    def __init__(self):
        super(MyApp, self).__init__()
        # Window 초기화
        self.setupUi(self)
        self.initUI()

        #hyper parameter
        self.init_dir = './'
        self.extensions = ['.jpg']


        ## 변수 초기화 :PyQt
        self.clicked = False
        self.image_path = []
        self.ori_img = None
        self.qlabel_num = 0

        # 이미지 경로 변수
        self.I1_path = ''
        self.I2_path = ''
        self.I3_path = ''
        self.I4_path = ''
        self.I5_path = ''
        self.I6_path = ''
        self.I7_path = ''
        self.I8_path = ''

        # 이미지 유무 체크 변수
        self.I1 = False
        self.I2 = False
        self.I3 = False
        self.I4 = False
        self.I5 = False
        self.I6 = False
        self.I7 = False
        self.I8 = False

        # 모델 로드 됐는지 확인하는 변수
        self.IsModelLoad = False

        # 단일 예측 모델 저장
        self.single_model = None
        # 단일 예측값 4개
        self.single_predict_value1 = 0
        self.single_predict_value2 = 0
        self.single_predict_value3 = 0
        self.single_predict_value4 = 0
        # check box 상태 확인 변수
        self.checkbox_1_state = False
        self.checkbox_2_state = False
        self.checkbox_3_state = False
        self.checkbox_4_state = False
        # check box 갯수 확인 변수
        self.num_checkbox = 0
        # 단일 예측값 저장하는 리스트
        self.single_predict_list = []
        # 임시로 예측 값 저장
        self.temp1 = ''
        self.temp2 = ''
        self.temp3 = ''
        self.temp4 = ''

        ###
        self.pushButton_option.clicked.connect(self.clicked_option)

        ## 버튼에 기능 연결
        # 종료버튼
        self.pushButton_quit.clicked.connect(self.program_quit)
        self.pushButton_clear_checkbox.clicked.connect(self.clear_checkbox)

        # 이미지 선택해서 띄우는 기능 (다른 이미지로 교체 가능)
        self.pushButton_selectImage1.clicked.connect(self._showImage)
        self.pushButton_selectImage2.clicked.connect(self._showImage)
        self.pushButton_selectImage3.clicked.connect(self._showImage)
        self.pushButton_selectImage4.clicked.connect(self._showImage)
        self.pushButton_selectImage5.clicked.connect(self._showImage)
        self.pushButton_selectImage6.clicked.connect(self._showImage)
        self.pushButton_selectImage7.clicked.connect(self._showImage)
        self.pushButton_selectImage8.clicked.connect(self._showImage)

        # 띄워놓은 이미지 지우는 기능
        self.pushButton_deleteImage1.clicked.connect(self.deleteImage)
        self.pushButton_deleteImage2.clicked.connect(self.deleteImage)
        self.pushButton_deleteImage3.clicked.connect(self.deleteImage)
        self.pushButton_deleteImage4.clicked.connect(self.deleteImage)
        self.pushButton_deleteImage5.clicked.connect(self.deleteImage)
        self.pushButton_deleteImage6.clicked.connect(self.deleteImage)
        self.pushButton_deleteImage7.clicked.connect(self.deleteImage)
        self.pushButton_deleteImage8.clicked.connect(self.deleteImage)

        # 단일 예측값 출력 버튼
        self.pushButton_predict_1.clicked.connect(self.printPredict)
        self.pushButton_predict_2.clicked.connect(self.printPredict)
        self.pushButton_predict_3.clicked.connect(self.printPredict)
        self.pushButton_predict_4.clicked.connect(self.printPredict)

        # 단일 예측 모델 로드
        self.pushButton_Load_single_model.clicked.connect(self.loadModel_single)

        # checkbox 연결 + 기본 상태
        self.checkBox_1.setCheckable(False)
        self.checkBox_2.setCheckable(False)
        self.checkBox_3.setCheckable(False)
        self.checkBox_4.setCheckable(False)
        self.checkBox_1.stateChanged.connect(self.checkBoxState)
        self.checkBox_2.stateChanged.connect(self.checkBoxState)
        self.checkBox_3.stateChanged.connect(self.checkBoxState)
        self.checkBox_4.stateChanged.connect(self.checkBoxState)

        # 멀티모달 예측 버튼
        self.pushButton_svm_1.clicked.connect(self.multimodalPredict)

        text = f'set 1 predict value :{self.single_predict_value1:.4f}\nSet 2 predict value :{self.single_predict_value2:.4f}\nSet 3 predict value :{self.single_predict_value3:.4f}\nSet 4 predict value :{self.single_predict_value4:.4f}'
        self.label_predictList.setText(text)
    def clicked_option(self):
        OptionWindow(self)

    def _showImage(self):
        sender = self.sender()
        self.image_path = QFileDialog.getOpenFileNames(self, 'Get Images', './image/final_image','Image Files(*.png *.jpg *.bmp)')[0]
        if self.image_path:
            self.image_path = self.image_path[0]
            self.qPixmapVar = QPixmap()
            self.qPixmapVar.load(self.image_path)
            if sender is self.pushButton_selectImage1:
                self.Image1.setPixmap(self.qPixmapVar)
                self.I1_path = self.image_path
                self.I1=True
                if self.checkBox_1.isChecked():
                    self.checkBox_1.setChecked(False)
                    self.checkBox_1.setCheckable(False)
            elif sender is self.pushButton_selectImage2:
                self.Image2.setPixmap(self.qPixmapVar)
                self.I2_path = self.image_path
                self.I2=True
                if self.checkBox_1.isChecked():
                    self.checkBox_1.setChecked(False)
                    self.checkBox_1.setCheckable(False)
            elif sender is self.pushButton_selectImage3:
                self.Image3.setPixmap(self.qPixmapVar)
                self.I3_path = self.image_path
                self.I3=True
                if self.checkBox_2.isChecked():
                    self.checkBox_2.setChecked(False)
                    self.checkBox_2.setCheckable(False)

            elif sender is self.pushButton_selectImage4:
                self.Image4.setPixmap(self.qPixmapVar)
                self.I4_path = self.image_path
                self.I4=True
                if self.checkBox_2.isChecked():
                    self.checkBox_2.setChecked(False)
                    self.checkBox_2.setCheckable(False)

            elif sender is self.pushButton_selectImage5:
                self.Image5.setPixmap(self.qPixmapVar)
                self.I5_path = self.image_path
                self.I5=True
                if self.checkBox_3.isChecked():
                    self.checkBox_3.setChecked(False)
                    self.checkBox_3.setCheckable(False)

            elif sender is self.pushButton_selectImage6:
                self.Image6.setPixmap(self.qPixmapVar)
                self.I6_path = self.image_path
                self.I6=True
                if self.checkBox_3.isChecked():
                    self.checkBox_3.setChecked(False)
                    self.checkBox_3.setCheckable(False)
            elif sender is self.pushButton_selectImage7:
                self.Image7.setPixmap(self.qPixmapVar)
                self.I7_path = self.image_path
                self.I7=True
                if self.checkBox_4.isChecked():
                    self.checkBox_4.setChecked(False)
                    self.checkBox_4.setCheckable(False)
            elif sender is self.pushButton_selectImage8:
                self.Image8.setPixmap(self.qPixmapVar)
                self.I8_path = self.image_path
                self.I8=True
                if self.checkBox_4.isChecked():
                    self.checkBox_4.setChecked(False)
                    self.checkBox_4.setCheckable(False)
        else:
            QMessageBox.warning(self, "QMessageBox", "이미지 선택 안됨", QMessageBox.Ok)
    def clear_checkbox(self):
        self.checkBox_1.setChecked(False)
        self.checkBox_1.setCheckable(False)
        self.checkBox_2.setChecked(False)
        self.checkBox_2.setCheckable(False)
        self.checkBox_3.setChecked(False)
        self.checkBox_3.setCheckable(False)
        self.checkBox_4.setChecked(False)
        self.checkBox_4.setCheckable(False)
        self.label_mm_predict.setText('_')

    def deleteImage(self):
        sender = self.sender()
        if sender is self.pushButton_deleteImage1:
            self.I1_path = ''
            self.Image1.clear()
            self.I1 = False
            if self.checkBox_1.isChecked():
                self.checkBox_1.setChecked(False)
                self.checkBox_1.setCheckable(False)

        elif sender is self.pushButton_deleteImage2:
            self.I2_path = ''
            self.Image2.clear()
            self.I2 = False
            if self.checkBox_1.isChecked():
                self.checkBox_1.setChecked(False)
                self.checkBox_1.setCheckable(False)

        elif sender is self.pushButton_deleteImage3:
            self.I3_path = ''
            self.Image3.clear()
            self.I3 = False
            if self.checkBox_2.isChecked():
                self.checkBox_2.setChecked(False)
                self.checkBox_2.setCheckable(False)

        elif sender is self.pushButton_deleteImage4:
            self.I4_path = ''
            self.Image4.clear()
            self.I4 = False
            if self.checkBox_2.isChecked():
                self.checkBox_2.setChecked(False)
                self.checkBox_2.setCheckable(False)

        elif sender is self.pushButton_deleteImage5:
            self.I5_path = ''
            self.Image1.clear()
            self.I5 = False
            if self.checkBox_3.isChecked():
                self.checkBox_3.setChecked(False)
                self.checkBox_3.setCheckable(False)

        elif sender is self.pushButton_deleteImage6:
            self.I6_path = ''
            self.Image6.clear()
            self.I6 = False
            if self.checkBox_3.isChecked():
                self.checkBox_3.setChecked(False)
                self.checkBox_3.setCheckable(False)

        elif sender is self.pushButton_deleteImage7:
            self.I7_path = ''
            self.Image7.clear()
            self.I7 = False
            if self.checkBox_4.isChecked():
                self.checkBox_4.setChecked(False)
                self.checkBox_4.setCheckable(False)

        elif sender is self.pushButton_deleteImage8:
            self.I8_path = ''
            self.Image8.clear()
            self.I8 = False
            if self.checkBox_4.isChecked():
                self.checkBox_4.setChecked(False)
                self.checkBox_4.setCheckable(False)

    def printPredict(self):
        sender = self.sender()
        # 모델 로드 됐는지 check
        if self.IsModelLoad is True:
            # 입력받은 버튼 확인
            if sender is self.pushButton_predict_1:
                # 이미지1과 2 둘다 불러왔는지 check
                if self.I1 is True and self.I2 is True:
                    # 이미지 둘다 불러와있으면 checkbox 활성화
                    self.checkBox_1.setCheckable(True)
                    ''' 
                    siamese_predict.py 에서 정의한 함수에 이미지 경로와
                    불러온 모델을 넣어 예측값을 반환
                    '''
                    self.single_predict_value1 = f'{print_predict_value(self.I1_path, self.I2_path, self.loaded_model):.4f}'

                    # 라벨에 예측값 출력
                    self.label_predict_value_1.setText(self.single_predict_value1)
                    if self.single_predict_value1 in self.single_predict_list:
                        self.single_predict_list.remove(self.single_predict_value1)
                    if self.checkBox_1.isChecked():
                        self.checkBox_1.setChecked(False)

                    # 라벨에 리스트 출력
                    self.label_predictList.setText('set 1 predict value :{}\nSet 2 predict value :{}\nSet 3 predict value :{}\nSet 4 predict value :{}'.format(self.single_predict_value1,self.single_predict_value2,self.single_predict_value3,self.single_predict_value4))
                else:
                    QMessageBox.warning(self, "QMessageBox", "Please put Images", QMessageBox.Ok)
            elif sender is self.pushButton_predict_2:
                if self.I3 is True and self.I4 is True:
                    self.checkBox_2.setCheckable(True)
                    self.single_predict_value2 = f'{print_predict_value(self.I3_path, self.I4_path, self.loaded_model):.4f}'
                    self.label_predict_value_2.setText(self.single_predict_value2)
                    if self.single_predict_value2 in self.single_predict_list:
                        self.single_predict_list.remove(self.single_predict_value2)
                    if self.checkBox_2.isChecked():
                        self.checkBox_2.setChecked(False)
                    self.label_predictList.setText(
                        'set 1 predict value :{}\nSet 2 predict value :{}\nSet 3 predict value :{}\nSet 4 predict value :{}'.format(
                            self.single_predict_value1, self.single_predict_value2, self.single_predict_value3,
                            self.single_predict_value4))

                else:
                    QMessageBox.warning(self, "QMessageBox", "Please put Images", QMessageBox.Ok)
            elif sender is self.pushButton_predict_3:
                if self.I5 is True and self.I6 is True:
                    self.checkBox_3.setCheckable(True)
                    self.single_predict_value3 = f'{print_predict_value(self.I5_path, self.I6_path, self.loaded_model):.4f}'
                    self.label_predict_value_3.setText(self.single_predict_value3)
                    if self.single_predict_value3 in self.single_predict_list:
                        self.single_predict_list.remove(self.single_predict_value3)
                    if self.checkBox_3.isChecked():
                        self.checkBox_3.setChecked(False)
                    self.label_predictList.setText(
                        'set 1 predict value :{}\nSet 2 predict value :{}\nSet 3 predict value :{}\nSet 4 predict value :{}'.format(
                            self.single_predict_value1, self.single_predict_value2, self.single_predict_value3,
                            self.single_predict_value4))

                else:
                    QMessageBox.warning(self, "QMessageBox", "Please put Images", QMessageBox.Ok)
            elif sender is self.pushButton_predict_4:
                if self.I7 is True and self.I8 is True:
                    self.checkBox_4.setCheckable(True)
                    self.single_predict_value4 = f'{print_predict_value(self.I7_path, self.I8_path, self.loaded_model):.4f}'
                    self.label_predict_value_4.setText(self.single_predict_value4)
                    if self.single_predict_value4 in self.single_predict_list:
                        self.single_predict_list.remove(self.single_predict_value4)
                    if self.checkBox_4.isChecked():
                        self.checkBox_4.setChecked(False)
                    self.label_predictList.setText(
                        'set 1 predict value :{}\nSet 2 predict value :{}\nSet 3 predict value :{}\nSet 4 predict value :{}'.format(
                            self.single_predict_value1, self.single_predict_value2, self.single_predict_value3,
                            self.single_predict_value4))

                else:
                    QMessageBox.warning(self, "QMessageBox", "Please put Images", QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "QMessageBox", "Please 'Load Model' first", QMessageBox.Ok)
    def loadModel_single(self):
        self.single_model_path = QFileDialog.getOpenFileNames(self, 'Select model', './model_weight')[0]
        if self.single_model_path:
            start_time = time.time()
            self.loaded_model = load_model_(self.single_model_path[0])
            QMessageBox.about(self, "QMessageBox", "Complete load model")

            self.IsModelLoad = True
        else:
            QMessageBox.warning(self, "QMessageBox", "모델이 선택되지 않았습니다.", QMessageBox.Ok)

    def checkBoxState(self, state):
        sender = self.sender()
        if sender == self.checkBox_1:
            if self.checkBox_1.isChecked():
                self.checkbox_1_state = True
                self.num_checkbox +=1
                self.single_predict_list.append(self.single_predict_value1)
            else:
                self.checkbox_1_state = False
                self.num_checkbox -=1
                if self.single_predict_value1 in self.single_predict_list:
                    self.single_predict_list.remove(self.single_predict_value1)
        elif sender == self.checkBox_2:
            if self.checkBox_2.isChecked():
                self.checkbox_2_state = True
                self.num_checkbox +=1
                self.single_predict_list.append(self.single_predict_value2)
            else:
                self.checkBox_2_state = False
                self.num_checkbox -=1
                if self.single_predict_value2 in self.single_predict_list:
                    self.single_predict_list.remove(self.single_predict_value2)
        elif sender == self.checkBox_3:
            if self.checkBox_3.isChecked():
                self.checkbox_3_state = True
                self.num_checkbox +=1
                self.single_predict_list.append(self.single_predict_value3)
            else:
                self.checkBox_3_state = False
                self.num_checkbox -=1
                if self.single_predict_value3 in self.single_predict_list:
                    self.single_predict_list.remove(self.single_predict_value3)
        elif sender == self.checkBox_4:
            if self.checkBox_4.isChecked():
                self.checkbox_4_state = True
                self.num_checkbox +=1
                self.single_predict_list.append(self.single_predict_value4)
            else:
                self.checkBox_4_state = False
                self.num_checkbox -=1
                if self.single_predict_value4 in self.single_predict_list:
                    self.single_predict_list.remove(self.single_predict_value4)

    # 멀티 모달 예측 버튼
    def multimodalPredict(self):
        print(self.single_predict_list)
        if len(self.single_predict_list) <=1:
            QMessageBox.warning(self, "QMessageBox", "Please check at least two checkboxes", QMessageBox.Ok)
        else:
            mm_predict_value = f'{multimodal(self.single_predict_list)}'
            print(mm_predict_value)
            if mm_predict_value == 0:
                QMessageBox.warning(self, "QMessageBox", "Please check at least two checkboxes", QMessageBox.Ok)
            else:
                self.label_mm_predict.setText(mm_predict_value)
    # 새창 열기
    def program_quit(self):
        self.press_esc = True
        QCoreApplication.instance().quit()

    def initUI(self):
        self.setWindowTitle('멀티모달을 이용한 필적 분석')
        self.setWindowIcon(QIcon('korean.png'))
        self.setGeometry(400, 200, 798, 619)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    myWindow = MyApp()
    myWindow.show()
    sys.exit(app.exec_())
