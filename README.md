# handwriting_verification_SW

## main_copy2.py
ㅡ 전체 UI 및 구동 파일

## siamese_predict.py
ㅡ 학습된 모델 weight를 불러오고 2개의 필적을 받아 예측 값을 return 함

## svm_model_load.py
ㅡ SVM으로 학습된 모델을 불러와서 다수의 예측 값을 결합하여 새로운 예측 값을 return 함

## resize_image_square.py, crop_image.py
ㅡ 불러온 원본 이미지를 crop하고 학습에 맞는 크기로 resize 시켜준다
