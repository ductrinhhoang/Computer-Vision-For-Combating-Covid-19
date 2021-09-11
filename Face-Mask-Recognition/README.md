## Face Mask Recognition
### Requirements
- onnxruntime
### Pretrained
- Link: https://drive.google.com/drive/folders/1Wqdsp_oMSPSPcEa-CjrlnkLhCqv-__Wv?usp=sharing
- Download **retinaface_r50_v1.onnx** and **face_recognition_r34.onnx**, put into **src/weights**
### Run
#### Step 1: Extract sample database embeddings from **database/images** 
```
  cd src
  python extract_database.py 
```
To save image & reference images, use flag --save_crop and --use_aug
```
  cd src
  python extract_database.py --save_crop --use_aug
```
More info
```
  python extract_database.py --help
```
this code will create .mat file in **database/mats**
#### Step 2: Test with query images
Test image in **test_images/recognition** folder
```
  cd src
  python detect_and_recognize.py
```
More info
```
  python detect_and_recognize.py --help
```
### Reference
- Insightface: https://github.com/deepinsight/insightface
