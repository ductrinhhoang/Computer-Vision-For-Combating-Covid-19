import cv2
import numpy as np
from utils.image import ImageData
from common import face_preprocess
from face_detectors import RetinafaceONNX
from face_processors import FaceEmbeddingBatchONNX
import time

class FaceModelBase:
    def __init__(self):
        self.embedd_model = None
        self.detector_input_size = None
        self.detector = None
        self.embedd_model = None
        self.embedding_batch_size = None

    def visualize_detect(self, base_image, bboxes, points):
        '''
            Visualize detection
        '''
        vis_image = base_image.copy()
        for i in range(len(bboxes)):
            pt1 = tuple(map(int, bboxes[i][0:2]))
            pt2 = tuple(map(int, bboxes[i][2:4]))
            cv2.rectangle(vis_image, pt1, pt2, (0, 255, 0), 1)
            for lm_pt in points[i]:
                cv2.circle(vis_image, tuple(map(int, lm_pt)), 3, (0, 0, 255), 3)
        return vis_image

    def get_inputs(self, bgr_image, threshold = 0.8):  # Only batchsize = 1
        '''
            Get boxes & 5 landmark points
            Input:
                - bgr_image: BGR image
            Output:
                - bboxes: face bounding boxes
                - points: 5 landmark points for each cor-response face
        '''
        # vis_image = bgr_image.copy()
        image = ImageData(bgr_image, self.detector_input_size)
        image.resize_image(mode='pad')
        inp = np.array([image.transformed_image], dtype = np.float32)
        inp = np.transpose(inp, (0, 3, 1, 2))     
        bboxes, points = self.detector.detect(inp, threshold=threshold)
        bboxes = bboxes[0]
        points = points[0]
        # Post processing
        bboxes = bboxes[:, :4]
        bboxes /= image.scale_factor
        points /= image.scale_factor
        del image
        return bboxes, points

    @staticmethod
    def get_face_align(bgr_img, bboxes, points, image_size='112,112'):
        '''
            Align face from given bounding boxes and landmark points
        '''
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        aligned = []
        for i in range(len(bboxes)):
            bbox_i = bboxes[i]
            points_i = points[i]
            nimg = face_preprocess.preprocess(rgb_img, bbox_i, points_i, image_size=image_size)
            aligned.append(nimg)
        return aligned

    def get_features(self, aligned):
        '''
            Extract embedding for face
            Input:
                - aligned: batch of RGB aligned images
        '''
        return self.embedd_model.get_features(aligned)

class FaceModelONNX(FaceModelBase):
    def __init__(self, \
        detector_weights = 'weights/retinaface_r50_v1.onnx',
        detector_input_size = (640, 640), 
        embedding_batch_size = 1,
        embedding_weight = "weights/face_recognition_r34.onnx"):
        '''
            Init Detector & Embedding Extractor
        '''
        super().__init__()
        print('[INFO] Load face detector')
        self.detector_input_size = detector_input_size
        self.detector = RetinafaceONNX(model_path = detector_weights,
                                        input_shape = self.detector_input_size)

        print('[INFO] Load face embedding')
        self.embedding_batch_size = embedding_batch_size
        self.embedd_model = FaceEmbeddingBatchONNX(model_path = embedding_weight,
                                                    batch_size = embedding_batch_size)
        
    
    

class FaceModelTRT(FaceModelBase):
    '''
        Face model with single-inference Retinaface & batch-inference Embedding
    '''
    pass

class FaceModelBatchTRT(FaceModelBase):
    '''
        Face model with batch inference (both Retinaface + Embedding)
    '''
    pass

if __name__ == '__main__':
    import os
    if not os.path.exists('sample'):
        os.mkdir('sample')
    face_model = FaceModelONNX()
    image = cv2.imread('../test_images/face_detector/Stallone.jpg')
    t0 = time.time()
    bboxes, points = face_model.get_inputs(image)
    img_faces = face_model.get_face_align(image, bboxes, points)
    embeds    = face_model.get_features(img_faces)
    t1 = time.time()
    print('Detect & extract embedding cost:',t1-t0)
    for ix, face in enumerate(img_faces):
        bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        cv2.imwrite('sample/face_{}.jpg'.format(ix), bgr)

