import os
import sys
import cv2
import numpy as np
import torch
import time
import backbones
import sklearn.preprocessing
import onnxruntime
# import tensorrt as trt        # Don't use TENSORRT
# from exec_backends.trt_backend import alloc_buf, inference
# from exec_backends.trt_loader import TrtModel

class FaceEmbedding(object):
    '''
     Face Embedding model using Pytorch Backend
    '''
    def __init__(self, model_path, gpu_id = 0, network = 'iresnet124', data_shape = (3, 112, 112)):
        image_size = (112, 112)
        self.image_size = image_size
        self.device = torch.device('cuda:{}'.format(gpu_id))
        weight = torch.load(model_path)
        self.model = eval("backbones.{}".format(network))(False)
        self.model.load_state_dict(weight)
        self.model.to(self.device)
        self.model.eval()

    def check_assertion(self, feats, eps = 1e-6):
        '''
            Make sure that face embedding model (or code) work normally
        '''
        for ix, feat in enumerate(feats):
            assert np.fabs(np.linalg.norm(feat) - 1.0) < eps, print(ix, np.linalg.norm(feat))

    @torch.no_grad()
    def get_features(self, batch_data, batch_size = 32, feat_dim = 512):
        '''
            Input: List of RGB image BxHxWx3
        '''
        # Divide to batch
        n_img = len(batch_data)
        feats_result = np.zeros((n_img, feat_dim))
        tot_batch = n_img//batch_size if n_img % batch_size == 0 else n_img//batch_size + 1
        # Preprocess
        aligned = np.transpose(np.array(batch_data), (0, 3, 1, 2))
        imgs = torch.Tensor(aligned).to(self.device)
        imgs.div_(255).sub_(0.5).div_(0.5)
        # Foreach batch
        for i in range(tot_batch):
            lower  = i*batch_size
            higher = min((i+1)*batch_size, n_img)
            feats_result[lower: higher] = self.model(imgs[lower: higher]).detach().cpu().numpy()
        norm  = np.linalg.norm(feats_result, axis = 1, keepdims = True)
        feats_result = np.divide(feats_result, norm)
        # self.check_assertion(feats_result)
        return feats_result

class FaceEmbeddingBatchONNX(object):
    '''
     Face Embedding model using ONNX Backend
    '''
    def __init__(self, input_shape = (112, 112), batch_size = 8, model_path = 'weights/face_recognition_r34.onnx'):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.model = onnxruntime.InferenceSession(model_path)

    def get_features(self, batch_data, feat_dim = 512):
        '''
         Extract feature embedding from list image
         input
              - batch_data: list RGB 112x112 face image
         return
              - feats_result: normalize feature embeddings
        '''
        # Allocate
        n_img       = len(batch_data)
        n_batch     = n_img//self.batch_size if n_img % self.batch_size == 0 else  n_img//self.batch_size + 1
        feats_batch = np.zeros((self.batch_size * n_batch, feat_dim), dtype = np.float32)
        # Preprocess
        aligned             = np.transpose(np.array(batch_data).astype("float32"), (0, 3, 1, 2))
        aligned             = ((aligned / 255.0) - 0.5)/0.5
        # Infer
        for i in range(n_batch):
            lower = i*self.batch_size
            higher = min((i+1)*self.batch_size, n_img)
            ort_inputs = {self.model.get_inputs()[0].name: aligned[lower:higher]}
            feats_batch[lower:higher] = self.model.run(None, ort_inputs)[0]    
        # L2 Normalize
        norm         = np.linalg.norm(feats_batch[: n_img], axis = 1, keepdims = True)
        feats_result = np.divide(feats_batch[: n_img], norm)
        return feats_result


if __name__ == '__main__':
    model = FaceEmbeddingBatchONNX(batch_size = 4)
    img1 = cv2.imread('../test_images/face_processor/crop.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread('../test_images/face_processor/TH.png')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    inp = [img1, img2]
    tik = time.time()
    embeds = model.get_features(inp)
    print(embeds)
    print(time.time() - tik)