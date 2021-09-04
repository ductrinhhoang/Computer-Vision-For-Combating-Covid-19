import os
import cv2
import torch
import argparse
import time
import numpy as np
from utils.image import ImageData
from detectors.retinaface import *
from exec_backends.trt_loader import TrtModel

class RetinafaceBatchTRT(object):
    def __init__(self, model_path, input_shape = (640, 640), batch_size = 4, post_process_type = 'SINGLE'):
        '''
            TensorRT-Retinaface with batch-inference
        '''
        print('[INFO] Create Retinaface TensorRT-runtime')
        self.model = TrtModel(model_path)
        self.input_shape = input_shape
        self.rac = 'net3'
        self.masks = False
        self.batch_size = batch_size
        self.post_process_type = post_process_type
        for size in self.input_shape:
            if size % 32 != 0:
                raise ValueError("Current support only size which is multiple of 32 for compabilities")

    def prepare(self, nms: float = 0.4, **kwargs):
        self.nms_threshold = nms
        self.landmark_std = 1.0
        _ratio = (1.,)
        fmc = 3
        if self.rac == 'net3':
            _ratio = (1.,)
        elif self.rac == 'net3l':
            _ratio = (1.,)
            self.landmark_std = 0.2
        else:
            assert False, 'rac setting error %s' % self.rac

        if fmc == 3:
            self._feat_stride_fpn = [32, 16, 8]
            self.anchor_cfg = {
                '32': {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '16': {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '8': {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
            }
        self.use_landmarks = True
        self.fpn_keys = []

        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s' % s)

        self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(cfg=self.anchor_cfg)))
        for k in self._anchors_fpn:
            v = self._anchors_fpn[k].astype(np.float32)
            self._anchors_fpn[k] = v
        
        self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
        # Create anchor
        print('Rebuild anchor')
        self.anchor_plane_cache = {}
        for _idx, s in enumerate(self._feat_stride_fpn):
            stride = int(s)
            width = int(self.input_shape[0]/stride)
            height = int(self.input_shape[1]/stride)
            K = width * height
            A = self._num_anchors['stride%s' % s]
            key = (height, width, stride)
            anchors_fpn = self._anchors_fpn['stride%s' % s]
            anchors = anchors_plane(height, width, stride, anchors_fpn)
            self.anchor_plane_cache[key] = np.tile(anchors.reshape((K * A, 4)), (self.batch_size, 1, 1))
        self.warm_up()

    def warm_up(self):
        '''
            Warm up NMS jit
        '''
        print('Warming up NMS jit...')
        tik = time.time()
        image = cv2.imread('test_images/lumia.jpg', cv2.IMREAD_COLOR)
        _ = self.detect([image], threshold=0.1)
        tok = time.time()
        print('Warming up complete, time cost = {}'.format(tok - tik))

    def detect(self, list_bgr_img, threshold: float = 0.6, log = False):
        # Preprocess
        list_image_data = self.preprocess_batch(list_bgr_img)
        batch_data = np.array([img.transformed_image for img in list_image_data], dtype = np.float32)
        # Allocate
        n_img       = len(batch_data)
        if n_img % self.batch_size == 0:
            to_pad = 0
        else:
            to_pad = ((n_img//self.batch_size)+1)*self.batch_size - n_img
        # print('To pad: {}'.format(to_pad))
        n_batch = (n_img + to_pad)//self.batch_size
        paded_batch = np.zeros((self.batch_size * n_batch, 3, self.input_shape[1], self.input_shape[0]), dtype = np.float32)
        # Preprocess
        aligned             = np.transpose(batch_data, (0, 3, 1, 2))
        paded_batch[:n_img] = aligned
        # print('will infer for {} batches'.format(n_batch))
        list_det = []
        list_landmarks = []
        for i in range(n_batch):
            lower = i*self.batch_size
            higher = (i+1)*self.batch_size
            rs = self.detect_single_batch(paded_batch[lower:higher], threshold=threshold)
            list_det.extend(rs[0])
            list_landmarks.extend(rs[1])
        # Divive by scale factor
        list_det = list_det[:n_img]
        # assert len(list_det) == len(list_image_data), \
        #     "Number of output not equal number of input: {} vs {}".format(len(list_det), len(list_image_data))
        for i in range(len(list_image_data)):
            list_det[i]       =  list_det[i][:, :4]/list_image_data[i].scale_factor
            list_landmarks[i] /= list_image_data[i].scale_factor
        return list_det, list_landmarks

    def detect_single_batch(self, batch_img, threshold: float = 0.6):
        batch_size = len(batch_img)
        assert batch_size == self.batch_size, "Model define with batch_size = {}, your input: {}".format(self.batch_size, batch_size) 
        tik = time.time()
        net_out = self.model.run(batch_img)
        tok = time.time()
        print('Infer time: {}'.format(tok-tik))
        # Sort cause output model while convert to TensorRT is shuffled
        indices = [4, 0, 1, 5, 2, 3, 6, 7, 8]
        sorted_net_out = [net_out[i] for i in indices]
        if self.post_process_type == 'BATCH':
            result = self.postprocess_batch(sorted_net_out, threshold = threshold, batch_size = batch_size)
            return result
        else:
            list_det = []
            list_landmarks = []
            for i in range(batch_size):
                single_net_out = [np.expand_dims(layer_out[i], 0) for layer_out in sorted_net_out]
                det, landmarks = self.postprocess(single_net_out, threshold = threshold)
                list_det.append(det)
                list_landmarks.append(landmarks)
            return list_det, list_landmarks

    def preprocess_batch(self, list_bgr_img):
        '''
            Preprocess image for batch-inference
        '''
        list_image_data = []
        for ix, bgr_image in enumerate(list_bgr_img):
            # print(bgr_image.shape)
            image = ImageData(bgr_image, self.input_shape)
            image.resize_image(mode='pad')     
            list_image_data.append(image)
        return list_image_data

    def postprocess_batch(self, net_out, threshold, batch_size):
        '''
            Post process for batch-inference
        '''
        proposals_list_batch = {i : [] for i in range(batch_size)}
        scores_list_batch = {i : [] for i in range(batch_size)}
        landmarks_list_batch = {i : [] for i in range(batch_size)}
        # t0 = time.time()
        # Foreach FPN layer
        for _idx, s in enumerate(self._feat_stride_fpn):
            _key = 'stride%s' % s
            stride = int(s)
            if self.use_landmarks:
                idx = _idx * 3
            else:
                idx = _idx * 2
            if self.masks:
                idx = _idx * 4

            A = self._num_anchors['stride%s' % s]
            
            scores_batch = net_out[idx]
            scores_batch = scores_batch[:, A:, :, :]
            idx += 1
            bbox_deltas_batch = net_out[idx]
            height, width = bbox_deltas_batch.shape[2], bbox_deltas_batch.shape[3]

            # K = height * width
            key = (height, width, stride)
            anchors_batch = self.anchor_plane_cache[key]

            scores_batch = clip_pad(scores_batch, (height, width))
            scores_batch = scores_batch.transpose((0, 2, 3, 1)).reshape((batch_size, -1, 1))

            bbox_deltas_batch = clip_pad(bbox_deltas_batch, (height, width))
            bbox_deltas_batch = bbox_deltas_batch.transpose((0, 2, 3, 1))
            bbox_pred_len = bbox_deltas_batch.shape[3] // A
            bbox_deltas_batch = bbox_deltas_batch.reshape((batch_size, -1, bbox_pred_len))
            proposals_batch = bbox_pred_batch(anchors_batch, bbox_deltas_batch)
            
            
            # Get proposal
            scores_batch = scores_batch.reshape((batch_size, -1))
            order_batch = np.argwhere(scores_batch >= threshold)

            # Get landmark
            if self.use_landmarks:
                idx += 1
                landmark_deltas_batch = net_out[idx]
                landmark_deltas_batch = clip_pad(landmark_deltas_batch, (height, width))
                landmark_pred_len = landmark_deltas_batch.shape[1] // A
                landmark_deltas_batch = landmark_deltas_batch.transpose((0, 2, 3, 1)).reshape((batch_size, -1, 5, landmark_pred_len // 5))
                landmark_deltas_batch *= self.landmark_std
                landmarks = landmark_pred_batch(anchors_batch, landmark_deltas_batch)

            # Foreach image
            for ib in range(batch_size): 
                order = [id[1] for id in order_batch if id[0] == ib]
                proposals_list_batch[ib].append(proposals_batch[ib, order])
                scores_list_batch[ib].append(scores_batch[ib, order].reshape((-1, 1)))
                if self.use_landmarks:
                    landmarks_list_batch[ib].append(landmarks[ib, order])
                
        # Foreach image
        list_det = []
        list_landmarks = []
        for ib in range(batch_size):
            proposals_list = proposals_list_batch[ib]
            scores_list = scores_list_batch[ib]
            landmarks_list = landmarks_list_batch[ib]

            proposals = np.vstack(proposals_list)
            landmarks = None
            if proposals.shape[0] == 0:
                if self.use_landmarks:
                    landmarks = np.zeros((0, 5, 2))
                list_det.append(np.zeros((0, 5)))
                list_landmarks.append(landmarks)
                continue

            scores = np.vstack(scores_list)
            scores_ravel = scores.ravel()
            order = scores_ravel.argsort()[::-1]
            proposals = proposals[order, :]
            scores = scores[order]

            if self.use_landmarks:
                landmarks = np.vstack(landmarks_list)
                landmarks = landmarks[order].astype(np.float32, copy=False)

            pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
            keep = nms(pre_det, thresh=self.nms_threshold)
            det = np.hstack((pre_det, proposals[:, 4:]))
            det = det[keep, :]
            if self.use_landmarks:
                landmarks = landmarks[keep]
            # t1 = time.time()
            list_det.append(det)
            list_landmarks.append(landmarks)
        return list_det, list_landmarks
    
    def postprocess(self, net_out, threshold):
        proposals_list = []
        scores_list = []
        mask_scores_list = []
        landmarks_list = []
        t0 = time.time()
        for _idx, s in enumerate(self._feat_stride_fpn):
            _key = 'stride%s' % s
            stride = int(s)
            if self.use_landmarks:
                idx = _idx * 3
            else:
                idx = _idx * 2
            if self.masks:
                idx = _idx * 4

            A = self._num_anchors['stride%s' % s]

            scores = net_out[idx]
            # print(scores.shape, idx)
            scores = scores[:, A:, :, :]
            idx += 1
            bbox_deltas = net_out[idx]
            height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

            # K = height * width
            key = (height, width, stride)
            # if key in self.anchor_plane_cache:
            #     anchors = self.anchor_plane_cache[key]
            # else:
            #     anchors_fpn = self._anchors_fpn['stride%s' % s]
            #     anchors = anchors_plane(height, width, stride, anchors_fpn)
            #     anchors = anchors.reshape((K * A, 4))
            #     if len(self.anchor_plane_cache) < 100:
            #         self.anchor_plane_cache[key] = anchors
            # print(height, width, stride, anchors_fpn, scores.shape)
            anchors = self.anchor_plane_cache[key][0]
            scores = clip_pad(scores, (height, width))
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

            bbox_deltas = clip_pad(bbox_deltas, (height, width))
            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
            bbox_pred_len = bbox_deltas.shape[3] // A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))

            proposals = bbox_pred(anchors, bbox_deltas)

            scores_ravel = scores.ravel()
            order = np.where(scores_ravel >= threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]

            proposals_list.append(proposals)
            scores_list.append(scores)

            if self.use_landmarks:
                idx += 1
                landmark_deltas = net_out[idx]
                landmark_deltas = clip_pad(landmark_deltas, (height, width))
                landmark_pred_len = landmark_deltas.shape[1] // A
                landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len // 5))
                landmark_deltas *= self.landmark_std
                landmarks = landmark_pred(anchors, landmark_deltas)
                landmarks = landmarks[order, :]
                landmarks_list.append(landmarks)

        proposals = np.vstack(proposals_list)
        landmarks = None
        if proposals.shape[0] == 0:
            if self.use_landmarks:
                landmarks = np.zeros((0, 5, 2))
            return np.zeros((0, 5)), landmarks

        scores = np.vstack(scores_list)

        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        proposals = proposals[order, :]
        scores = scores[order]

        if self.use_landmarks:
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)
        if self.masks:
            mask_scores = np.vstack(mask_scores_list)
            mask_scores = mask_scores[order]
            pre_det = np.hstack((proposals[:, 0:4], scores, mask_scores)).astype(np.float32, copy=False)
        else:
            pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
        keep = nms(pre_det, thresh=self.nms_threshold)
        det = np.hstack((pre_det, proposals[:, 4:]))
        det = det[keep, :]
        if self.use_landmarks:
            landmarks = landmarks[keep]
        t1 = time.time()
        return det, landmarks