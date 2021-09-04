import os
import cv2
import glob
import numpy as np
import argparse
import scipy.io as sio
from shutil import rmtree
from face_model import FaceModelONNX

def create_augmentation_with_weights(lst_face_img, weights = [1.05, 1.075, 1.1, 1.2]):
    '''
     Create mask 1-5 augmentation image with penalty weights
    '''
    lst_face_aug = []
    lst_weights  = []
    temp = int(112/8)
    for ix, face_img in enumerate(lst_face_img):
        height, width, _ = face_img.shape
        assert width == 112 and height == 112, "Invalid shape: {}x{}".format(width, height)
        # Mask 1/8
        face_temp = face_img.copy()
        face_temp[7*temp: 8*temp, : , :] = 0
        lst_face_aug.append(face_temp)
        lst_weights.append(weights[0])
        # Mask 2/8
        face_temp = face_img.copy()
        face_temp[6*temp: 8*temp, : , :] = 0
        lst_face_aug.append(face_temp)
        lst_weights.append(weights[1])
        # Mask 3/8
        face_temp = face_img.copy()
        face_temp[5*temp: 8*temp, : , :] = 0
        lst_face_aug.append(face_temp)
        lst_weights.append(weights[2])
        # Mask 4/8
        face_temp = face_img.copy()
        face_temp[4*temp: 8*temp, : , :] = 0
        lst_face_aug.append(face_temp)
        lst_weights.append(weights[3])
    return lst_face_aug, lst_weights

def extract_single_id(lst_path_img, id, save_folder, use_aug = True, assert_one = True):
    '''
     Extract embedding for single ID
    '''
    print('- Processing ID: ', id)
    count = 0
    lst_img_faces = []
    lst_embeds    = []
    lst_weights   = []
    # Extract single image
    for path_img in lst_path_img:
        print('-- Processing:', path_img)
        img = cv2.imread(path_img)
        bboxes, points = face_model.get_inputs(img)
        img_faces = face_model.get_face_align(img, bboxes, points)
        assert len(img_faces) == 1, "Database image require single face per image, found {} faces in {}".format(path_img, len(img_faces))
        if use_aug:
            img_faces, weights = create_augmentation_with_weights(img_faces)
        else:
            weights = [1.0] * len(img_faces)
        embeds    = face_model.get_features(img_faces)
        lst_img_faces.append(img_faces)
        lst_embeds.append(embeds)
        lst_weights.append(np.array(weights))
        count += len(img_faces)
    # Concatenate all
    lst_img_faces = np.concatenate(lst_img_faces)
    lst_embeds    = np.concatenate(lst_embeds)
    lst_weights   = np.concatenate(lst_weights)
    # Dump to disk
    assert len(lst_img_faces) == len(lst_embeds) and len(lst_img_faces) == len(lst_weights), \
        "Something went wrong"
    save_path = os.path.join(save_folder, str(id) + '.mat')
    save_dict = {'id': str(id), 'features': lst_embeds, 'faces': lst_img_faces, 'weights': lst_weights}
    sio.savemat(save_path, save_dict)
    print('-- Saved {} face images to {}'.format(count, save_path))
    return lst_img_faces

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract database image and dump to disk')
    parser.add_argument('--data_folder', type=str,
                        default='../database/images',
                        help='path of database folder')
    parser.add_argument('--output_folder', type=str,
                        default='../database/mats',
                        help='path to output folder')
    parser.add_argument('--crop_folder', type=str,
                        default='../database/cropped',
                        help='path to face cropped folder')
    parser.add_argument('--save_crop',
                        action = 'store_true',
                        help='Save cropped faces to disk')
    parser.add_argument('--use_aug',
                        action = 'store_true',
                        help='Use augmentation')
    args = parser.parse_args()
    # Prepare
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    if args.save_crop:
        if not os.path.exists(args.crop_folder):
            os.makedirs(args.crop_folder)
    # Face model
    face_model = FaceModelONNX()
    # Extract
    for folder in glob.glob(args.data_folder + '/*'):
        # Assume id is name of folder
        id = os.path.basename(folder)
        lst_path_img  = glob.glob(folder + '/*.jpg')
        lst_path_img += glob.glob(folder + '/*.png')
        lst_img_faces = extract_single_id(lst_path_img, id, args.output_folder, args.use_aug)
        if args.save_crop:
            folder_for_id = os.path.join(args.crop_folder, id)
            if os.path.exists(folder_for_id):
                rmtree(folder_for_id)
            os.makedirs(folder_for_id)
            for ix, rgb_face in enumerate(lst_img_faces):
                cv2.imwrite(os.path.join(folder_for_id, 'face_{}.jpg'.format(ix)), cv2.cvtColor(rgb_face, cv2.COLOR_RGB2BGR))

