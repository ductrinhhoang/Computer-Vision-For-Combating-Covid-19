import os
import cv2
import glob
import numpy as np
import argparse
import scipy.io as sio
from shutil import rmtree
from sklearn.neighbors import KNeighborsClassifier
from face_model import FaceModelONNX
from utils.visualize import visualize_detect, visualize_recognition


def load_matfile(matfile):
    '''
     Load matfile
    '''
    data = sio.loadmat(matfile)
    return data

def rebuild_knn_dict(lst_matfiles):
    '''
     Rebuild KNN & Face ID Dictionary from list .mat files
    '''
    count_id = 0  
    lst_embeds = []  # KNN Features
    lst_labels     = []  # KNN id
    dct_weights = dict()
    dct_id      = dict()
    print('- Rebuild KNN')
    for matfile in lst_matfiles:
        print('-- Load:', matfile)
        data = load_matfile(matfile)
        id   = np.squeeze(data['id'])
        features = data['features']
        weights  = np.squeeze(data['weights']) # Don't know why scipy.io.savemat expands 1 dims

        for ix in range(len(features)):
            
            lst_embeds.append(features[ix])
            lst_labels.append(count_id)

            dct_weights[count_id] = weights[ix]
            dct_id[count_id]      = id 
            count_id += 1
        print('-- Found total {} faces for id: {}'.format(len(features), id))
    # Repack
    lst_embeds = np.array(lst_embeds)
    lst_labels     = np.array(lst_labels, dtype = np.uint16)
    assert len(lst_embeds) == len(lst_labels), \
            "Number of faces did not equal number of id: {} vs {}".format(len(lst_embeds), len(lst_labels))
    print('- Found total {} faces in database folder'.format(len(lst_embeds)))
    return lst_embeds, lst_labels, dct_id, dct_weights

def query_knn(feature, knn_model, dct_id, dct_weights, n_neighbors = 3, threshold = 1.05):
    '''
     Query KNN with penalty weights
    '''
    # 1. Query KNN
    neigh_dist, neigh_ind = knn_model.kneighbors(X=np.array([feature]), n_neighbors=n_neighbors, return_distance=True)
    # 2. Apply penalty weights
    neigh_dist = np.squeeze(neigh_dist)
    neigh_ind  = np.squeeze(neigh_ind)
    penalty_dist = []
    for ix in range(n_neighbors):
        penalty_dist.append(neigh_dist[ix] * dct_weights[neigh_ind[ix]])
    penalty_dist = np.array(penalty_dist)
    # 3. Get post-process result
    best_distance = np.amin(penalty_dist)
    best_face_id = neigh_ind[np.argmin(penalty_dist)]
    best_person_id = dct_id[best_face_id]
    if best_distance >= threshold:
        best_person_id = None
    return best_face_id, best_distance, best_person_id



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract database image and dump to disk')
    parser.add_argument('--test_folder', type=str,
                        default='../test_images/recognition',
                        help='path of test folder')
    parser.add_argument('--output_folder', type=str,
                        default='../result',
                        help='path of result folder')
    parser.add_argument('--mat_folder', type=str,
                        default='../database/mats',
                        help='path of mat folder')
    parser.add_argument('--threshold', type=float,
                        default=1.05,
                        help='distance threshold')
    parser.add_argument('--build_top_k', type=int,
                        default=3,
                        help='KNN top K for building')

    args = parser.parse_args()
    # Init folder
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok = True)

    # Rebuild KNN model
    lst_matfiles = glob.glob(args.mat_folder + '/*.mat')
    if len(lst_matfiles) == 0:
        raise FileNotFoundError("Not found any id, check 'extract_database.py' for extract database from image")
    lst_embeds, lst_labels, dct_id, dct_weights = rebuild_knn_dict(lst_matfiles)
    knn_model = KNeighborsClassifier(n_neighbors=args.build_top_k)
    knn_model.fit(lst_embeds, lst_labels)

    # Face model
    face_model = FaceModelONNX()

    # Extract
    lst_test_images  = glob.glob(args.test_folder + '/*.jpg')
    lst_test_images += glob.glob(args.test_folder + '/*.png')
    lst_test_images += glob.glob(args.test_folder + '/*.jpeg')
    for test_image_path in lst_test_images:
        print('- Process image: ', test_image_path)
        image     = cv2.imread(test_image_path)
        vis_image = image.copy() 
        # Detect face
        bboxes, points = face_model.get_inputs(image)
        # Align face
        img_faces = face_model.get_face_align(image, bboxes, points)
        # Extract embedding
        embeds    = face_model.get_features(img_faces)
        vis_image = visualize_detect(vis_image,
                                    bboxes,
                                    color = (0, 0, 255),
                                    thinkness = 3)
        print('-- Found {} faces'.format(len(embeds)))
        # KNN search & visualize
        for ix, feature in enumerate(embeds):
            best_face_id, best_distance, best_person_id = query_knn(feature,
                                                                knn_model,
                                                                dct_id,
                                                                dct_weights,
                                                                n_neighbors = args.build_top_k,
                                                                threshold = args.threshold)
            if best_person_id is not None:
                text  = '{} ({})'.format(best_person_id, int(best_distance*1000)/1000)
                vis_image = visualize_recognition(vis_image,
                                                text,
                                                bboxes[ix],
                                                color = (0, 0, 255))
        # Write image
        output_path = os.path.join(args.output_folder, os.path.basename(test_image_path))
        cv2.imwrite(output_path, vis_image)
        print('-- Writed to: ', output_path)

        