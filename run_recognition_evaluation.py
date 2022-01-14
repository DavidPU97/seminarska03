import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist 
from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition import Evaluation

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config_recognition.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def clean_file_name(self, fname):
        return fname.split('/')[1].split(' ')[0]

    def get_annotations(self, annot_f):
        d = {}
        with open(annot_f) as f:
            lines = f.readlines()
            for line in lines:
                (key, val) = line.split(',')
                # keynum = int(self.clean_file_name(key))
                d[key] = int(val)
        return d

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        iou_arr = []
        preprocess = Preprocess()
        eval = Evaluation()

        cla_d = self.get_annotations(self.annotations_path)
        
        # Change the following extractors, modify and add your own

        import feature_extractors.your_super_extractor.my_super_extractor as super_ext
        my_lbp = super_ext.LBP()

        # Pixel-wise comparison:
        import feature_extractors.pix2pix.extractor as p2p_ext
        pix2pix = p2p_ext.Pix2Pix()
        
        lbp_features_arr = []
        lbp_features_arr_histogram = []
        lbp_features_arr_hog = []
        hog_features_arr_lbp = []
        plain_features_arr = []
        hog_features_arr = []
        sobel_features_arr = []
        canny_features_arr = []
        daisy_features_arr = []
        y = []

        for im_name in im_list:
            
            # Read an image
            img = cv2.imread(im_name)

            # y.append(cla_d['/'.join(im_name.split('/')[-2:])])
            y.append(cla_d[im_name.split('\\')[0].split('/')[-1]+'/'+im_name.split('\\')[1]])

            # Apply some preprocessing here
            
            # Run the feature extractors

            plain_features = pix2pix.extract(img)
            plain_features_arr.append(plain_features)

            lbp_features = my_lbp.lbp(img)
            lbp_features_arr.append(lbp_features)

            lbp_features_hist = my_lbp.lbp_hist(img)
            lbp_features_arr_histogram.append(lbp_features_hist)

            lbp_features_hog = my_lbp.lbp_hog(img)
            lbp_features_arr_hog.append(lbp_features_hog)

            hog_features = my_lbp.hog(img)
            hog_features_arr.append(hog_features)

            hog_features_lbp = my_lbp.hog_lbp(img)
            hog_features_arr_lbp.append(hog_features_lbp)

            sobel_features = my_lbp.sobel(img)
            sobel_features_arr.append(sobel_features)

            canny_features = my_lbp.canny(img)
            canny_features_arr.append(canny_features)

            daisy_features = my_lbp.daisy(img)
            daisy_features_arr.append(daisy_features)


        Y_plain = cdist(plain_features_arr, plain_features_arr, 'jensenshannon')
        Y_lbp = cdist(lbp_features_arr, lbp_features_arr, 'jensenshannon')
        Y_lbp_hist = cdist(lbp_features_arr_histogram, lbp_features_arr_histogram, 'jensenshannon')
        Y_lbp_hog = cdist(lbp_features_arr_hog, lbp_features_arr_hog, 'jensenshannon')
        Y_sobel = cdist(sobel_features_arr, sobel_features_arr, 'jensenshannon')
        Y_canny = cdist(canny_features_arr, canny_features_arr, 'jensenshannon')
        Y_daisy = cdist(daisy_features_arr, daisy_features_arr, 'jensenshannon')
        Y_hog = cdist(hog_features_arr, hog_features_arr, 'jensenshannon')
        Y_hog_lbp = cdist(hog_features_arr_lbp, hog_features_arr_lbp, 'jensenshannon')

        r1 = eval.compute_rank1(Y_plain, y)
        print('Pix2Pix Rank-1[%]', r1)
        r5 = eval.compute_rank5(Y_plain, y)
        print('Pix2Pix Rank-5[%]', r5)

        r1_lbp = eval.compute_rank1(Y_lbp, y)
        print('LBP Rank-1[%]', r1_lbp)
        r5_lbp = eval.compute_rank5(Y_lbp, y)
        print('LBP Rank-5[%]', r5_lbp)

        r1_lbp_hist = eval.compute_rank1(Y_lbp_hist, y)
        print('LBP Histogram Rank-1[%]', r1_lbp_hist)
        r5_lbp_hist = eval.compute_rank5(Y_lbp, y)
        print('LBP Histogram Rank-5[%]', r5_lbp_hist)

        r1_lbp_hog = eval.compute_rank1(Y_lbp_hog, y)
        print('LBP HOG Rank-1[%]', r1_lbp_hog)
        r5_lbp_hog = eval.compute_rank5(Y_lbp_hog, y)
        print('LBP HOG Rank-5[%]', r5_lbp_hog)

        r1_sobel = eval.compute_rank1(Y_sobel, y)
        print('Sobel Rank-1[%]', r1_sobel)
        r5_sobel = eval.compute_rank5(Y_sobel, y)
        print('Sobel Rank-5[%]', r5_sobel)

        r1_canny = eval.compute_rank1(Y_canny, y)
        print('Canny Rank-1[%]', r1_canny)
        r5_canny = eval.compute_rank5(Y_canny, y)
        print('Canny Rank-5[%]', r5_canny)

        r1_daisy = eval.compute_rank1(Y_daisy, y)
        print('Daisy Rank-1[%]', r1_daisy)
        r5_daisy = eval.compute_rank5(Y_daisy, y)
        print('Daisy Rank-5[%]', r5_daisy)

        r1_hog = eval.compute_rank1(Y_hog, y)
        print('HOG Rank-1[%]', r1_hog)
        r5_hog = eval.compute_rank5(Y_hog, y)
        print('HOG Rank-5[%]', r5_hog)

        r1_hog_lbp = eval.compute_rank1(Y_hog_lbp, y)
        print('HOG LBP Rank-1[%]', r1_hog_lbp)
        r5_hog_lbp = eval.compute_rank5(Y_hog_lbp, y)
        print('HOG LBP Rank-5[%]', r5_hog_lbp)

if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()