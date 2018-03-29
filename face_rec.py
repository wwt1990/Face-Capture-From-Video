import os
import sys 

import numpy as np
import pandas as pd

import cv2
import openface
import openface.helper
from openface.data import iterImgs

from operator import itemgetter

import pickle
import glob

import multiprocessing
import argparse


from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


import random
import shutil




file_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(file_dir, 'models')
dlib_model_dir = os.path.join(model_dir, 'dlib')
openface_model_dir = os.path.join(model_dir, 'openface')
batch_rep_dir = os.path.join(file_dir, 'batch_represent')

verobse = False

def alignMain(args):
    openface.helper.mkdirP(args.output_aligned_path)

    imgs = list(iterImgs(args.training_file_path))

    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)

    landmarkMap = {
        'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
        'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    }
    if args.landmarks not in landmarkMap:
        raise Exception("Landmarks unrecognized: {}".format(args.landmarks))

    landmarkIndices = landmarkMap[args.landmarks]

    align = openface.AlignDlib(args.dlibFacePredictor)

    nFallbacks = 0
    for imgObject in imgs:
        if verbose:
            print("=== {} ===".format(imgObject.path))
        outDir = os.path.join(args.output_aligned_path, imgObject.cls)
        openface.helper.mkdirP(outDir)
        outputPrefix = os.path.join(outDir, imgObject.name)
        imgName = outputPrefix + ".png"

        if os.path.isfile(imgName):
            if args.verbose:
                print("  + Already found, skipping.")
        else:
            rgb = imgObject.getRGB()
            if rgb is None:
                if args.verbose:
                    print("  + Unable to load.")
                outRgb = None
            else:
                outRgb = align.align(args.size, rgb,
                                     landmarkIndices=landmarkIndices)
                                     # skipMulti=args.skipMulti)
                if outRgb is None and args.verbose:
                    print("  + Unable to align.")

            if args.fallbackLfw and outRgb is None:
                nFallbacks += 1
                deepFunneled = "{}/{}.jpg".format(os.path.join(args.fallbackLfw,
                                                               imgObject.cls),
                                                  imgObject.name)
                shutil.copy(deepFunneled, "{}/{}.jpg".format(os.path.join(args.output_aligned_path,
                                                                          imgObject.cls),
                                                             imgObject.name))

            if outRgb is not None:
                if args.verbose:
                    print("  + Writing aligned file to disk.")
                outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(imgName, outBgr)

    if args.fallbackLfw:
        print('nFallbacks:', nFallbacks)

def preprocess(training_file_path, num_jobs):
    class Args():
        def __init__(self, training_file_path, output_aligned_path, verbose):
            self.training_file_path = training_file_path
            self.mode = 'align'
            self.dlibFacePredictor = os.path.join(
                dlib_model_dir, "shape_predictor_68_face_landmarks.dat")
            self.landmarks = 'outerEyesAndNose'
            self.size = 96
            self.output_aligned_path = output_aligned_path
            self.skipMulti = True
            self.verbose = verbose
            self.fallbackLfw = False

    output_aligned_path = '{0}_aligned'.format(training_file_path)
    if not os.path.exists(output_aligned_path):
        os.makedirs(output_aligned_path)
    argsForAlign = Args(training_file_path, output_aligned_path, verbose)
    jobs = []
    for i in range(num_jobs):
        p = multiprocessing.Process(
            target=alignMain, args=(
                argsForAlign,))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()

    output_feature_path = '{0}_features'.format(training_file_path)
    if not os.path.exists(output_feature_path):
        os.makedirs(output_feature_path)

    # If your dataset has changed, delete the cache file. 
    os.system(
        os.path.join(batch_rep_dir, 'main.lua') + ' -outDir ' +
        output_feature_path + ' -data ' + output_aligned_path)

    train(output_feature_path)


def train(output_feature_path):
    print("Loading features.")
    fname = os.path.join(output_feature_path, 'labels.csv')
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the _directory.
    fname = os.path.join(output_feature_path, 'reps.csv')
    embeddings = pd.read_csv(fname, header=None).as_matrix()   # (4325, 128)
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    clf = SVC(C=1, kernel='linear', probability=True)

    clf.fit(embeddings, labelsNum)

    fName = os.path.join(output_feature_path, 'classifier.pkl')
    print("Saving classifier to {}".format(fName))
    
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)

    print("Done with saving.")


def getFace(imgPath, img_dim=96):
    bgrImg = cv2.imread(imgPath)  
    if bgrImg is None:
        if verbose:
            print ('Unable to load image: {}'.format(imgPath))
        return
    
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgbImg)   
    
    if bb is None:
        if verbose:
            print ('Unable to find a face: {}'.format(imgPath))
        return None, None, None, None
    
    landmarks = align.findLandmarks(rgbImg, bb)
    if landmarks is None:
        if verbose:
            print ('Unable to find a face: {}'.format(imgPath))
        return None, None, None, None
    alignedFace = align.align(img_dim, rgbImg, bb=1, landmarks=landmarks, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)   
    if alignedFace is None:
        if verbose:
            print ('Unable to align image: {}'.format(imgPath))
        return None, None, None, None
    return alignedFace, rgbImg, landmarks, bgrImg   # (96, 96, 3)


def getRep(alignedFace):
    rep = net.forward(alignedFace)
    return rep

def infer(list_files, output_file_path, target_characters, lock):
    for img_path in list_files:
        alignedFace, rgbImg, landmarks, bgrImg = getFace(img_path)
        if alignedFace is None:
            continue
        
        lock.acquire()
        rep = getRep(alignedFace).reshape(1, -1)
        lock.release()
        
        predictions = svm.predict_proba(rep)[0]
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        print person
        confidence = predictions[maxI]
        
        if person in target_characters:
            rgbImg = align.align(128, rgbImg, bb=1, landmarks=landmarks, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            bgrImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2BGR)
            img_name = img_path.split('/')[-1].split('.')[0]
            cv2.imwrite('{0}/output_{1}/{2:.3f}_{3}.jpg'.format(output_file_path, person, confidence, img_name), bgrImg)

if __name__ == '__main__':
    print('start')

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--num_jobs', type=int, help='Number of parallel tasks.', default=1)
    parser.add_argument('--input_file_path', type=str, help='Path to full captures of an episode.', required=True)
    parser.add_argument('--training_file_path', type=str, help='Path to training images.', required=True)
    parser.add_argument('--output_file_path', type=str, help='Path to outputs of targeted faces.', required=True)
    parser.add_argument('--target_characters', type=str, help='', default='00baiqian')

    args = parser.parse_args()

    verbose = args.verbose
    num_jobs = args.num_jobs
    input_file_path = args.input_file_path
    training_file_path = args.training_file_path
    output_file_path = args.output_file_path
    target_characters = args.target_characters.split(',')

 
    preprocess(training_file_path, num_jobs)
    print('Done with preprocessing')

    for target_character in target_characters:
        output_dir_path = os.path.join(output_file_path, 'output_{0}'.format(target_character))
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
    
    dlibFacePredictor = os.path.join(dlib_model_dir, 'shape_predictor_68_face_landmarks.dat')

    align = openface.AlignDlib(dlibFacePredictor)
    net = openface.TorchNeuralNet(os.path.join(openface_model_dir, 'nn4.small2.v1.t7'))

    output_feature_path = '{0}_features'.format(training_file_path)
    with open(os.path.join(output_feature_path, 'classifier.pkl'), 'r') as f:
        (le, svm) = pickle.load(f)

    distributed_files = []
    for i in range(num_jobs):
        distributed_files.append([])

    for i, img_path in enumerate(glob.iglob('{0}/*.jpg'.format(input_file_path))):
        bucket = i % num_jobs
        distributed_files[bucket].append(img_path)

    jobs = []
    lock = multiprocessing.Lock()
    for i in range(num_jobs):
        p = multiprocessing.Process(
            target=infer,
            args=(distributed_files[i], output_file_path, target_characters, lock))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()





