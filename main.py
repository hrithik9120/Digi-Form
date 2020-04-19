from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from WordSegmentation import wordSegmentation, prepareImg
import glob
from PIL import Image,ImageOps
from numpy import mean
import numpy as np
import os,os.path
import math
from autocorrect import Speller
import random as r
import tensorflow as tf
class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnCorpus = '../data/corpus.txt'


def train(model, loader):
    "train NN"
    epoch = 0 # number of training epochs since start
    bestCharErrorRate = float('inf') # best valdiation character error rate
    noImprovementSince = 0 # number of epochs no improvement of character error rate occured
    earlyStopping = 5 # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

        # validate
        charErrorRate = validate(model, loader)

        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break


def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0],'/', iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')

    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
    return charErrorRate


def clearlong_slice():
    for i in os.listdir('../src/slice'):
        os.remove('../src/slice/%s'%i)

def clearsegitr():
    for i in os.listdir('../src/out'):
        os.remove('../src/out/%s'%i)


# def clearSummary():
# 	j=0
# 	for i in os.listdir('../src'):
# 		os.remove('../src/summary%d.png'%j)
# 		j+=1

def long_slice(image_path, outdir, slice_size):

    print(image_path)
    """slice an image into parts slice_size tall"""
    img=Image.open(image_path)








    #img1=ImageOps.grayscale(img)
    width,height = img.size
    upper = 0
    left = 0
    slices = int(math.ceil(height/slice_size))



    count = 1
    for slice in range(slices):
        #if we are at the end, set the lower bound to be the bottom of the image
        if count == slices:
            lower = height
        else:
            lower = int(count * slice_size)

        bbox = (left, upper, width, lower)
        working_slice = img.crop(bbox)
        upper += slice_size
        #save the slice
        working_slice.save(os.path.join(outdir, "slice_" + str(count-1)+".png"))
        count +=1

def segitr(img):

    #"""reads images jrom data/ and outputs the word-segmentation to out/"""
    #Tk().withdraw() # we don't want a full GUI, so keep the root window jrom appearing
    #imgFiles = askopenjilename()
    # read input images jrom 'in' directory
    #imgFiles = img



    # read image, prepare it by resizing it to jixed height and converting it to grayscale
    #cv2.imshow("img",img)
    img1 = prepareImg(cv2.imread(img),100)
    pxmin = np.min(img1)
    pxmax = np.max(img1)
    imgContrast = (img1 - pxmin) / (pxmax - pxmin) * 255

    kernel = np.ones((3, 3), np.uint8)
    img1 = cv2.erode(imgContrast, kernel, iterations = 1)
     #clearsegitr()
    #cv2.imshow("img1",img1

        # execute segmentation with given parameters
        # -kernelSize: size oj jilter kernel (odd integer)
        # -sigma: standard deviation oj Gaussian junction used jor filter kernel
        # -theta: approximated width/height ratio oj words, filter junction is distorted by this factor
        # - minArea: ignore word candidates smaller than specified area
    res = wordSegmentation(img1, kernelSize=75, sigma=27 , theta=16)
    #multiples of 5
    # write output to 'out/inputFileName' directory

        # iterate over all segmented words
    print('\n\nSegmented into %d words\n\n'%len(res))
    for (j, w) in enumerate(res):
        (wordBox, wordImg) = w
        (x, y, w, h) = wordBox
        cv2.imwrite('../src/out/%d.png'%j, wordImg) # save word
        # cv2.imwrite('../src/%d.png'%(j+r.randint(0,100)), wordImg)
        cv2.rectangle(img1,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
        # output summary image with bounding boxes around words
        cv2.imwrite('../src/summary%d.png'%(j+r.randint(0,100)), img1)

def infer(model,ipath):
    rec = []
    "recognize text in image provided by file path"
    spell= Speller(lang='en')
   # misspelled = spell.unknown(['Name', 'Branch', 'Roll', 'No'])
    #Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    #long_slice(ipath,'../src/slice/',200)
    long_slice(ipath,'../src/slice/',200)

    #iterate over slices
    slice = [cv2.imread(file) for file in glob.glob("../src/slice/slice_*.png")]
    #print(len(slice))
    for j in range(0,len(slice)):

        segitr('../src/slice/slice_%d.png'%j)
    #segitr(askopenfilename())

        images = [file for file in glob.glob("../src/out/*.png")]

        for i in range(0,len(images)):
            img = preprocess((cv2.imread('../src/out/%d.png'%i,cv2.IMREAD_GRAYSCALE)), Model.imgSize)
            #cv2.imshow(img)
            #img = preprocess((cv2.imread(images[i])), Model.imgSize)
            batch = Batch(None, [img])
            (recognized, probability) = model.inferBatch(batch, True)
            print('Recognized:' + recognized[0]+'\t\t'+'Probability:',probability[0])
            rec.append(recognized[0])
        images.clear()
        clearsegitr()
    print(rec)
    return rec

def inferweb(ipath):
    decoderType = DecoderType.BestPath
    model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=False)
    rec = infer(model,ipath)
    # rec = map(lambda x:x.lower(), rec)
    return rec



def main():
    "main function"
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')
    parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
    parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding', action='store_true')
    parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')

    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    # train or validate on IAM dataset
    if args.train or args.validate:
        # load training data, create TF model
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

        # save characters of model for inference mode
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

        # save words contained in dataset into file
        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

        # execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=True)
            validate(model, loader)

    # infer text on test image
    else:
        print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)

        clearlong_slice()
        clearsegitr()
        #clearSummary()
        Tk().withdraw()
        ipath = askopenfilename()
        rec = infer(model)


if __name__ == '__main__':
    main()
