import os
import lmdb
import cv2
import numpy as np

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            return False
    except:
        print("Image is invalid!")
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode('ascii'), v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """

    assert (len(imagePathList) == len(labelList))

    nSamples = len(imagePathList)

    env = lmdb.open(outputPath, map_size=36399847833)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue

        with open(imagePath, 'rb') as f:
            imageBin = f.read()
            
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode('utf-8') #sua loi

        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i]).encode('utf-8')
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples).encode('ascii')
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def get_mydataset():
    lines = open('C:/Users/hp/Documents/Nam3/NCKH/data_split/test_image/gt.txt', 'r', encoding='utf-8').readlines()
    image_list = []
    label_list = []
    for line in lines:
        line = line.strip()
        image, label = line.split()
        image_list.append(image)
        label_list.append(label)

    return image_list, label_list

if __name__ == '__main__':
    imgList, labelList = get_mydataset()
    print("The length of the list is ", len(imgList))

    '''Input the address you want to generate the lmdb file.'''
    createDataset('C:/Users/hp/Documents/Nam3/NCKH/saving_lmdb'.format(), imgList, labelList)