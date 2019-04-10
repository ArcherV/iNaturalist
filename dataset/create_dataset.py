import os
import lmdb
import cv2
import json
import numpy as np


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(key=k.encode(), value=v)


def createDataset(outputPath, json_file, train=True):
    with open(json_file, 'r') as file:
        dic = json.load(file)
    images = dic['images']
    if train:
        categories = dic['categories']
        annotations = dic['annotations']
        categories = json.dumps(categories)
    nSamples = len(images)

    env = lmdb.open(outputPath, map_size=int(1e12))
    cache, cnt = {}, 0
    for i in range(nSamples):
        imagePath = images[i]['file_name']
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        imageKey = 'image-%09d' % cnt
        cache[imageKey] = imageBin
        if train:
            label = annotations[i]['category_id']
            labelKey = 'label-%09d' % cnt
            cache[labelKey] = label.to_bytes(4, byteorder='little')

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt
    cache['nSamples'] = nSamples.to_bytes(4, byteorder='little')
    if train:
        cache['categories'] = categories.encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    createDataset('testdata', 'test2019.json', False)

    # Below is how to access the data from the lmdb database
    # env = lmdb.open('traindata', max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    # with env.begin(write=False) as txn:
    #     nSamples = int.from_bytes(txn.get('nSamples'.encode()), byteorder='little')
    #     categories = json.loads(str(txn.get('categories'.encode()), encoding='utf-8'))
    #     la = int.from_bytes(txn.get(('label-%09d' % 0).encode()), byteorder='little')
    #     # img_buf = txn.get(('image-%09d' % 0).encode())
    #     # buf = six.BytesIO()
    #     # buf.write(img_buf)
    #     # buf.seek(0)
    #     # img = np.array(Image.open(buf))
    # print(categories)
