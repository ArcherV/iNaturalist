import random
import sys
import lmdb
import json
import six
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import sampler


class lmdbDataset(Dataset):
    def __init__(self, root=None, input_size=(448, 448), transform=None, is_train=True):
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % root)
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int.from_bytes(txn.get('nSamples'.encode()), byteorder='little')
            categories = json.loads(str(txn.get('categories'.encode()), encoding='utf-8'))
            self.categories = categories
            self.nSamples = nSamples

        self.is_train = is_train
        self.transform = transform
        # augmentation params
        self.im_size = input_size # can change this to train on higher res
        self.mu_data = [0.485, 0.456, 0.406]
        self.std_data = [0.229, 0.224, 0.225]
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.25

        # augmentations
        self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
        self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            img_buf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(img_buf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')
            except IOError:
                print('Corrupted image for %d' % index)
                sys.exit(1)

            label_key = 'label-%09d' % index
            label = int.from_bytes(txn.get(label_key.encode()), byteorder='little')

        if self.transform:
            img = self.transform(img)
        if self.is_train:
            img = self.scale_aug(img)
            img = self.flip_aug(img)
            img = self.color_aug(img)
        else:
            img = self.center_crop(img)

        img = self.tensor_aug(img)
        img = self.norm_aug(img)

        return index, img, label

    def getCategories(self):
        return self.categories


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)
