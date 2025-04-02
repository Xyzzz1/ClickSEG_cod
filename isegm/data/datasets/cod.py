import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from isegm.utils.misc import get_bbox_from_mask, get_labels_with_sizes
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class CODEvaluationDataset(ISDataset):
    def __init__(self, dataset_path, split='val', **kwargs):
        super(CODEvaluationDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'Imgs'
        self._insts_path = self.dataset_path / 'GT'

        with open('/home/yrq/dataset/COD/train.txt', 'r') as f:
            self.dataset_samples = [x.strip() for x in f.readlines()]

    def get_sample(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.jpg')
        inst_info_path = str(self._insts_path / f'{image_name}.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(inst_info_path, cv2.IMREAD_GRAYSCALE)
        instances_mask = np.array(mask, dtype=np.int32) // 255

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index), image_name


if __name__ == "__main__":
    dataset = CODEvaluationDataset('/home/yrq/dataset/COD/TrainDataset')
    for i in range(len(dataset.dataset_samples)):
        a = dataset.get_sample(i)
        pass
