import os
import numpy as np
from typing import Tuple, Union
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import argparse

import PIL
import time

import pytorch_image_classification
from pytorch_image_classification import get_default_config, create_transform
from pytorch_image_classification.utils import (
    create_logger,
    get_rank,
)

from collections import OrderedDict
import csv

from augmentation_tta import TTA
from torch.nn import functional as F
from can import CAN, PrepareCAN

# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()


def pil_grayscale_loader(path):
    with open(path, 'rb') as f:
        image = PIL.Image.open(f)
        return image.copy()


def predictCan(config, model, test_loader, logger, ttaNum):
    device = torch.device(config.device)
    AUGLOOP = ttaNum

    model.eval()
    resultProb = []
    imageNames = []

    start = time.time()
    pred_label_all = []
    pred_label_all_dict = OrderedDict()

    num_classes = 10

    with torch.no_grad():
        for image_name, image in test_loader:

            result_set = 0
            image_set = TTA(image, AUGLOOP)

            for image in image_set:
                image = (image - 0.406796) / 0.134280
                image = image.to(device)

                outputs = model(image)

                mid_result = F.softmax(outputs)

                mid_result[0][0] = mid_result[0][0]

                result_set += mid_result

            print(result_set[0][0])

            result_set = result_set / len(image_set)
            resultProb.append(result_set[0].cpu().numpy().tolist())
            imageNames.append(image_name)
        preparecan = PrepareCAN(resultProb, imageNames, thre=0.9)  # thre不能太小
        A0, A0_names, B0, B0_names = preparecan.split()
        delta_q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        can = CAN(alpha=1, d=1, delta_q=delta_q, A0=A0, B0=B0)
        B0_ = can.reAdjusted()
        results = np.concatenate((A0, B0_), axis=0)
        imageNames_ = A0_names + B0_names
        for prob, image_name in zip(results, imageNames_):
            preds = [prob.argmax() + 10 - num_classes]
            print(preds)
            pred_label_all.append(preds)
            imageID_classID = {image_name: preds}
            pred_label_all_dict.update(imageID_classID)

        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')
        prob_class = np.sum(results, axis=0) / results.shape[0]
        print(prob_class)

        predictLabelAll = np.concatenate(pred_label_all)
        return predictLabelAll, pred_label_all_dict


class TestDataset(Dataset):

    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.file_list = os.listdir(self.file_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_name, img = self.image_read(idx)
        img = self.transform(img)
        return image_name, img

    def image_read(self, idx):
        return self.file_list[idx], PIL.Image.open(os.path.join(self.file_path, self.file_list[idx]))


class Normalize:

    def __call__(self, image: PIL.Image.Image) -> np.ndarray:
        image = np.asarray(image).astype(np.float32) / 255.
        return image


class ToTensor:
    def __call__(
            self, data: Union[np.ndarray, Tuple[np.ndarray, ...]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(data, tuple):
            return tuple([self._to_tensor(image) for image in data])
        else:
            return self._to_tensor(data)

    @staticmethod
    def _to_tensor(data: np.ndarray) -> torch.Tensor:
        if len(data.shape) == 3:
            return torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(data[None, :, :].astype(np.float32))


def list2str(list):
    list_str = "".join([str(x) for x in list])
    return list_str


def csv_writer(fileName='', dataDict={}):
    with open(fileName, "w") as csvFile:
        fieldnames = ['image_id', 'class_id']
        writer = csv.DictWriter(csvFile, fieldnames)
        writer.writeheader()
        csvWriter = csv.writer(csvFile)
        for k, v in dataDict.items():
            name = list2str(list(filter(str.isdigit, k[0])))
            csvWriter.writerow([name, str(v[0])])
        csvFile.close()


# def loadConfig():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--checkpoints", type=str, default=
#     '/media/disk1/ligongzhe/workspace/pytorch-cla-bk/experiments/sar2022/plp/exp09_balance_lr_0.01/checkpoint_00010.pth')
#     parser.add_argument("--ttaNum", type=int, default=12)
#
#     args = parser.parse_args()
#     return args


def main():
    # args = loadConfig()
    config = get_default_config()
    config.merge_from_file('./configs/sar10/shake_shake.yaml')

    logger = create_logger(name=__name__, distributed_rank=get_rank())

    test_transform = transforms.Compose([
        transforms.Resize([56, 56]),
        Normalize(),
        ToTensor(),
    ])

    # Data pipline
    dataset = TestDataset(config.test.dataset_dir, transform=test_transform)
    dataloader = DataLoader(dataset,
                            batch_size=config.test.batch_size,
                            num_workers=config.test.dataloader.num_workers,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=config.test.dataloader.pin_memory)
    device = torch.device(config.device)
    model = pytorch_image_classification.create_model(config)
    checkpoint = torch.load(config.test.checkpoint)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    predict_all, dict = predictCan(config, model, dataloader, logger, config.test.ttaNum)
    os.makedirs('./result', exist_ok=True)
    csv_writer('./result/results.csv', dict)


if __name__ == '__main__':
    main()
