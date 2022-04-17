import numpy as np
from PIL import Image

import imgaug as ia
from imgaug import augmenters as iaa
from torchvision import transforms

ia.seed(1)

loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()


def TTA(image, AUGLOOP):
    new_image = []
    new_image.append(image)
    image = unloader(image.squeeze(0))
    # 影像增强
    seq = iaa.Sequential([
        iaa.Rot90([0, 1, 2, 3]),
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        iaa.Fliplr(0.5),  # 镜像
        iaa.Multiply((0.9, 1.2)),  # change brightness, doesn't affect BBs
        iaa.GaussianBlur(sigma=(0, 0.3)),  # iaa.GaussianBlur(0.5),
        iaa.Affine(
            translate_px={"x": 0, "y": 0},
            scale=(0.9, 1.2),
            rotate=(0, 0)
        )
    ])

    if AUGLOOP != 0:
        for epoch in range(AUGLOOP):
            seq_det = seq.to_deterministic()
            img = image
            img = np.asarray(img)
            image_auged = seq_det.augment_images([img])[0]
            image_after_aug = Image.fromarray(image_auged)
            image_after_aug = loader(image_after_aug).unsqueeze(0)
            new_image.append(image_after_aug)

    return new_image
