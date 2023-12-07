import argparse
import os.path

import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import sam_model_registry


def draw_masks_fromDict(image, masks_generated):
    masked_image = image.copy()
    for i in range(len(masks_generated)):
        masked_image = np.where(np.repeat(masks_generated[i]['segmentation'].astype(int)[:, :, np.newaxis], 3, axis=2),
                                np.random.choice(range(256), size=3),
                                masked_image)
        masked_image = masked_image.astype(np.uint8)
    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)


def A(args):
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam = sam_model_registry[args.model_type](checkpoint=args.pth).to(device=args.device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    segmented_image = draw_masks_fromDict(image, masks)
    cv2.imshow('vis', segmented_image)
    cv2.waitKey(5)
    new_image_name = args.image_path.split('/')[-1].split('.')[0] + '.png'
    cv2.imwrite(os.path.join(args.output, new_image_name), segmented_image)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default=None)
    parser.add_argument('--model-type', type=str, default='vit_b')
    parser.add_argument('--pth', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    A(args)
