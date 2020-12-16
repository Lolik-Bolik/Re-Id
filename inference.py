from torchreid.utils import FeatureExtractor
import torchvision
import torch
import argparse
from glob import glob
import os
import utils
import numpy as np

def inference(opts):
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='/home/alexander/Downloads/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
        device='cuda'
    )
    # extract feautures from gallery
    test_data = torchvision.datasets.ImageFolder(root=os.path.join(opts.data_path, "gallery"))
    gallery_imgs_paths, target_ids = zip(*test_data.imgs)
    gallery_features = extractor(list(gallery_imgs_paths))
    
    # extract feature from queries images
    queries_imgs_paths = sorted(glob(os.path.join(opts.data_path, "queries", "*")))
    queries_features = extractor(queries_imgs_paths)
    dist_matrix = utils.compute_distance_matrix(gallery_features, queries_features)
    ind = np.argmin(dist_matrix.cpu().numpy(), axis=0)
    print(np.take(target_ids, ind))


if  __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference ReId Net.')
    parser.add_argument('--data_path', type=str, default="test_data", help='path to test data')
    parser.add_argument('--batch_size', type=int, default=1, help='path to test data')
    args = parser.parse_args()
    inference(args)