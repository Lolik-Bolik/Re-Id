from torchreid.utils import FeatureExtractor
import torchvision
import torch
import argparse
from glob import glob
import os
import utils
import numpy as np
# import plotly.express as px
import cv2 

def visualization(queries_imgs_paths, gallery_imgs_paths, distances, scale_percent):
    for query_path, gallery_path, distance in zip(queries_imgs_paths, gallery_imgs_paths, distances):
        print(query_path, gallery_path, distance)
        img1 = cv2.imread(query_path)
        img2 = cv2.imread(gallery_path)
        canvas = np.ones(img1.shape, dtype=np.uint8) * 255
        canvas = cv2.putText(canvas, f"{round(distance, 2)}", 
                            (15, 15) , cv2.FONT_HERSHEY_SIMPLEX,  
                            0.5, (255, 0, 0) , 1, cv2.LINE_AA)
        result_image = cv2.hconcat([img1,canvas, img2])
        width = int(result_image.shape[1] * scale_percent / 100)
        height = int(result_image.shape[0] * scale_percent / 100)
        new_dim = (width, height)
        result_image = cv2.resize(result_image, new_dim, interpolation=cv2.INTER_CUBIC)
        # result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        # fig = px.imshow(result_image)
        # fig.show()
        cv2.imshow('Result', result_image)
        save_dir = os.path.join("results_images", os.path.basename(query_path))
        cv2.imwrite(f"{save_dir}",result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def inference(opts):
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=opts.path_to_model,
        device='cuda'
    )
    # extract feautures from gallery
    test_data = torchvision.datasets.ImageFolder(root=os.path.join(opts.data_path, "gallery"))
    gallery_imgs_paths, target_ids = zip(*test_data.imgs)
    gallery_features = extractor(list(gallery_imgs_paths))
    
    # extract feature from queries images
    queries_imgs_paths = sorted(glob(os.path.join(opts.data_path, "queries", "*")))
    queries_features = extractor(queries_imgs_paths)
    dist_matrix = utils.compute_distance_matrix(gallery_features, queries_features).cpu().numpy()
    ind = np.argmin(dist_matrix, axis=0)
    distances_values = np.take_along_axis(dist_matrix, ind[np.newaxis, :], axis=0)[0]
    visualization(queries_imgs_paths, np.take(gallery_imgs_paths, ind), distances_values, 300)


if  __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference ReId Net.')
    parser.add_argument('--data_path', type=str, default="test_data", help='path to test data')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--path_to_model', type=str, default="/home/alexander/Downloads/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth", help='path to model ')
    args = parser.parse_args()
    inference(args)