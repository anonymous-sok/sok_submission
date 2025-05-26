TINY_IMAGENET_200_FOLDER = {
    "train": None,
    "val": None,
    "test": "/opt/dlami/nvme/tiny-imagenet-200/test/images",
    "format": ".jpeg"
}

CIFAR_10_FOLDER = {
    "test": "/opt/dlami/nvme/cifar-10-test-batch/test_batch",
    "format": "pickle"
}

SLOWTRACK_DATASET_FOLDER = {
    "test":"/opt/dlami/nvme/SlowTrack/dataset",
    "format": ".jpg"
}

FLICKR_8K_FOLDER = {
    "dataset": "/opt/dlami/nvme/flickr8k",
}

FLICKR_8K_5_5_HDF5 = {
    "train": "/opt/dlami/nvme/flickr8k_5_5/TRAIN_IMAGES_flickr8k_5_cap_per_img_5_min_word_freq.hdf5",
    "val": "/opt/dlami/nvme/flickr8k_5_5/VAL_IMAGES_flickr8k_5_cap_per_img_5_min_word_freq.hdf5",
    "test": "/opt/dlami/nvme/flickr8k_5_5/TEST_IMAGES_flickr8k_5_cap_per_img_5_min_word_freq.hdf5",
    "format": ".hdf5"
}

COCO_FOLDER = {
    "train": "/opt/dlami/nvme/coco/train2014",
    "val": "/opt/dlami/nvme/coco/val2014",
    "format": ".jpg"
}

COCO_5_3_HDF5 = {
    "train": "/opt/dlami/nvme/coco_5_3/TRAIN_IMAGES_coco_5_cap_per_img_3_min_word_freq.hdf5",
    "val": "/opt/dlami/nvme/coco_5_3/VAL_IMAGES_coco_5_cap_per_img_3_min_word_freq.hdf5",
    "test": "/opt/dlami/nvme/coco_5_3/TEST_IMAGES_coco_5_cap_per_img_3_min_word_freq.hdf5",
    "format": ".hdf5"
}

DATA_FORMAT = {
    "TINY_IMAGENET_200_FOLDER": ".jpeg",
    "CIFAR_10_FOLDER": "pickle",
    "SLOWTRACK_DATASET_FOLDER": ".jpg",
    "FLICKR_8K_FOLDER": ".hdf5",
    "COCO_FOLDER": ".jpg", 
    "COCO_5_3_HDF5": ".hdf5",
}

ADV = [
    {
        "path": "/opt/dlami/nvme/CVPR22_NICGSlowDown/adv", 
        "data_format": ".adv"
    },
    {
        "path": "/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_resnet56_sdn_ic_only", 
        "data_format": ".pickle"
    },
    {
        "path": "/opt/dlami/nvme/DeepSloth/samples/cifar10/cifar10_vgg16bn_sdn_ic_only",
        "data_format": ".pickle"
    },
    {
        "path": "/opt/dlami/nvme/SlowTrack/botsort_stra_0.5",
        "data_format": ".jpg"
    },
    {
        "path": "/opt/dlami/nvme/SlowTrack/botsort_stra_0.25",
        "data_format": ".jpg"
    },
    {
        "path": "/opt/dlami/nvme/SlowTrack/botsort_stra_0.75",
        "data_format": ".jpg"
    },
]




if __name__ == "__main__":
    import os
    import pickle
    import pdb
    import numpy as np
    from PIL import Image as _PILImage
    from io import BytesIO
    import cv2
    
    def unpickle(file):
        """
        CIFAR-10
        """
        with open(file, 'rb') as fo:
            dict_ = pickle.load(fo, encoding='bytes')
        return dict_
    
    for adv_path in ADV:
        data_format = adv_path["data_format"]
        test_path = adv_path["path"]
        
        if data_format == ".pickle":
            file_names = [f for f in os.listdir(test_path) if f.endswith(data_format) and os.path.isfile(os.path.join(test_path, f))]
            for file_name in file_names:
                file_path = os.path.join(test_path, file_name)
                pdb.set_trace()
                data = unpickle(file_path)[0]
                print(f"DIR TO FILE: {os.path.join(test_path, file_name)}")
                print(f"SHAPE: {data.shape}")