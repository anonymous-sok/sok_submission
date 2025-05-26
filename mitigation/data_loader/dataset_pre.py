import os
import cv2
import sys
import pdb
import copy
import h5py
import torch
import pickle
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed



sys.path.append("../")
random.seed(0)  
torch.manual_seed(0) 
torch.backends.cudnn.deterministic = True  


def unpickle(file):
    """
    CIFAR-10
    """
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_


def process_and_save(idx: int):
    img = data[idx]
    name = image_filename[idx]

    jpeg_img = defense.jpeg.slq(img, qualities=(90,), patch_size=8)
    nl_img   = defense.spatial.non_local_spatial_smoothing(
                   img,
                   h=10,
                   templateWindowSize=7,
                   searchWindowSize=21)

    Image.fromarray(jpeg_img).save(os.path.join(jpeg_path,   name))
    Image.fromarray(nl_img).save(os.path.join(spatial_path, name))


def process_pickle(idx):
    # flat -> HWC (32×32×3)
    if data[idx].ndim == 1:
        
        img = data[idx].reshape(32, 32, 3)
        jpg = defense.jpeg.slq(img, qualities=(90,), patch_size=8)
        nl  = defense.spatial.non_local_spatial_smoothing(
                    img, h=10, templateWindowSize=7, searchWindowSize=21)
        
        return idx, jpg.reshape(-1), nl.reshape(-1)
    
    elif data[idx].ndim == 3 and data[idx].shape[0] == 3:
        
        img = data[idx].transpose(1, 2, 0) # CHW -> HWC
        jpg = defense.jpeg.slq(img, qualities=(90,), patch_size=8)
        nl  = defense.spatial.non_local_spatial_smoothing(
                    img, h=10, templateWindowSize=7, searchWindowSize=21)
        return idx, jpg.transpose(2, 0, 1), nl.transpose(2, 0, 1)


def process_hdf5(idx):
    img_hwc = data[idx].transpose(1, 2, 0)
    jpg = defense.jpeg.slq(img_hwc, qualities=(90,), patch_size=8)
    nl  = defense.spatial.non_local_spatial_smoothing(
              img_hwc, h=10,
              templateWindowSize=7,
              searchWindowSize=21)
    return idx, jpg.transpose(2, 0, 1), nl.transpose(2, 0, 1)


def process_image(file_name, test_path, jpeg_path, spatial_path):
    image_path = os.path.join(test_path, file_name)
    image_array = np.array(Image.open(image_path).convert("RGB"))
    assert image_array.ndim == 3, "must be 3D"

    jpeg_image = defense.jpeg.slq(image_array, qualities=(90,), patch_size=8)
    non_local_image = defense.spatial.non_local_spatial_smoothing(image_array, h=10, templateWindowSize=7, searchWindowSize=21)

    Image.fromarray(jpeg_image).save(os.path.join(jpeg_path, file_name))
    Image.fromarray(non_local_image).save(os.path.join(spatial_path, file_name))
    

def process_adv_file(test_path, file_name):
    batches = torch.load(
        os.path.join(test_path, file_name),
        weights_only=False,
        map_location="cpu"
    )
    jpeg_batches = copy.deepcopy(batches)
    spatial_batches = copy.deepcopy(batches)


    total_iters = len(batches) * len(batches[0][0][0])
    pbar = tqdm(total=total_iters, desc=f"{file_name}", leave=False)
    for i in range(len(batches)):
        for j in range(len(batches[i])):
            ori_image_tensor, _ = batches[i][0]
            adv_image_tensor, _ = batches[i][1]
            for k in range(len(ori_image_tensor)):
                hwc_ori = ori_image_tensor[k].permute(1, 2, 0).cpu().numpy()
                hwc_adv = adv_image_tensor[k].permute(1, 2, 0).cpu().numpy()

                ori_jpeg = defense.jpeg.slq(hwc_ori, qualities=(90,), patch_size=8)
                adv_jpeg = defense.jpeg.slq(hwc_adv, qualities=(90,), patch_size=8)
                ori_spatial = defense.spatial.non_local_spatial_smoothing(
                    hwc_ori, h=10, templateWindowSize=7, searchWindowSize=21
                )
                adv_spatial = defense.spatial.non_local_spatial_smoothing(
                    hwc_adv, h=10, templateWindowSize=7, searchWindowSize=21
                )

                # 转回 tensor
                jpeg_batches[i][0][0][k]     = torch.from_numpy(ori_jpeg).permute(2,0,1)
                jpeg_batches[i][1][0][k]     = torch.from_numpy(adv_jpeg).permute(2,0,1)
                spatial_batches[i][0][0][k]  = torch.from_numpy(ori_spatial).permute(2,0,1)
                spatial_batches[i][1][0][k]  = torch.from_numpy(adv_spatial).permute(2,0,1)

                pbar.update(1)
    pbar.close()


    jpeg_out  = os.path.join(test_path, "jpeg",    file_name)
    spatial_out= os.path.join(test_path, "spatial", file_name)
    os.makedirs(os.path.dirname(jpeg_out),   exist_ok=True)
    os.makedirs(os.path.dirname(spatial_out), exist_ok=True)
    torch.save(jpeg_batches,   jpeg_out)
    torch.save(spatial_batches, spatial_out)
    
    
if __name__ == '__main__':
    from hardcoded_data_path import *
    import defense.jpeg
    import defense.spatial

    adv_paths = ADV

    for adv_path in adv_paths:
        
        data_format = adv_path["data_format"]
        test_path = adv_path["path"]
        
        print(f"Processing {test_path:<100}", f"---> format {data_format}")
        file_names = [f for f in os.listdir(test_path) if f.endswith(data_format) and os.path.isfile(os.path.join(test_path, f))]
        
        jpeg_path = os.path.join(test_path, "jpeg")
        spatial_path = os.path.join(test_path, "spatial")
        os.makedirs(jpeg_path, exist_ok=True)
        os.makedirs(spatial_path, exist_ok=True)
        
        if data_format == ".jpg" or data_format == ".jpeg" or data_format == ".png":

            data = []
            for file_name in tqdm(file_names):
                image_path = os.path.join(test_path, file_name)
                image_array = np.array(Image.open(image_path).convert("RGB"))
                data.append(image_array)
                assert image_array.ndim == 3, "must be 3D"
                
                jpeg_image = defense.jpeg.slq(image_array, qualities=(90,), patch_size=8)
                non_local_image = defense.spatial.non_local_spatial_smoothing(image_array, h=10, templateWindowSize=7, searchWindowSize=21)
                
                Image.fromarray(jpeg_image).save(os.path.join(jpeg_path, file_name))
                Image.fromarray(non_local_image).save(os.path.join(spatial_path, file_name))
                    
        if data_format == ".pickle":
            for file_name in file_names:   
                try:
                    unpickled_dict = unpickle(os.path.join(test_path, file_name))
                    
                    data = unpickled_dict[0]

                    data_jpeg_copy = copy.deepcopy(data)
                    data_non_local_copy = copy.deepcopy(data)
                    
                    d_num = data.shape[0]
                    d_shape = data.shape[1:]
                                            
                    print(f"Processing {test_path:<100}", f"---> filename {file_name}")
                    num_workers = 64
                    with Pool(processes=num_workers) as pool:
                        for idx, jpg_HWC, nl_HWC in tqdm(
                                pool.imap_unordered(process_pickle, range(d_num)),
                                total=d_num,
                                desc="Pickle defend"):
                            data_jpeg_copy[idx]     = jpg_HWC
                            data_non_local_copy[idx] = nl_HWC
                                
                    jpeg_path = os.path.join(test_path, "jpeg", file_name)
                    spatial_path = os.path.join(test_path, "spatial", file_name)
                    os.makedirs(os.path.dirname(jpeg_path), exist_ok=True)
                    os.makedirs(os.path.dirname(spatial_path), exist_ok=True)
                    
                    jpeg_unpickled_dict = (data_jpeg_copy, unpickled_dict[1])
                    spatial_unpickled_dict = (data_non_local_copy, unpickled_dict[1])
                    
                    with open(jpeg_path, 'wb') as f:
                        pickle.dump(jpeg_unpickled_dict, f)
                    
                    with open(spatial_path, 'wb') as f:
                        pickle.dump(spatial_unpickled_dict, f)
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
                    continue
                  
        if data_format == ".adv":
            continue
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=mp.cpu_count()) as pool:
                
                pool.starmap(process_adv_file, [(test_path, f) for f in file_names])
            continue
            for file_name in file_names:
            
                batches = torch.load(os.path.join(test_path, file_name), weights_only=False, map_location="cpu")
                jpeg_batches = copy.deepcopy(batches)
                spatial_batches = copy.deepcopy(batches)
                
                print(f"Processing {test_path:<100}", f"---> filename {file_name}")
                pbar = tqdm(total=len(batches) * len(batches[0][0][0]))       
                       
                for i in range(len(batches)):
                    for j in range(len(batches[i])):
                        
                        ori_image_tensor, ori_caption_len = batches[i][0][0], batches[i][0][1]
                        adv_image_tensor, adv_caption_len = batches[i][1][0], batches[i][1][1]
                                                
                        for k in range(len(ori_image_tensor)):
                            # hwc_ori = ori_image_tensor[k].transpose(1, 2, 0).numpy()
                            # hwc_adv = adv_image_tensor[k].transpose(1, 2, 0).numpy()
                            
                            hwc_ori = batches[i][0][0][k].permute(1, 2, 0).cpu().numpy()
                            hwc_adv = batches[i][1][0][k].permute(1, 2, 0).cpu().numpy()
                            
                            ori_jpeg_image_array = defense.jpeg.slq(hwc_ori, qualities=(90,), patch_size=8)
                            adv_jpeg_image_array = defense.jpeg.slq(hwc_adv, qualities=(90,), patch_size=8)
                            ori_spatial_image_array = defense.spatial.non_local_spatial_smoothing(hwc_ori, h=10, templateWindowSize=7, searchWindowSize=21)
                            adv_spatial_image_array = defense.spatial.non_local_spatial_smoothing(hwc_adv, h=10, templateWindowSize=7, searchWindowSize=21)
                            
                            ori_jpeg_image_tensor = torch.from_numpy(ori_jpeg_image_array).permute(2, 0, 1)
                            adv_jpeg_image_tensor = torch.from_numpy(adv_jpeg_image_array).permute(2, 0, 1)
                            ori_spatial_image_tensor = torch.from_numpy(ori_spatial_image_array).permute(2, 0, 1)
                            adv_spatial_image_tensor = torch.from_numpy(adv_spatial_image_array).permute(2, 0, 1)
                            
                            jpeg_batches[i][0][0][k] = ori_jpeg_image_tensor
                            jpeg_batches[i][1][0][k] = adv_jpeg_image_tensor
                            
                            spatial_batches[i][0][0][k] = ori_spatial_image_tensor
                            spatial_batches[i][1][0][k] = adv_spatial_image_tensor
                            
                            pbar.update(1)
                    
                jpeg_path = os.path.join(test_path, "jpeg", file_name)
                spatial_path = os.path.join(test_path, "spatial", file_name)
                os.makedirs(os.path.dirname(jpeg_path), exist_ok=True)
                os.makedirs(os.path.dirname(spatial_path), exist_ok=True)
                torch.save(jpeg_batches, jpeg_path)
                torch.save(spatial_batches, spatial_path)
            
    """
    ===========================================================================================================================================================
    """
    raise ValueError("======================================== This is a test for the dataset pre-processing script. ========================================")
    """
    ===========================================================================================================================================================
    """
    
    data_paths = [
        TINY_IMAGENET_200_FOLDER,
        SLOWTRACK_DATASET_FOLDER,
        COCO_FOLDER, 
        FLICKR_8K_5_5_HDF5,
        CIFAR_10_FOLDER,
        COCO_5_3_HDF5,
    ]
    
    for path in data_paths:
        try:
            test_path = path["test"]
        except KeyError:
            test_path = path["val"]
        
        data_format = path["format"]
        
        print(f"Processing {test_path:<100}", f"---> format {data_format}")

        if data_format == ".jpg" or data_format == ".jpeg" or data_format == ".png":
            image_paths = [os.path.join(test_path, img) for img in os.listdir(test_path) if img.lower().endswith(data_format)]
            image_filename = [p.split("/")[-1] for p in image_paths]
            data = []
        
            for image_path in tqdm(image_paths):
                image_array = np.array(Image.open(image_path).convert("RGB"))
                data.append(image_array)
                
                assert image_array.ndim == 3, "must be 3D"
            
            data_jpeg_copy = copy.deepcopy(data)
            data_non_local_copy = copy.deepcopy(data)
            
            file_name = os.path.basename(test_path.removeprefix('/opt/dlami/nvme/').split("/")[0])
            jpeg_path = os.path.join("../img_clean_jpeg", file_name)
            spatial_path = os.path.join("../img_clean_spatial", file_name)
            os.makedirs(jpeg_path, exist_ok=True)
            os.makedirs(spatial_path, exist_ok=True)
            
            num_workers = 64
            with Pool(processes=num_workers) as pool:
                list(tqdm(pool.imap_unordered(process_and_save, range(len(data))),
                        total=len(data),
                        desc="Defending & saving images"))
                
            # for idx in tqdm(range(len(data))):
            #     image_array = data[idx]
                
            #     jpeg_image = defense.jpeg.slq(image_array, qualities=(90,), patch_size=8)
            #     non_local_image = defense.spatial.non_local_spatial_smoothing(image_array, h=10, templateWindowSize=7, searchWindowSize=21)
                
            #     Image.fromarray(jpeg_image).save(os.path.join(jpeg_path, image_filename[idx]))
            #     Image.fromarray(non_local_image).save(os.path.join(spatial_path, image_filename[idx]))
        
        
        if data_format == ".hdf5":

            with h5py.File(test_path, 'r') as f:
                
                data = f['images'][:]
                d_num = data.shape[0]
                d_shape = data[0].shape # Channel, Height, Width
                
                data_jpeg_copy = copy.deepcopy(data)
                data_non_local_copy = copy.deepcopy(data)
                
                num_workers = 64
                with Pool(processes=num_workers) as pool:
                    for idx, jpg_arr, nl_arr in tqdm(
                            pool.imap_unordered(process_hdf5, range(d_num)),
                            total=d_num,
                            desc="HDF5 defend"):
                        data_jpeg_copy[idx]     = jpg_arr
                        data_non_local_copy[idx] = nl_arr
                        
                # for idx in tqdm(range(d_num)):
                #     assert data[idx].ndim == 3, "must be 3D"
                    
                #     image_array = data[idx].transpose(1, 2, 0) # Channel, Height, Width -> Height, Width, Channel
                    
                #     jpeg_image = defense.jpeg.slq(image_array, qualities=(90,), patch_size=8).transpose(2, 0, 1) 
                #     non_local_image = defense.spatial.non_local_spatial_smoothing(image_array, h=10, templateWindowSize=7, searchWindowSize=21).transpose(2, 0, 1)
                    
                #     data_jpeg_copy[idx] = jpeg_image
                #     data_non_local_copy[idx] = non_local_image

                file_name = os.path.basename(test_path)

                jpeg_path = os.path.join("../img_clean_jpeg", file_name)
                spatial_path = os.path.join("../img_clean_spatial", file_name)

                with h5py.File(jpeg_path, 'w') as f_1:
                    f_1.create_dataset('images', data=data_jpeg_copy, compression='gzip')

                with h5py.File(spatial_path, 'w') as f_2:
                    f_2.create_dataset('images', data=data_non_local_copy, compression='gzip')
                    
        if data_format == "pickle" or data_format == ".pickle":
            unpickled_dict = unpickle(test_path)
            data = unpickled_dict[b'data']
            
            data_jpeg_copy = copy.deepcopy(data)
            data_non_local_copy = copy.deepcopy(data)
            
            d_num = data.shape[0]
            d_shape = data.shape[1:]
            
            num_workers = 64
            with Pool(processes=num_workers) as pool:
                for idx, jpg_flat, nl_flat in tqdm(
                        pool.imap_unordered(process_pickle, range(d_num)),
                        total=d_num,
                        desc="Pickle defend"):
                    data_jpeg_copy[idx]     = jpg_flat
                    data_non_local_copy[idx] = nl_flat
                
            # for idx in tqdm(range(d_num)):
            #     image_array = data[idx].reshape(32, 32, 3)
                
            #     jpeg_image = defense.jpeg.slq(image_array, qualities=(90,), patch_size=8)
            #     non_local_image = defense.spatial.non_local_spatial_smoothing(image_array, h=10, templateWindowSize=7, searchWindowSize=21)
                
            #     data_jpeg_copy[idx] = jpeg_image.reshape(3072,)
            #     data_non_local_copy[idx] = non_local_image.reshape(3072,)
                
            file_name = test_path.split("/")[-1]
            
            jpeg_path = os.path.join("../img_clean_jpeg", f"{file_name}")
            spatial_path = os.path.join("../img_clean_spatial", f"{file_name}")
            
            with open(jpeg_path, 'wb') as f:
                pickle.dump(data_jpeg_copy, f)
            
            with open(spatial_path, 'wb') as f:
                pickle.dump(data_non_local_copy, f)
                

