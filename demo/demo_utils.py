import numpy as np
import os

from ResSR.utils import image_utils, decimation_utils

def normalize_image(img, high, low):
    """
    Normalize image.

    Args:
        img (np.ndarray): The input image array.
        high (float): The upper bound of the desired range.
        low (float): The lower bound of the desired range.

    Returns:
        ndarray: The normalized image array.

    """
    return (img - low) / (high - low)

def create_false_color(img, triple_idx, low, high, save_path=None):
    """
    Create a false color image by mapping pixel values from three different channels of the input image to RGB channels.

    Args:
        img (numpy.ndarray): The input image with three channels, shape must be N x M x 3.
        triple_idx (tuple): A tuple of three indices representing the channels to use for each RGB channel.
        low (float): The lower bound for pixel value normalization.
        high (float): The upper bound for pixel value normalization.
        save_path (str, optional): The path to save the false color image. Defaults to None.

    Returns:
        numpy.ndarray: The false color image.

    """
    img_false_color = np.zeros((img.shape[0], img.shape[1], 3))
    img_false_color[:, :, 0] = np.clip(normalize_image(img[:, :, triple_idx[0]], high, low), 0, 1)
    img_false_color[:, :, 1] = np.clip(normalize_image(img[:, :, triple_idx[1]], high, low), 0, 1)
    img_false_color[:, :, 2] = np.clip(normalize_image(img[:, :, triple_idx[2]], high, low), 0, 1)

    if save_path is not None:
        image_utils.im_save(img_false_color, save_path)

    return img_false_color

def save_paper_figures(img_name, results_subdir, crop_region=None, methods=['ResSR', 'LRTA', 'DSen2', 'SupReME'], false_color_20m=[[5, 3, 0], [3, 2, 1], [5, 4, 2], [4, 3, 2]], false_color_60m=[[1, 1, 0], [0, 0, 0], [1, 1, 1]]):
    """
    Save paper figures for the given image.

    Parameters:
        img_name (str): The name of the image.
        results_subdir (str): The subdirectory where the results are stored.
        crop_region (tuple): The region to crop the image. Default is None.
        methods (list): The list of methods to include in the figures. Default is ['ResSR', 'LRTA', 'DSen2', 'SupReME'].
        false_color_20m (list): The false color combinations for 20m resolution images. Default is [[5, 3, 0], [3, 2, 1], [5, 4, 2], [4, 3, 2]].
        false_color_60m (list): The false color combinations for 60m resolution images. Default is [[1, 1, 0]].

    Returns:
        None
    """
    if img_name == "APEX":
        img_dir = '../data/APEX_dataset/' + img_name
        results_dir = '../results/APEX_dataset/uncropped/' + img_name + "/" + results_subdir  + "/"
    else:
        img_dir = "../data/Sentinel-2/" + img_name 
        results_dir = "../results/Sentinel-2/cropped_0_1080,0_1080/" + img_name + "/" + results_subdir + "/"
    
    save_path = "../paper_figures/" + img_name + "/" + results_subdir + "/"

    os.makedirs(save_path, exist_ok=True)

    (
        y_10m,
        y_20m,
        y_60m,
        band_name_list_10m,
        band_name_list_20m,
        band_name_list_60m,
    ) = image_utils.load_S2_images(img_dir)

    if crop_region is None:
        cropped = False
    else:
        cropped = True

    # #####
    # Create simulated data
    # #####
    simulated = True if "simulated" in results_dir else False
    if simulated:
        L_synth = 2 if results_dir=="simulated_L=2" else 6
    else:
        L_synth = 1
    LRTA_valid = False if L_synth == 6 else True

    if simulated:
        y_10m_gt = y_10m
        y_20m_gt = y_20m
        y_60m_gt = y_60m

        y_10m_lr = np.zeros((y_10m.shape[0] // L_synth, y_10m.shape[1] // L_synth, y_10m.shape[2]))
        y_20m_lr = np.zeros((y_20m.shape[0] // L_synth, y_20m.shape[1] // L_synth, y_20m.shape[2]))
        y_60m_lr = np.zeros((y_60m.shape[0] // L_synth, y_60m.shape[1] // L_synth, y_60m.shape[2]))
        for i in range(y_10m.shape[2]):
            y_10m_lr[:, :, i] = decimation_utils.simulated_data_decimation(y_10m[:, :, i], L_synth)
        for i in range(y_20m.shape[2]):
            y_20m_lr[:, :, i] = decimation_utils.simulated_data_decimation(y_20m[:, :, i], L_synth)
        for i in range(y_60m.shape[2]):
            y_60m_lr[:, :, i] = decimation_utils.simulated_data_decimation(y_60m[:, :, i], L_synth)
        y_10m = y_10m_lr
        y_20m = y_20m_lr
        y_60m = y_60m_lr

    if img_name == 'APEX':
        y_10m_gt = y_10m
        y_20m_gt = y_20m
        y_60m_gt = y_60m
        
        y_20m_lr = np.zeros((y_20m.shape[0] // 2, y_20m.shape[1] // 2, y_20m.shape[2]))
        y_60m_lr = np.zeros((y_60m.shape[0] // 6, y_60m.shape[1] // 6, y_60m.shape[2]))

        for i in range(y_20m.shape[2]):
            y_20m_lr[:, :, i] = decimation_utils.simulated_data_decimation(y_20m[:, :, i], 2)
        for i in range(y_60m.shape[2]):
            y_60m_lr[:, :, i] = decimation_utils.simulated_data_decimation(y_60m[:, :, i], 6)

        y_20m = y_20m_lr
        y_60m = y_60m_lr


    if "ResSR" in methods:
        (
            y_10m_res_sr,
            y_20m_res_sr,
            y_60m_res_sr,
            band_name_list_10m,
            band_name_list_20m,
            band_name_list_60m,
        ) = image_utils.load_S2_images(results_dir + "ResSR/")

    if "PCA_SR" in methods:
        (
            y_10m_pca_sr,
            y_20m_pca_sr,
            y_60m_pca_sr,
            band_name_list_10m,
            band_name_list_20m,
            band_name_list_60m,
        ) = image_utils.load_S2_images(results_dir + "PCA_SR/")

    if "SupReME" in methods:
        (
            y_10m_supreme,
            y_20m_supreme,
            y_60m_supreme,
            band_name_list_10m,
            band_name_list_20m,
            band_name_list_60m,
        ) = image_utils.load_S2_images(results_dir + "SupReME/")

    if "DSen2" in methods:
        (
            y_10m_dsen2,
            y_20m_dsen2,
            y_60m_dsen2,
            band_name_list_10m,
            band_name_list_20m,
            band_name_list_60m,
        ) = image_utils.load_S2_images(results_dir + "DSen2/")

    if "LRTA" in methods and LRTA_valid:
        (
            y_10m_lrta,
            y_20m_lrta,
            band_name_list_10m,
            band_name_list_20m,
        ) = image_utils.load_images(results_dir + "LRTA/")

    if cropped:
        y_20m = y_20m[crop_region[0]//2:crop_region[2]//2, crop_region[1]//2:crop_region[3]//2, :]
        if (simulated and L_synth == 2) or img_name == 'APEX':
            y_20m_gt = y_20m_gt[crop_region[0]:crop_region[2], crop_region[1]:crop_region[3], :]
        if "LRTA" in methods and LRTA_valid:
            y_20m_lrta = y_20m_lrta[crop_region[0]:crop_region[2], crop_region[1]:crop_region[3], :]
        if "DSen2" in methods:
            y_20m_dsen2 = y_20m_dsen2[crop_region[0]:crop_region[2], crop_region[1]:crop_region[3], :]
        if "SupReME" in methods:
            y_20m_supreme = y_20m_supreme[crop_region[0]:crop_region[2], crop_region[1]:crop_region[3], :]
        if "ResSR" in methods:
            y_20m_res_sr = y_20m_res_sr[crop_region[0]:crop_region[2], crop_region[1]:crop_region[3], :]
        if "PCA_SR" in methods:
            y_20m_pca_sr = y_20m_pca_sr[crop_region[0]:crop_region[2], crop_region[1]:crop_region[3], :]
        if not simulated or img_name != 'APEX':
            y_10m = y_10m[crop_region[0]:crop_region[2], crop_region[1]:crop_region[3], :]

    
    for triple_idx in false_color_20m:
        high = np.max([np.percentile(y_20m[:, :, triple_idx[i]], 98) for i in range(3)]) + 0.01
        low = np.min([np.percentile(y_20m[:, :, triple_idx[i]], 2) for i in range(3)]) - 0.01

        idx_str = "[" + str(triple_idx[0]) + ", " + str(triple_idx[1]) + ", " + str(triple_idx[2]) + "]"
        
        false_color_imgs = {}
        if not simulated or img_name != 'APEX':
            high_10m = np.percentile(y_10m[:, :, 0], 98) + 0.01
            low_10m = np.percentile(y_10m[:, :, 0], 2) - 0.01
            false_color_imgs['10m Bands'] = create_false_color(y_10m, (2, 1, 0), low_10m, high_10m, save_path=save_path + idx_str + "_10m.png")
        false_color_imgs['Low Resolution'] = create_false_color(y_20m, triple_idx, low, high, save_path=save_path + idx_str + "_LR.png")
        if (simulated and L_synth == 2) or img_name == 'APEX':
            false_color_imgs['Ground Truth'] = create_false_color(y_20m_gt, triple_idx, low, high, save_path=save_path + idx_str + "_GT.png")
        if "LRTA" in methods and LRTA_valid:
            false_color_imgs['LRTA'] = create_false_color(y_20m_lrta, triple_idx, low, high, save_path=save_path + idx_str + "_LRTA.png")
        if "DSen2" in methods:
            false_color_imgs['DSen2'] = create_false_color(y_20m_dsen2, triple_idx, low, high, save_path=save_path + idx_str + "_DSen2.png")
        if "SupReME" in methods:
            false_color_imgs['SupReME'] = create_false_color(y_20m_supreme, triple_idx, low, high, save_path=save_path + idx_str + "_SupReME.png")
        if "PCA_SR" in methods:
            false_color_imgs['PCA_SR'] = create_false_color(y_20m_pca_sr, triple_idx, low, high, save_path=save_path + idx_str + "_PCA_SR.png")
        if "ResSR" in methods:
            false_color_imgs['ResSR'] = create_false_color(y_20m_res_sr, triple_idx, low, high, save_path=save_path + idx_str + "_ResSR.png")

        image_utils.im_show(
            false_color_imgs,
            "",
            1,
            len(false_color_imgs.keys()),
            (len(false_color_imgs.keys()) * 7, 6),
            display_range=(np.min(false_color_imgs['Low Resolution']), np.max(false_color_imgs['Low Resolution'])),
            nchannels=3,
            save_path= save_path + idx_str + "_comp.png",
            cmap="viridis",
        )

    if cropped:
        
        y_60m = y_60m[crop_region[0]//6:crop_region[2]//6, crop_region[1]//6:crop_region[3]//6, :]
        if (simulated and L_synth == 6) or img_name == 'APEX':
            y_60m_gt = y_60m_gt[crop_region[0]:crop_region[2], crop_region[1]:crop_region[3], :]
        if "DSen2" in methods:  
            y_60m_dsen2 = y_60m_dsen2[crop_region[0]:crop_region[2], crop_region[1]:crop_region[3], :]
        if "SupReME" in methods:
            y_60m_supreme = y_60m_supreme[crop_region[0]:crop_region[2], crop_region[1]:crop_region[3], :]
        if "ResSR" in methods:
            y_60m_res_sr = y_60m_res_sr[crop_region[0]:crop_region[2], crop_region[1]:crop_region[3], :]
        if "PCA_SR" in methods:
            y_60m_pca_sr = y_60m_pca_sr[crop_region[0]:crop_region[2], crop_region[1]:crop_region[3], :]

    for triple_idx in false_color_60m:
        high = np.max([np.percentile(y_60m[:, :, triple_idx], 99) for i in range(3)]) + 0.01
        low = np.min([np.percentile(y_60m[:, :, triple_idx], 1) for i in range(3)]) - 0.01

        idx_str = "[" + str(triple_idx[0]) + ", " + str(triple_idx[1]) + ", " + str(triple_idx[2]) + "]"
        
        false_color_imgs = {}
        if not simulated or img_name != 'APEX':
            high_10m =np.percentile(y_10m[:, :, 0], 99) + 0.01
            low_10m = np.percentile(y_10m[:, :, 0], 1) - 0.01
            false_color_imgs['10m Bands'] = create_false_color(y_10m, (2, 1, 0), low_10m, high_10m, save_path=save_path + idx_str + "_10m.png")
        false_color_imgs['Low Resolution'] = create_false_color(y_60m, triple_idx, low, high, save_path=save_path + idx_str + "_LR.png")
        if (simulated and L_synth == 6) or img_name == 'APEX':
            false_color_imgs['Ground Truth'] = create_false_color(y_60m_gt, triple_idx, low, high, save_path=save_path + idx_str + "_GT.png")
        if "DSen2" in methods:
            false_color_imgs['DSen2'] = create_false_color(y_60m_dsen2, triple_idx, low, high, save_path=save_path + idx_str + "_DSen2.png")
        if "SupReME" in methods:
            false_color_imgs['SupReME'] = create_false_color(y_60m_supreme, triple_idx, low, high, save_path=save_path + idx_str + "_SupReME.png")
        if "PCA_SR" in methods:
            false_color_imgs['PCA_SR'] = create_false_color(y_60m_pca_sr, triple_idx, low, high, save_path=save_path + idx_str + "_PCA_SR.png")
        if "ResSR" in methods:
            false_color_imgs['ResSR'] = create_false_color(y_60m_res_sr, triple_idx, low, high, save_path=save_path + idx_str + "_ResSR.png")

        image_utils.im_show(
            false_color_imgs,
            "",
            1,
            len(false_color_imgs.keys()),
            (len(false_color_imgs.keys()) * 7, 6),
            display_range=(np.min(false_color_imgs['Low Resolution']), np.max(false_color_imgs['Low Resolution'])),
            nchannels=3,
            save_path= save_path + idx_str + "_comp.png",
            cmap="viridis",
        )