import os
import cv2
import numpy as np
import h5py
import copy
import argparse
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder

import SyntheticArticulatedData.generation.calibrations as calibrations
from SyntheticArticulatedData.generation.generator_v2 import vertical_flip, buffer_to_real, white_bg
from SyntheticArticulatedData.generation.mujocoDoubleCabinetParts import set_two_door_control


def combine_datasets(files_in, file_out):
    def change_key_id_by(key_in, delta=0):
        return "obj_" + str(int(copy.copy(key_in).replace("obj_", "")) + delta).zfill(6)

    bigDataset = h5py.File(file_out, 'w')
    i = 0
    for f in files_in:
        data_in = h5py.File(f, 'r')
        for k in data_in.keys():
            bigDataset.copy(data_in[k], change_key_id_by(k, i))
        i += len(data_in.keys())
        data_in.close()
    bigDataset.close()
    print("Combined datasets in file {}".format(file_out))


def subsample_dataset(file_in, sub_size, file_out, choose_random=False):
    orig_data = h5py.File(file_in, 'r')
    if sub_size > len(orig_data.keys()):
        return orig_data
    else:
        if choose_random:
            ids = np.random.choice(len(orig_data.keys()), size=sub_size)
        else:
            ids = np.arange(0, sub_size)

        sub_dataset = h5py.File(file_out, 'w')
        original_keys = list(orig_data.keys())
        for i, id in enumerate(ids):
            sub_dataset.copy(orig_data[original_keys[id]], "obj_" + str(i).zfill(6))
        orig_data.close()
        sub_dataset.close()
        print("Created subsampled dataset file at: {}".format(file_out))


def shuffle_dataset(file_in, file_out):
    orig_data = h5py.File(file_in, 'r')
    n = len(orig_data.keys())
    orig_data.close()
    subsample_dataset(file_in, n, file_out)
    print("Created shuffled dataset")


def debug_sample(file_in, obj_type, savedir, img_idx=0, masked=False):
    # Load image in mujoco
    model = load_model_from_path(file_in)
    sim = MjSim(model)
    modder = TextureModder(sim)

    if obj_type == 4 or obj_type == 5:
        # MULTI CABINET: get double the params.
        set_two_door_control(sim, 'cabinet2' if obj_type == 4 else 'refrigerator')
    else:
        if obj_type == 1:
            sim.data.ctrl[0] = 0.05
        # elif obj.geom[3] == 1:
        #     sim.data.ctrl[0] = -0.2
        else:
            sim.data.ctrl[0] = -0.2

    t = 0
    #########################
    IMG_WIDTH = calibrations.sim_width
    IMG_HEIGHT = calibrations.sim_height
    #########################

    while t < 4000:
        sim.forward()
        sim.step()

        if t % 500 == 0:
            img, depth = sim.render(IMG_WIDTH, IMG_HEIGHT, camera_name='external_camera_0', depth=True)
            img = vertical_flip(img)
            img = white_bg(img)
            imgfname = os.path.join(savedir, 'img' + str(img_idx).zfill(6) + '.png')
            cv2.imwrite(imgfname, img)
            img_idx += 1

        t += 1
    return img_idx


def debug_samples_using_ids(data_dir, obj_type, sample_ids, masked=False):
    img_idx = sample_ids[0] * 8  # Current frequency set at 8 images per object
    for i in sample_ids:
        fname = os.path.join(data_dir, 'scene' + str(i).zfill(6) + '.xml')
        img_idx = debug_sample(fname, obj_type, data_dir, img_idx, masked=masked)
    print("Images generated and saved in {}".format(data_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Datasets utils. Available functions: combine_datasets, "
                                                 "split_datasets")
    parser.add_argument('-i', '--files-in', type=str, nargs='+',
                        help='Input Dataset filename(s) or directories(s), separated by spaces')
    parser.add_argument('-o', '--file-out', type=str, help='output file name with path')

    parser.add_argument('-c', '--combine', action='store_true', default=False, help='Combine provided datasets')

    parser.add_argument('-sub', '--subsample', action='store_true', default=False, help='Randomly subsample dataset')
    parser.add_argument('-ns', '--sub_size', type=int, default=0, help='New size of subsampled dataset')
    parser.add_argument('--random', action='store_true', default=False, help='Should choose samples randomly?')

    parser.add_argument('--shuffle', action='store_true', default=False, help='Shuffle Dataset?')

    parser.add_argument('-d', '--debug-dataset', action='store_true', default=False,
                        help='Should generate images to visualize samples or not?. Input file argument corresponds '
                             'to input directory where scene files are stored.')
    parser.add_argument('--obj-id', type=int, default=0, help='Object class index')
    parser.add_argument('-sid', '--sample-ids', type=int, nargs='+', help='Sample ids to visualize')
    parser.add_argument('--masked', action='store_true', default=False, help='Masked image?')

    args = parser.parse_args()
    if args.combine:
        combine_datasets(files_in=args.files_in, file_out=args.file_out)
    elif args.subsample:
        subsample_dataset(file_in=args.files_in[0], sub_size=args.sub_size, file_out=args.file_out,
                          choose_random=args.random)
    elif args.shuffle:
        shuffle_dataset(file_in=args.files_in[0], file_out=args.file_out)
    elif args.debug_dataset:
        debug_samples_using_ids(data_dir=args.files_in[0], obj_type=args.obj_id, sample_ids=args.sample_ids,
                                masked=args.masked)
    else:
        print("Function implemented yet: combine, subsample, debug_sample!")
