import numpy as np
import h5py
import copy
import argparse


def combine_datasets(filenames, output_dir):
    def change_key_id_by(key_in, delta=0):
        return "obj_" + str(int(copy.copy(key_in).replace("obj_", "")) + delta).zfill(6)

    bigDataset = h5py.File(output_dir + 'complete_data.hdf5', 'w')
    i = 0
    for f in filenames:
        data_in = h5py.File(f, 'r')
        for k in data_in.keys():
            bigDataset.copy(data_in[k], change_key_id_by(k, i))
        i += len(data_in.keys())
        data_in.close()
    bigDataset.close()
    print("Combined datasets in file {}".format(output_dir + 'complete_data.hdf5'))


def subsample_dataset(file_in, sub_size, output_dir):
    orig_data = h5py.File(file_in, 'r')
    if sub_size >= len(orig_data.keys()):
        return orig_data
    else:
        ids = np.random.choice(len(orig_data.keys()), size=sub_size)
        sub_dataset = h5py.File(output_dir + 'complete_data.hdf5', 'w')
        original_keys = list(orig_data.keys())
        for i, id in enumerate(ids):
            sub_dataset.copy(orig_data[original_keys[id]], "obj_" + str(i).zfill(6))
        sub_dataset.close()
        print("Created subsampled dataset file at: {}".format(output_dir + 'complete_data.hdf5'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Datasets utils. Available functions: combine_datasets, "
                                                 "split_datasets")
    parser.add_argument('-i', '--input_files', type=str, nargs='+',
                        help='Input Dataset name(s), separated by spaces')
    parser.add_argument('-o', '--output-dir', type=str, help='path to output dir')
    parser.add_argument('-c', '--combine', action='store_true', default=False, help='Combine provided datasets')
    parser.add_argument('-sub', '--subsample', action='store_true', default=False, help='Randomly subsample dataset')
    parser.add_argument('-ns', '--sub_size', type=int, default=0, help='New size of subsampled dataset')

    args = parser.parse_args()
    if args.combine:
        combine_datasets(filenames=args.input_files, output_dir=args.output_dir)
    elif args.subsample:
        subsample_dataset(file_in=args.input_files, sub_size=args.sub_size, output_dir=args.output_dir)
    else:
        print("Only combine and subsample functions implemented yet!")
