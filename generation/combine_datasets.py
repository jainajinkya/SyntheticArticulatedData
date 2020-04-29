import h5py
import copy
import argparse


def main(args):
    def change_key_id_by(key_in, delta=0):
        return "obj_" + str(int(copy.copy(key_in).replace("obj_", "")) + delta).zfill(6)

    bigDataset = h5py.File(args.output_dir + 'complete_data.hdf5', 'w')
    i = 0
    for f in args.filenames:
        data_in = h5py.File(f, 'r')
        for k in data_in.keys():
            bigDataset.copy(data_in[k], change_key_id_by(k, i))
        i += len(data_in.keys())
        data_in.close()
    bigDataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to combine datasets")
    parser.add_argument('--filenames', '--i', type=str, nargs='+',
                        help='Enter dataset names with path to join, separated by spaces')
    parser.add_argument('--output-dir', '--o', type=str, help='path to output dir')
    main(parser.parse_args())