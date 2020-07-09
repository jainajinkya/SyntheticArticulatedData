import argparse
import os

from generation.inspect_data import make_animations
from sapien_dataset.generator_sapien import SceneGeneratorSapien


def main(args):
    print("NOTE: THIS NEEDS MUJOCO TO BE RUN IN HEADLESS MODE. MAKE SURE YOU DO `unset LD_PRELOAD` BEFORE RUNNING THIS")
    if args.obj == 'microwave':
        # All have local axis along +y
        # test_obj_idxs = ['7304', '7349']
        # train_obj_idxs = ['7119', '7128', '7236']

        # All have local axis along +z
        # test_obj_idxs = ['7167', '7263', '7310']
        # train_obj_idxs = ['7366']

        # Mixed
        test_obj_idxs = ['7304', '7349', '7167']
        train_obj_idxs = ['7119', '7128', '7236', '7263', '7310', '7366', '7273', '7265']

    # initialize Generator
    scenegen = SceneGeneratorSapien(obj_idxs=train_obj_idxs,
                                    xml_dir=args.xml_dir,
                                    root_dir=args.dir,
                                    debug_flag=args.debug,
                                    masked=args.masked)

    # make root directory
    os.makedirs(os.path.abspath(args.dir), exist_ok=True)

    # set generator's target directory for train data
    train_dir = os.path.abspath(os.path.join(args.dir, args.obj))
    print('Generating training data in %s ' % train_dir)
    os.makedirs(train_dir, exist_ok=False)
    scenegen.savedir = train_dir

    # generate train scenes
    scenegen.generate_scenes(args.n, args.obj)

    # set generator's target directory for test data
    test_dir = os.path.abspath(os.path.join(args.dir, args.obj + '-test'))
    os.makedirs(test_dir, exist_ok=False)
    print('Generating test data in %s ' % test_dir)
    scenegen.savedir = test_dir

    # generate test scenes
    scenegen.obj_idxs = test_obj_idxs
    scenegen.generate_scenes(int(args.n / 10), args.obj)

    # generate visualization for sanity
    if args.debug:
        make_animations(train_dir, args.n * 16, use_color=args.debug)
        make_animations(test_dir, int(args.n/10) * 16, use_color=args.debug)


parser = argparse.ArgumentParser(description="tool for generating articulated object data")
parser.add_argument('--n', type=int, default=int(1),
                    help='number of examples to generate')
parser.add_argument('--dir', type=str, default='../microtrain/')
parser.add_argument('--obj', type=str, default='microwave')
# parser.add_argument('--obj-xml-file', type=str, help='path to object xml file')
parser.add_argument('--xml-dir', type=str, help='path to object xml file directory')
parser.add_argument('--masked', action='store_true', default=False, help='remove background of depth images?')
parser.add_argument('--debug', action='store_true', default=False)
main(parser.parse_args())
