import argparse
import os

from generation.inspect_data import make_animations
from sapien_dataset.generator_sapien import SceneGenerator as SceneGenerator_sapien


def main(args):
    print("NOTE: THIS NEEDS MUJOCO TO BE RUN IN HEADLESS MODE. MAKE SURE YOU DO `unset LD_PRELOAD` BEFORE RUNNING THIS")

    # initialize Generator
    scenegen = SceneGenerator_sapien(object_xml_file=args.obj_xml_file,
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
    # scenegen.generate_scenes(int(args.n / 5), args.obj)
    scenegen.generate_scenes(int(args.n / 10), args.obj)

    # generate visualization for sanity
    if args.debug:
        make_animations(os.path.join(args.dir, args.obj), min(100, args.n * 16), use_color=args.debug)


parser = argparse.ArgumentParser(description="tool for generating articulated object data")
parser.add_argument('--n', type=int, default=int(1),
                    help='number of examples to generate')
parser.add_argument('--dir', type=str, default='../microtrain/')
parser.add_argument('--obj', type=str, default='microwave')
parser.add_argument('--obj-xml-file', type=str, help='path to object xml file')
parser.add_argument('--masked', action='store_true', default=False, help='remove background of depth images?')
parser.add_argument('--debug', action='store_true', default=False)
main(parser.parse_args())
