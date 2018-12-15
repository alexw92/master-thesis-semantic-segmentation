import tensorflow as tf
import model
import sys
import argparse
import os


# FLAGS = tf.app.flags.FLAGS
#
# tf.app.flags.DEFINE_boolean('local_config', False, """ dont use this on slurm """)
# tf.app.flags.DEFINE_string('testing', '', """ checkpoint file """)
# tf.app.flags.DEFINE_string('finetune', '', """ finetune checkpoint file """)
# tf.app.flags.DEFINE_integer('batch_size', "5", """ batch_size """)
# tf.app.flags.DEFINE_float('learning_rate', "1e-4", """ initial lr """)
# tf.app.flags.DEFINE_float('gpu_usage', "1.0", """ the fraction of gpu memory used """)
# tf.app.flags.DEFINE_string('log_dir', "./=master/code/Segmentation_Models/SegNet/snapshots", """ dir to store ckpt """)
# tf.app.flags.DEFINE_string('image_dir', "./master/code/Segmentation_Models/SegNet/list/de_top14_train_list.txt",
#                            """ path to OSM train image list """)
# tf.app.flags.DEFINE_string('test_dir', "./master/code/Segmentation_Models/SegNet/list/de_top14_test_list.txt",
#                            """ path to OSM test image list """)
# tf.app.flags.DEFINE_string('val_dir', "./master/code/Segmentation_Models/SegNet/list/de_top14_test_list.txt",
#                            """ path to OSM val image list """)
# tf.app.flags.DEFINE_integer('max_steps', "150000", """ max_steps """)
# tf.app.flags.DEFINE_integer('image_h', "600", """ image height """)
# tf.app.flags.DEFINE_integer('image_w', "600", """ image width """)
# tf.app.flags.DEFINE_string('datadir', "/autofs/stud/werthmann/master/ANN_DATA/de_top14_17_600x600_cropped",
#                            """ path to OSM data dir """)
# tf.app.flags.DEFINE_string('output_dir', "./master/code/Segmentation_Models/SegNet/output",
#                            """ path to folder to save the output image if flag save_image is True""")
# #tf.app.flags.DEFINE_integer('image_h', "360", """ image height """)
# #tf.app.flags.DEFINE_integer('image_w', "480", """ image width """)
# tf.app.flags.DEFINE_integer('image_c', "3", """ image channel (RGB) """)
# tf.app.flags.DEFINE_integer('num_class', "6", """ total class number """)
# tf.app.flags.DEFINE_boolean('save_image', True, """ whether to save predicted image """)
# tf.app.flags.DEFINE_string('dataset', "detop14",
#                            """ the name of the dataset; has to be specified in Inputs.py as well""")
def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced SegNet")
    parser.add_argument("--local_config", type=bool, default=False,
                        help="dont use this on slurm")
    parser.add_argument("--infere", type=bool, default=False,
                        help="If True inference will be calculated instead of testing training or finetune")
    parser.add_argument("--max_infere", type=int, default=10,
                        help="number of files to infere, will only be used if --infere=True and if more than "
                             "--max_infere files are available")
    parser.add_argument("--testing", type=str, default='',
                        help="checkpoint file")
    parser.add_argument("--finetune", type=str, default='',
                        help="finetune checkpoint file")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="batch_size")
    parser.add_argument("--gpu_usage", type=float, default=1.0,
                        help="gpu usage fraction")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="initial lr")
    parser.add_argument("--log_dir", type=str, default="./master/code/Segmentation_Models/SegNet/snapshots",
                        help="dir to store ckpt")
    parser.add_argument("--train_dir", type=str,
                        default="./master/code/Segmentation_Models/SegNet/list/de_top14_r_train_list.txt",
                        help=" path to OSM train image list")
    parser.add_argument("--test_dir", type=str,
                        default="./master/code/Segmentation_Models/SegNet/list/de_top14_r_test_list.txt",
                        help=" path to OSM test image list ")
    parser.add_argument("--val_dir", type=str,
                        default="./master/code/Segmentation_Models/SegNet/list/de_top14_r_test_list.txt",
                        help="path to OSM val image list")
    parser.add_argument("--max_steps", type=int, default=1000000,
                        help="max steps")
    parser.add_argument("--image_h", type=int, default=600,
                        help="image height")
    parser.add_argument("--image_w", type=int, default=600,
                        help="image width")
    parser.add_argument("--datadir", type=str, default="./master/ANN_DATA/de_top14_17_600x600_cropped",
                        help="path to OSM data dir")
    parser.add_argument("--output_dir", type=str, default="./master/code/Segmentation_Models/SegNet/output",
                        help="path to folder to save the output image if flag save_image is True")
    parser.add_argument("--image_c", type=int, default=3,
                        help=" image channel (RGB)")
    parser.add_argument("--num_class", type=int, default=6,
                        help=" total class number")
    parser.add_argument("--save_image", type=bool, default=True,
                        help="whether to save predicted image")
    parser.add_argument("--dataset", type=str, default="detop15",
                        help="the name of the dataset; (has to be specified in Inputs.py as well)",
                        choices=["detop15", "eutop25", "worldtiny2k", "kaggle_dstl", "vaihingen",
                                 "detop15_nores", "eutop25_nores", "worldtiny2k_nores"])
    parser.add_argument("--max_runtime", type=float, default=5.0,
                        help=" maximum runtime in fraction hours")
    parser.add_argument("--patience", type=int, default=5,
                        help=" number of validations to wait for improvement until training stops")
    parser.add_argument("--not_restore_last", type=bool, default=False,
                        help=" If true last layer will not restored from checkpoint")
    parser.add_argument("--use_weights", type=str, default="False",
                        help=" If true median freq weights for the dataset will be used")
    return parser.parse_args()


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


def checkArgs():
    args = get_arguments()
    if args.local_config:
        args.test_dir = "./list/de_top15_r_test_list.txt"
        args.train_dir = "./list/de_top15_r_train_list.txt"
        args.val_dir = "./list/de_top15_r_val_list.txt"
        args.datadir = '../../ANN_DATA/de_top15_cropped'
        args.log_dir = "./snapshots"
        args.batch_size = 1
    if args.infere:
        print('The model is set to Inference')
        print("check point file: %s"%args.testing)
        print("inference file: %s"%args.test_dir)
    elif args.testing != '':
        print('The model is set to Testing')
        print("check point file: %s"%args.testing)
        print("testing file: %s"%args.test_dir)
    elif args.finetune != '':
        print('The model is set to Finetune from ckpt')
        print("check point file: %s"%args.finetune)
        print("Train file: %s"%args.train_dir)
        print("Val file: %s"%args.val_dir)
    else:
        print('The model is set to Training')
        print("Max training Iteration: %d"%args.max_steps)
        print("Initial lr: %f"%args.learning_rate)
        print("Train file: %s"%args.train_dir)
        print("Val file: %s"%args.val_dir)

    print("Batch Size: %d"%args.batch_size)
    print("Log dir: %s"%args.log_dir)
    print("Max runtime: %.2f"%args.max_runtime)
    return args


def main():
    print("SEGNET")
    args = checkArgs()
    print("SAVE TO "+args.log_dir)
    args.use_weights = str2bool(args.use_weights)
    if args.infere:
        model.infere(args)
    elif args.testing:
        model.test(args)
    elif args.finetune:
        model.training(args, is_finetune=True)
    else:
        model.training(args, is_finetune=False)


if __name__ == '__main__':
  main()
