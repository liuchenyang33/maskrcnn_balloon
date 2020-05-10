import numpy as np
import skimage.draw
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from mrcnn import model as maskrcnn
from utils.balloon_dataset import BalloonDataset, BalloonConfig
from utils.draw_segmention_utils import detect_and_draw_segmentation


# 命令行参数
parser = argparse.ArgumentParser(
    description='气球分割模型maskrcnn训练')
parser.add_argument("--command", type=str, default='train',
                    help="'train' or 'test' 训练还是进行测试")
parser.add_argument('--dataset', type=str, default='./balloon_data',
                    help='气球分割数据集目录')
parser.add_argument('--weights', type=str, default='imagenet',
                    help="预训练模型权重"
                         "imagenet:https://github.com/fchollet/"
                         "deep-learning-models/releases/"
                         "download/v0.2/resnet50_weights"
                         "_tf_dim_ordering_tf_kernels_notop.h5")
parser.add_argument('--logs', type=str, default='./logs/',
                    help='打印日志目录')
parser.add_argument('--img', type=str, default='./images/2917282960_06beee649a_b.jpg',
                    help='需要进行检测分割的图片目录')
parser.add_argument('--video', type=str, default='./images/v0200fd10000bq043q9pskdh7ri20vm0.MP4',
                    help='需要进行检测分割的视频目录')
parser.add_argument('--model', type=str, default='./logs/mask_rcnn_balloon.h5',
                    help='指定测试使用的训练好的模型文件')


def train(model):
    """训练模型逻辑
    :param model: maskrcnn模型
    :return:
    """
    # 1、获取分割数据集
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(args.dataset, "train")
    dataset_train.prepare()

    # 2、获取分割验证数据集
    dataset_val = BalloonDataset()
    dataset_val.load_balloon(args.dataset, "val")
    dataset_val.prepare()

    # 3、开始训练
    # print("go to train：")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='heads')


if __name__ == '__main__':

    args = parser.parse_args()

    # 1、进行参数传入判断
    if args.command == "train":
        assert args.dataset, "指定训练的时候必须传入 --dataset数据目录"
    elif args.command == "test":
        assert args.image or args.video,\
               "指定测试的时候必须提供图片或者视频"

    # 2、配置模型的参数、数据集的训练读取配置
    if args.command == "train":
        class InferenceConfig(BalloonConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 2
        config = InferenceConfig()
    else:
        # 测试的配置修改：设置batch_size为1，Batch size = GPU_COUNT * IMAGES_PER_GPU
        class InferenceConfig(BalloonConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # 3、创建模型
    if args.command == "train":
        model = maskrcnn.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = maskrcnn.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
    # 4、训练测试逻辑实现
    if args.command == "train":
        # 选择加载的预训练模型类别并下载
        if args.weights.lower() == "imagenet":
            weights_path = model.get_imagenet_weights()
        else:
            raise ValueError("提供一种预训练模型种类")

        # 加载预训练模型权重
        print("Loading weights ", weights_path)
        model.load_weights(weights_path, by_name=True)

        # 进行训练
        train(model)

    elif args.command == "test":
        model.load_weights(args.model, by_name=True)
        # 进行检测
        detect_and_draw_segmentation(args, model)
    else:
        # print("'{}' 传入参数无法识别. "
        #       "请使用 'train' or 'test'".format(args.command))
        pass