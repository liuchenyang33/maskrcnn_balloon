# -*- coding:utf-8 -*-
import os
import json
import sys
import numpy as np
sys.path.append("../")
from mrcnn import utils, visualize
from mrcnn.config import Config
import skimage


class BalloonConfig(Config):
    """继承MaskRCNN的模型配置信息
    修改其中需要的训练集数据信息
    """
    # 给配置一个名称
    NAME = "balloon"

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    # 类别数量（包括背景），气球类别+1
    NUM_CLASSES = 1 + 1

    # 一个epoch的步数
    STEPS_PER_EPOCH = 10

    # 检测的时候过滤置信度的阈值
    DETECTION_MIN_CONFIDENCE = 0.9


class BalloonDataset(utils.Dataset):
    """气球分割数据集获取类
    """
    def load_balloon(self, dataset_dir, subset):
        """
        加载数据集
        :param dataset_dir: 数据集目录
        :param subset: 训练集还是测试机
        :return:
        """
        # 添加数据集类别数量
        self.add_class("balloon", 1, "balloon")

        # 是否提供在训练或者验证集字符串
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        # Load annotations
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        # }
        # 读取标注区域：
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())

        # 如果annotations不存在直接跳过
        annotations = [a for a in annotations if a['regions']]
        # 添加每张图片的坐标
        for a in annotations:
            # 获取所有多边形的x, y 的所有点坐标,存储在shape_attributes
            # 判断其中类型是否是字典，若果字典
            if isinstance(a['regions'], dict):
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # 读取图片内容获取长宽
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            # 加入到image_info字典当中
            self.add_image(
                "balloon",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """加载图片中的mask返回每个图片的mask及其id
        :param image_id: 图片ID
        :return: masks: 一个实例的布尔形状 [height, width, instance count]
        class_ids: 类别的 1D 数组
        """
        # 如果不是balloon类别的图片数据，默认返回空
        image_info = self.image_info[image_id]
        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        # 将坐标转换成bitmap [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            # 获取图片像素中的这个mask多边形区域中像素下标，将其标记为1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # 返回mask区域标记 [height, width, instance count]
        # 以及mask物体的个数
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


if __name__ == '__main__':
    dataset_train = BalloonDataset()
    dataset_train.load_balloon("../balloon_data/", "train")
    dataset_train.prepare()

    # 打印结果
    print("图片数量: {}".format(len(dataset_train.image_ids)))
    print("类别数量: {}".format(dataset_train.num_classes))
    for i, info in enumerate(dataset_train.class_info):
        print("{}. {}".format(i, info['name']))

    # 1、随机选择部分图片进行展示mask区域
    image_id = np.random.choice(dataset_train.image_ids, 1)[0]
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # 2、计算bbox
    bbox = utils.extract_bboxes(mask)

    from mrcnn.model import log
    log("img", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
    # 显示mask,以及bbox
    visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names)

    # 3、计算anchor结果
    config = BalloonConfig()
    config.BACKBONE_SHAPES = [[256, 256], [128, 128], [64, 64], [32, 32], [16, 16]]
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             config.BACKBONE_SHAPES,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # 打印anchor相关信息
    num_levels = len(config.BACKBONE_SHAPES)
    anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
    print("Count: ", anchors.shape[0])
    print("Scales: ", config.RPN_ANCHOR_SCALES)
    print("ratios: ", config.RPN_ANCHOR_RATIOS)
    print("Anchors per Cell: ", anchors_per_cell)
    print("Levels: ", num_levels)
    anchors_per_level = []
    for l in range(num_levels):
        num_cells = config.BACKBONE_SHAPES[l][0] * config.BACKBONE_SHAPES[l][1]
        anchors_per_level.append(anchors_per_cell * num_cells // config.RPN_ANCHOR_STRIDE ** 2)
        print("Anchors in Level {}: {}".format(l, anchors_per_level[l]))

    # 4、anchor到rois感兴趣区域
    from mrcnn import model
    random_rois = 2000
    # 获取4个数据测试看结果
    g = model.DataGenerator(dataset_train, config,
                            shuffle=True,
                            random_rois=random_rois,
                            detection_targets=True)
    # 针对数据集的GT计算得到rpn的预测框以及mrcnn的输出预测框

    [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, rpn_rois, rois], \
    [mrcnn_class_ids, mrcnn_bbox, mrcnn_mask] = g.__getitem__(0)

    # 打印rois以及mrcnn
    log("rois", rois)
    log("mrcnn_class_ids", mrcnn_class_ids)
    log("mrcnn_bbox", mrcnn_bbox)
    log("mrcnn_mask", mrcnn_mask)

    # 打印GT结果
    log("gt_class_ids", gt_class_ids)
    log("gt_boxes", gt_boxes)
    log("gt_masks", gt_masks)
    log("rpn_match", rpn_match, )
    log("rpn_bbox", rpn_bbox)
    image_id = image_meta[0][0]
    print("image_id: ", image_id)

    # 5、对于其中一张图片进行anchor的refine
    # 获取正负样本匹配结果
    b = 0
    positive_anchor_ids = np.where(rpn_match[b] == 1)[0]
    print("Positive anchors: {}".format(len(positive_anchor_ids)))
    negative_anchor_ids = np.where(rpn_match[b] == -1)[0]
    print("Negative anchors: {}".format(len(negative_anchor_ids)))
    neutral_anchor_ids = np.where(rpn_match[b] == 0)[0]
    print("Neutral anchors: {}".format(len(neutral_anchor_ids)))

    # 对于标记为正样本anchor进行位置refine计算
    indices = np.where(rpn_match[b] == 1)[0]
    refined_anchors = utils.apply_box_deltas(anchors[indices], rpn_bbox[b, :len(indices)] * config.RPN_BBOX_STD_DEV)
    log("anchors", anchors)
    log("refined_anchors", refined_anchors)

    # 获取其中默认第一张图片的数据，打印正样本标记结果和负样本标记结果
    sample_image = model.unmold_image(normalized_images[b], config)
    # ROI的类别数量
    for c, n in zip(dataset_train.class_names, np.bincount(mrcnn_class_ids[b].flatten())):
        if n:
            print("{:23}: {}".format(c[:20], n))

    # 展示正样本输出结果
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, figsize=(16, 16))
    visualize.draw_boxes(sample_image, boxes=anchors[positive_anchor_ids],
                         refined_boxes=refined_anchors, ax=ax)
    # 展示负样本输出
    visualize.draw_boxes(sample_image, boxes=anchors[negative_anchor_ids])

    # 6、正负rois区域比例
    print("Positive ROIs: ", mrcnn_class_ids[b][mrcnn_class_ids[b] > 0].shape[0])
    print("Negative ROIs: ", mrcnn_class_ids[b][mrcnn_class_ids[b] == 0].shape[0])
    print("Positive Ratio: {:.2f}".format(
        mrcnn_class_ids[b][mrcnn_class_ids[b] > 0].shape[0] / mrcnn_class_ids[b].shape[0]))