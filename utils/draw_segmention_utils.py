import numpy as np
import skimage


def draw_segmentation(image, mask):
    """
    对图片进行分割区域的标记
    :param image: 输出图片 RGB img [height, width, 3]
    :param mask: 分割区域[height, width, instance count]
    :return: 返回黑白图片，并且将分割区域保留原来的颜色
    """
    # 1、将彩色图片变成灰度图，并保留image以及同份灰色的图片
    # 注意从RGB到灰度图格式格式会从float32到unit8转换
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

    # 2、将彩色格式中mask部分保留其余部分都设置成灰色，
    # 判断[height,width, count]
    if mask.shape[-1] > 0:
        # 求出每个像素点需要保留和显示灰度图的判断
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        segmentation = np.where(mask, image, gray).astype(np.uint8)
    else:
        segmentation = gray.astype(np.uint8)
    return segmentation


def detect_and_draw_segmentation(args, model):
    """
    检测结果并画出分割区域
    :param args: 命令行参数
    :param model: 模型
    :return:
    """
    if not args.image or not args.video:
        raise ValueError("请提供要检测的图片或者视频路径之一")

    # 传入的图片
    if args.image:
        print("正在分割图片：{}".format(args.image))
        # 1、读取图片
        image = skimage.io.imread(args.image)
        # 2、模型检测返回结果
        r = model.detect([image], verbose=1)[0]
        # 3、画出分割区域
        segmentation = draw_segmentation(image, r['masks'])
        # 4、保存输出
        file_name = "./images/segment_{}".format(args.image.split("/")[-1])
        skimage.io.imsave(file_name, segmentation)

    if args.video:
        import cv2
        # 1、获取视频的读取
        vcapture = cv2.VideoCapture(args.video)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # 2、定义video writer后续写入
        file_name = "./images/segmentation_{}".format(args.video.split("/")[-1])
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                  fps, (width, height))

        # 3、循环获取每帧数据进行处理，完成之后写入本地文件
        count = 0
        success = True
        while success:
            print("帧数: ", count)
            # 读取图片
            success, image = vcapture.read()
            if success:
                # OpenCV 返回的BGR格式转换成RGB
                image = image[..., ::-1]
                # 模型检测mask
                r = model.detect([image], verbose=0)[0]
                # 画出区域
                segmentation = draw_segmentation(image, r['masks'])
                # RGB -> BGR
                segmentation = segmentation[..., ::-1]
                # 添加这张图到video writer
                vwriter.write(segmentation)
                count += 1
        vwriter.release()

    print("保存到检测结果到路径文件：", file_name)