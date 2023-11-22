import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import zipfile
import io
import numpy as np
import random
import re
import pickle
from glob import glob

from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, \
    RandomHorizontalFlip, CenterCrop, TenCrop
from videotransforms.volume_transforms import ClipToTensor

"""Contains video frame paths and ground truth labels for a single split (e.g. train videos). """


class Split():
    def __init__(self):
        self.gt_a_list = []
        self.videos = []

    # 通过遍历整个数据集合，和split进行比较之后，填充完    self.gt_a_list = [] 以及 self.videos = [] 两个list
    def add_vid(self, paths, gt_a):
        self.videos.append(paths)  # paths对应某个类下的所有images的路径， 所以videos，是一个list的list
        self.gt_a_list.append(gt_a)  # gt_a某个类的下标索引,所以 gt_a_list 应该是 [ 1,1,1,1,1,1,1,1,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7]

    # label是动作类标号，idx为动作对应的视频标号
    def get_rand_vid(self, label, idx=-1):
        match_idxs = []
        for i in range(len(self.gt_a_list)):  # 视频序列标号 11111111666666667777777777
            if label == self.gt_a_list[i]:
                match_idxs.append(i)  # match_idxs [6,6,6,6,6,6]
        # match_idxs为动作标号为label的动作，对应的视频序号
        if idx != -1:
            return self.videos[match_idxs[idx]], match_idxs[idx]  # match_idxs[idx]表示 选出的动作label的视频中第idx个 ，
        # self.videos[match_idxs[idx]] 为这个视频
        random_idx = np.random.choice(match_idxs)  # 在某个类对应的视频中随机选择一个
        return self.videos[random_idx], random_idx

    # gt_a_list中值和label一致的数量
    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    # 不重复的类的集合
    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    # 这个方法似乎没有用
    def get_max_video_len(self):
        max_len = 0
        for v in self.videos:
            l = len(v)
            if l > max_len:
                max_len = l
        return max_len

    # 类的数量
    def __len__(self):
        return len(self.gt_a_list)


"""Dataset for few-shot videos, which returns few-shot tasks. """


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        # 计数器
        self.get_item_counter = 0
        # 要加载的数据集路径
        self.data_dir = args.path
        # 要加载的frame个数
        self.seq_len = args.seq_len
        # 是加载训练数据还是测试数据
        self.train = True
        self.tensor_transform = transforms.ToTensor()
        # Input image size to the CNN after cropping
        self.img_size = args.img_size
        # split-list配置文件，通过这个配置文件来在整个数据集中进行样本选择
        self.annotation_path = args.traintestlist
        # 类别数，每个类别中的样本数
        self.way = args.way
        self.shot = args.shot
        # 查询集中，每个类别的查询个数
        self.query_per_class = args.query_per_class
        # 保存选择到的train样本和test样本
        self.train_split = Split()
        self.test_split = Split()
        # 对样本进行transfrom的操作集合，将定义的操作存放在self.transform{}字典中
        self.setup_transforms()
        # load the paths of all videos in the train and test splits.
        self._select_fold()
        self.read_dir()

    """ return the current split being used """

    # self.train_test_lists 是从配置文件中读取的
    # 是遍历磁盘存放的video拆帧之后的数据集, split参数是数据集中视频的名字，对应一个目录，目录中是视频对应的拆帧
    # 根据对样本的遍历看其是属于train split list还是test split list,然后 返回对应的容器。
    # read_dir中调用self.get_train_or_test_db()目的是将数据存到容器中，其他地方调用此方法目的是取数据容器
    def get_train_or_test_db(self, split=None):
        if split is None:
            get_train_split = self.train
        else:
            # print("split:" + split)
            if split in self.train_test_lists["train"]:
                get_train_split = True
            elif split in self.train_test_lists["test"]:
                get_train_split = False
            else:
                return None
        if get_train_split:
            return self.train_split  # 返回一个容器
        else:
            return self.test_split  # 返回一个容器

    ##########################call the following function in __init__()########################################
    """Setup crop sizes/flips for augmentation during training and centre crop for testing"""

    def setup_transforms(self):
        video_transform_list = []
        video_test_list = []

        if self.img_size == 84:
            video_transform_list.append(Resize(96))
            video_test_list.append(Resize(96))
        elif self.img_size == 224:
            video_transform_list.append(Resize(256))
            video_test_list.append(Resize(256))
        else:
            print("img size transforms not setup")
            exit(1)
        video_transform_list.append(RandomHorizontalFlip())
        video_transform_list.append(RandomCrop(self.img_size))

        video_test_list.append(CenterCrop(self.img_size))

        self.transform = {}
        self.transform["train"] = Compose(video_transform_list)
        self.transform["test"] = Compose(video_test_list)

    """Loads all videos into RAM from an uncompressed zip. Necessary as the filesystem has a large block size, which is unsuitable for lots of images. """
    """Contains some legacy code for loading images directly, but this has not been used/tested for a while so might not work with the current codebase. """

    def read_dir(self):
        # load zipfile into memory  #data_dir:要加载的数据集路径
        if self.data_dir.endswith('.zip'):
            self.zip = True
            zip_fn = os.path.join(self.data_dir)
            self.mem = open(zip_fn, 'rb').read()
            self.zfile = zipfile.ZipFile(io.BytesIO(self.mem))
        else:
            self.zip = False

        # go through zip and populate splits with frame locations and action groundtruths
        if self.zip:
            dir_list = list(set([x for x in self.zfile.namelist() if '.jpg' not in x]))

            class_folders = list(set([x.split(os.sep)[-3] for x in dir_list if len(x.split(os.sep)) > 2]))
            class_folders.sort()
            self.class_folders = class_folders
            video_folders = list(set([x.split(os.sep)[-2] for x in dir_list if len(x.split(os.sep)) > 3]))
            video_folders.sort()
            self.video_folders = video_folders

            class_folders_indexes = {v: k for k, v in enumerate(self.class_folders)}
            video_folders_indexes = {v: k for k, v in enumerate(self.video_folders)}

            img_list = [x for x in self.zfile.namelist() if '.jpg' in x]
            img_list.sort()

            c = self.get_train_or_test_db(video_folders[0])

            last_video_folder = None
            last_video_class = -1
            insert_frames = []
            for img_path in img_list:

                class_folder, video_folder, jpg = img_path.split(os.sep)[-3:]

                if video_folder != last_video_folder:
                    if len(insert_frames) >= self.seq_len:
                        # 获得split（）单例，然后add_vid
                        c = self.get_train_or_test_db(last_video_folder.lower())
                        if c != None:
                            c.add_vid(insert_frames, last_video_class)
                        else:
                            pass
                    insert_frames = []
                    class_id = class_folders_indexes[class_folder]
                    vid_id = video_folders_indexes[video_folder]

                insert_frames.append(img_path)
                last_video_folder = video_folder
                last_video_class = class_id

            c = self.get_train_or_test_db(last_video_folder)
            if c != None and len(insert_frames) >= self.seq_len:
                c.add_vid(insert_frames, last_video_class)
        else:
            class_folders = os.listdir(self.data_dir)
            #如果把ssv2中的数字字符串
            if class_folders[0].isnumeric():
                class_folders.sort(key=lambda x: int(x))
            else:
                class_folders.sort()

            self.class_folders = class_folders
            for class_folder in class_folders:
                # print("class_folder:" + class_folder)
                # video_folders = os.listdir(os.path.join(self.data_dir, class_folder))
                path_temp = os.path.join(self.data_dir, class_folder).replace('\\', '/')
                video_folders = os.listdir(path_temp)
                video_folders.sort()
                if self.args.debug_loader:
                    video_folders = video_folders[0:1]
                # 遍历某个动作文件夹下面的 每个视频文件夹，用名字在split文件中进行检索，如果存在则将该视频文件夹归档，作为小样本任务的support或query set
                for video_folder in video_folders:
                    c = self.get_train_or_test_db(video_folder)
                    if c == None:
                        continue
                    imgs = os.listdir(os.path.join(self.data_dir, class_folder, video_folder))
                    if len(imgs) < self.seq_len:
                        continue
                    imgs.sort()
                    paths = [os.path.join(self.data_dir, class_folder, video_folder, img) for img in imgs]
                    paths.sort()
                    class_id = class_folders.index(class_folder)
                    c.add_vid(paths, class_id)  # 拿到某一个类别中 某一个video的所有帧图片 路径之后，调用c.add_vid进行帧选择。
        print("loaded {}".format(self.data_dir))
        print("train: {}, test: {}".format(len(self.train_split), len(self.test_split)))

    """ load the paths of all videos in the train and test splits. """

    def _select_fold(self):
        lists = {}
        for name in ["train", "test"]:
            fname = "{}list{:02d}.txt".format(name,
                                              self.args.split)  # 例如trainlist07, testlist07等等 self.args.split是03或者07这种
            f = os.path.join(self.annotation_path,
                             fname)  # self.annotation_path： 特定的split list的文件夹，fname： splitfile的文件名
            f = f.replace('\\', '/')
            selected_files = []
            with open(f, "r") as fid:
                data = fid.readlines()
                # data = [x.replace(' ', '_').lower() for x in data]  #weaving basket--> weaving_basket
                data = [x.replace(' ', '_') for x in data]  # weaving basket--> weaving_basket
                data = [x.strip().split(" ")[0] for x in data]
                data = [os.path.splitext(os.path.split(x)[1])[0] for x in data]
                # comment by guofei why???
                # if "kinetics" in self.args.path:
                #    data = [x[0:11] for x in data]

                selected_files.extend(data)
            lists[name] = selected_files
        self.train_test_lists = lists

    ############################################################################################################

    """ Set len to large number as we use lots of random tasks. Stopping point controlled in run.py. """

    def __len__(self):
        c = self.get_train_or_test_db()
        return 1000000
        return len(c)

    """ Get the classes used for the current split """

    def get_split_class_list(self):
        c = self.get_train_or_test_db()
        classes = list(set(c.gt_a_list))
        classes.sort()
        return classes

    """Loads a single image from a specified path """

    def read_single_image(self, path):
        if self.zip:
            with self.zfile.open(path, 'r') as f:
                with Image.open(f) as i:
                    i.load()
                    return i
        else:
            with Image.open(path) as i:
                i.load()
                return i

    """Gets a single video sequence. Handles sampling if there are more frames than specified. """

    def get_seq(self, label, idx=-1):
        c = self.get_train_or_test_db()
        # paths为动作为label视频序号为idx的视频对应的路径，该路径下是一堆的frames， vid_id是动作标签为label对应的一个动作视频的标号
        paths, vid_id = c.get_rand_vid(label, idx)
        n_frames = len(paths)
        if n_frames == self.args.seq_len:
            idxs = [int(f) for f in range(n_frames)]
        else:
            if self.train:
                excess_frames = n_frames - self.seq_len
                excess_pad = int(min(5, excess_frames / 2))
                if excess_pad < 1:
                    start = 0
                    end = n_frames - 1
                else:
                    start = random.randint(0, excess_pad)
                    end = random.randint(n_frames - 1 - excess_pad, n_frames - 1)
            else:
                start = 1
                end = n_frames - 2

            if end - start < self.seq_len:
                end = n_frames - 1
                start = 0
            else:
                pass

            idx_f = np.linspace(start, end, num=self.seq_len)
            idxs = [int(f) for f in idx_f]

            if self.seq_len == 1:
                idxs = [random.randint(start, end - 1)]

        imgs = [self.read_single_image(paths[i]) for i in idxs]
        if (self.transform is not None):
            if self.train:
                transform = self.transform["train"]
            else:
                transform = self.transform["test"]
            # 对imgs进行了transform 操作之后再转换为tensor
            imgs = [self.tensor_transform(v) for v in transform(imgs)]
            imgs = torch.stack(imgs)
        return imgs, vid_id  # vid_id是动作标签为label对应的一个随机动作  imgs对这个动作对应的抽取之后的长度为seq_len的帧图片s

    # __getitem__ 调用 read_single_image 调用 get_split_class_list
    """returns dict of support and target images and labels"""

    def __getitem__(self, index):

        # select classes to use for this task
        # c是一个split对象，其内容为gt_a_list, video。 gt_a_list:为一个序列，序列标号对应动作类别 ; video:为视频 每一个视频对应一个序列标号，video内容为frame数组
        c = self.get_train_or_test_db()
        classes = c.get_unique_classes()  # 其实就是self.gt_a_list 中 不重复元素的列表
        # 从class中选取 self.way个 例如 [4,8,12,21,23]
        batch_classes = random.sample(classes, self.way)

        if self.train:
            n_queries = self.args.query_per_class
        else:
            n_queries = self.args.query_per_class_test

        support_real_class=[]
        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []
        # batch_classes为选出来的support set  例如：batch_classes [4,8,12,21,23]
        for bl, bc in enumerate(batch_classes):
            # bl是类在batch_classes得下标，bc是类编号
            # select shots from the chosen classes
            # 计算出选出的类中的视频数量，gt_a_list中值和bc一致的数量，有几个就说明有几个视频的类为bc
            n_total = c.get_num_videos_for_class(bc)
            # 选出shot 和query
            idxs = random.sample([i for i in range(n_total)], self.args.shot + n_queries)  # 选出support 和query的样本编号
            # idx为选出来的视频标号，为某一个类下的标号，而vid_id是这个视频的整体的标号
            for idx in idxs[0:self.args.shot]:
                vid, vid_id = self.get_seq(bc, idx)  # bc为动作类标号，idx为动作对应的一个视频的标号， 输出 vid是选好的帧, vid_id是视频标号
                support_set.append(vid)
                support_labels.append(bl)  # support_labels 是类下标
                bc_index = classes.index(bc)
                real_support_labels.append(bc_index)
            for idx in idxs[self.args.shot:]:
                vid, vid_id = self.get_seq(bc, idx)
                target_set.append(vid)
                target_labels.append(bl)  # target_labels 是类下标
                bc_index =  classes.index(bc)
                real_target_labels.append(bc_index)  # real_target_labels 是类标号
        # support_set 的每个元素为8 3 224 224
        s = list(zip(support_set, support_labels, real_support_labels))
        random.shuffle(s)
        support_set, support_labels,real_support_labels = zip(*s)  # 返解压

        t = list(zip(target_set, target_labels, real_target_labels))
        random.shuffle(t)
        target_set, target_labels, real_target_labels = zip(*t)

        # 沿第一维压缩
        support_set = torch.cat(support_set)
        target_set = torch.cat(target_set)
        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        real_support_labels = torch.FloatTensor(real_support_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)
        batch_classes = torch.FloatTensor(batch_classes)

        return {"support_set": support_set, "support_labels": support_labels, "target_set": target_set,
                "target_labels": target_labels, "real_support_labels": real_support_labels, "real_target_labels": real_target_labels,
                "batch_class_list": batch_classes}


