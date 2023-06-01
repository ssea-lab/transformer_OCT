import torch
import numpy as np
import math
import os
import sys
import shutil
import random
import torch.nn.functional as F
from PIL import Image
from PIL import ImageFile
from PIL import ImageFilter, ImageOps
from torchvision import transforms
from torch import nn
from timm.data import create_transform
from tensorboardX import SummaryWriter
from constants import OCT_DEFAULT_MEAN, OCT_DEFAULT_STD

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Class DataUpdater, used to update the statistic data such as loss, accuracy.
class DataUpdater(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


# Class Matrics, used to calculate top_k accuracy, sensitivity, specificity, etc.
class Matrics(object):
    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # Pytorch 1.7
                res.append(correct_k.mul_(100.0 / batch_size))

            return res


# define the dataAugmentation for supervised_train
class DataAugmentationForSupervisedTrain(object):
    def __init__(self, args):
        self.transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),  # 默认为224, 384, 448
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),  # shape is (3, H, W)
            transforms.Normalize(OCT_DEFAULT_MEAN,
                                 OCT_DEFAULT_STD)
        ])

    def __call__(self, image):
        return self.transform(image)  # shape is (3, H, W)

    def __repr__(self):
        repr = "(DataAugmentationForSupervisedTrain,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        return repr


# define the dataAugmentation for val_or_test
class DataAugmentationForValOrTest(object):
    def __init__(self, args):
        self.transform = transforms.Compose([
            transforms.Resize((args.image_size + 32, args.image_size + 32), interpolation=3),  # 默认为224, 384, 448
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),  # shape is (3, H, W)
            transforms.Normalize(OCT_DEFAULT_MEAN,
                                 OCT_DEFAULT_STD)
        ])

    def __call__(self, image):
        return self.transform(image)  # shape is (3, H, W)

    def __repr__(self):
        repr = "(DataAugmentationForValOrTest,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        return repr


# define the dataAugmentation for dino_ssl_train
class DataAugmentationForDinoSslTrain(object):
    def __init__(self, args):
        global_crops_scale = args.global_crops_scale
        local_crops_scale = args.local_crops_scale
        local_crops_number = args.local_crops_number
        global_img_size = args.global_img_size
        local_img_size = args.local_img_size
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(OCT_DEFAULT_MEAN, OCT_DEFAULT_STD),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(global_img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(global_img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_img_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops  # shape is list of tensor [(3, H, W)]

    def __repr__(self):
        pass


# define the dataAugmentation for dino_fine_tune_train
class DataAugmentationForDinoFineTuneTrain(object):
    def __init__(self, args):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(OCT_DEFAULT_MEAN, OCT_DEFAULT_STD),
        ])

    def __call__(self, image):
        return self.transform(image)


# define the dataAugmentation for mae_ssl_train
class DataAugmentationForMAESslTrain(object):
    def __init__(self, args):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(OCT_DEFAULT_MEAN),
                                 std=torch.tensor(OCT_DEFAULT_STD))
        ])

    def __call__(self, image):
        return self.transform(image)


# define the dataAugmentation for mae_fine_tune_train
class DataAugmentationForMAEFineTuneTrain(object):
    def __init__(self, args):
        self.transform = create_transform(
            input_size=args.image_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=OCT_DEFAULT_MEAN,
            std=OCT_DEFAULT_STD,
        )

    def __call__(self, image):
        return self.transform(image)

    def __repr__(self):
        pass


class DataAugmentationForMAEFineTuneTrainEnsemble(object):
    def __init__(self, args):
        self.transforms = [
            create_transform(
                input_size=args.image_size_list[i],
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=OCT_DEFAULT_MEAN,
                std=OCT_DEFAULT_STD,
            )
            for i in range(len(args.image_size_list))]

    def __call__(self, image):
        return [transform(image) for transform in self.transforms]

    def __repr__(self):
        pass


class DataAugmentationForMAEFineTuneTestEnsemble(object):
    def __init__(self, args):
        self.transforms = [
            transforms.Compose([
                transforms.Resize((args.image_size_list[i] + 32, args.image_size_list[i] + 32), interpolation=3),
                # 默认为224, 384, 448
                transforms.CenterCrop(args.image_size_list[i]),
                transforms.ToTensor(),  # shape is (3, H, W)
                transforms.Normalize(OCT_DEFAULT_MEAN,
                                     OCT_DEFAULT_STD)
            ])
            for i in range(len(args.image_size_list))
        ]

    def __call__(self, image):
        return [transform(image) for transform in self.transforms]

    def __repr__(self):
        pass


# Class ImageProcessor, used to normalize, get augmentation
class ImageProcessor(object):
    def __init__(self, args, augmentation='supervised_train'):
        self.augmentation = augmentation
        self.args = args

    def __call__(self, img):
        if self.augmentation == 'supervised_train':
            data_augmentation = DataAugmentationForSupervisedTrain(self.args)
        elif self.augmentation == 'val_or_test':
            if not self.args.ensemble:
                data_augmentation = DataAugmentationForValOrTest(self.args)
            else:
                data_augmentation = DataAugmentationForMAEFineTuneTestEnsemble(self.args)
        elif self.augmentation == 'dino_ssl_train':
            data_augmentation = DataAugmentationForDinoSslTrain(self.args)
        elif self.augmentation == 'dino_fine_tune_train':
            data_augmentation = DataAugmentationForDinoFineTuneTrain(self.args)
        elif self.augmentation == 'mae_ssl_train':
            data_augmentation = DataAugmentationForMAESslTrain(self.args)
        elif self.augmentation == 'mae_fine_tune_train':
            if not self.args.ensemble:
                data_augmentation = DataAugmentationForMAEFineTuneTrain(self.args)
            else:
                data_augmentation = DataAugmentationForMAEFineTuneTrainEnsemble(self.args)
        else:
            data_augmentation = None
            print('not support data augmentation type')
            sys.exit(-1)
        # 不同数据增强方式返回的图片数量不同，后面需要做适当的调整
        img = data_augmentation(img)
        return img


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        if hasattr(backbone, 'fc'):
            backbone.fc = nn.Identity()
        if hasattr(backbone, 'head'):
            backbone.head = nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # shape student_output:[ncrops*batch_size, dim]
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    print("Set warmup steps = %d" % warmup_iters)

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def create_directory(directory):
    """
    Create the directory.
    :param directory: str; the directory path we want to create.
    :return: None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def save_checkpoint_ssl(state, train_mode, model_name, is_best, filename='checkpoint.pth.tar'):
    filename = os.path.join("./checkpoint", train_mode, model_name, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join("./checkpoint", train_mode, model_name, 'model_best.pth.tar'))


def save_checkpoint(state, checkpoint_dir, is_best, filename='checkpoint.pth'):
    filename = os.path.join(checkpoint_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, 'model_best.pth'))


def get_mean_and_std(file='dataset/patch_train.txt'):
    image_list = []
    with open(file, 'r', encoding='utf8') as fin:
        for line in fin:
            name = line.split('\t')[0]
            a = Image.open(name).convert('RGB')
            a = transforms.Resize((500, 500))(a)
            b = transforms.ToTensor()(a)
            image_list.append(b)
    images = torch.stack(image_list, dim=0)  # [data_size, 3, H, W]
    for i in range(images.size(1)):
        ith_std_mean = torch.std_mean(images[:, i, :, :])
        # print(ith_std_mean)


def get_full_model_name(short_name, image_size):
    """
    this function is used to map the short model name to full short name
    :param short_name: short model name include ['Vit-T', 'Vit-S','Vit-B', 'Swin-T', 'Swin-S', 'Swin-B']
    :param image_size: image size
    :return: mapping dictionary from short model name to full model name
    """
    mapping_dict = {'Vit-T': 'deit_tiny_distilled_patch16_224',
                    'Vit-S': 'deit_small_distilled_patch16_224',
                    'Vit-B': ['deit_base_distilled_patch16_224', 'deit_base_distilled_patch16_384'],
                    'Swin-T': 'swin_tiny_patch4_window7_224',
                    'Swin-S': 'swin_small_patch4_window7_224',
                    'Swin-B': ['swin_base_patch4_window7_224_in22k', 'swin_large_patch4_window12_384_in22k'],
                    'vgg19': 'vgg19',
                    'resnet50': 'resnet50',
                    'resnet101': 'resnet101'
                    }
    if short_name == 'Vit-B' or short_name == 'Swin-B':
        if image_size == 224:
            return mapping_dict[short_name][0]
        else:
            return mapping_dict[short_name][1]
    else:
        return mapping_dict[short_name]


if __name__ == '__main__':
    pre_train_weight = 'mvit_imagenet_weight/MViTv2_B_in1k.pyth'
    mvit_checkpoint = torch.load(pre_train_weight, map_location='cpu')
    for key, value in mvit_checkpoint['model_state'].items():
        print(key)
