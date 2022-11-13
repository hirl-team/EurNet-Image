import datetime
import json
import math
import os
from pathlib import Path
import sys
import time

from fvcore.nn import FlopCountAnalysis
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.utils import accuracy
import torch
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import InterpolationMode

from eurnet import datasets as eurnet_datasets
from eurnet import backbones
from eurnet.datasets import sampler
from eurnet.runners.base import BaseRunner
from eurnet.runners.base import BaseRunner
from eurnet.utils import misc
import eurnet.utils.dist as dist_utils
from eurnet.utils.misc import NativeScalerWithGradNormCount as NativeScaler


class ClassificationRunner(BaseRunner):
    """
    Runner for image classification.
    """
    filtered_keys = []

    def custom_end(self):
        if self.local_rank == 0:
            local_metric_file = os.path.join(self.args.output_dir, "metric.json")
            metric_dict = {"acc": self.metric}
            json_string = json.dumps(metric_dict)
            with open(local_metric_file, "w") as f:
                f.write(json_string)

    def build_transform(self, is_train, args):
        use_resize = args.input_size > 32
        if is_train:
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.auto_aug_policy,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                interpolation="bicubic",
            )
            if not use_resize:
                transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
            return transform

        t = []
        if use_resize and args.get("test_crop", True):
            size = int((256 / 224) * args.input_size)
            t.append(
                transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(args.input_size))
        else:
            t.append(
                transforms.Resize((args.input_size, args.input_size), interpolation=InterpolationMode.BICUBIC)
            )
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

        return transforms.Compose(t)

    def build_train_dataset(self, args, transform):
        if args.dataset == "imagenet":
            root = os.path.join(args.data_path, "train")
            dataset = datasets.ImageFolder(root, transform=transform)
        elif args.dataset == "imagenet22k":
            anno_file = "ILSVRC2011fall_whole_map_train.txt"
            dataset = eurnet_datasets.ImageNet22k(args.data_path, anno_file, transform=transform)
        else:
            raise NotImplementedError("Only ImageNet and ImageNet-22K are supported for image classification now.")

        return dataset

    def build_val_dataset(self, args, transform):
        if args.dataset == "imagenet":
            root = os.path.join(args.data_path, "val")
            dataset = datasets.ImageFolder(root, transform=transform)
        elif args.dataset == "imagenet22k":
            anno_file = "ILSVRC2011fall_whole_map_val.txt"
            dataset = eurnet_datasets.ImageNet22k(args.data_path, anno_file, transform=transform)
        else:
            raise NotImplementedError("Only ImageNet and ImageNet-22K are supported for image classification now.")

        return dataset

    def build_dataset(self, args):
        transform_train = self.build_transform(True, args)
        transform_val = self.build_transform(False, args)

        dataset_train = self.build_train_dataset(args, transform_train)
        dataset_val = self.build_val_dataset(args, transform_val)

        repeated_aug = args.get("repeated_aug", False)
        if repeated_aug:
            sampler_train = sampler.RASampler(
                dataset_train, num_replicas=dist_utils.get_world_size(), rank=dist_utils.get_rank(), shuffle=True)
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=dist_utils.get_world_size(), rank=dist_utils.get_rank(), shuffle=True)

        if args.dist_eval:
            if len(dataset_val) % dist_utils.get_world_size() != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=dist_utils.get_world_size(), rank=dist_utils.get_rank(), shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        loader_train = torch.utils.data.DataLoader(
                        dataset_train, sampler=sampler_train,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=True, drop_last=True,
                    )
        loader_val = torch.utils.data.DataLoader(
                    dataset_val, sampler=sampler_val,
                    batch_size=int(args.batch_size),
                    num_workers=args.num_workers,
                    pin_memory=True, drop_last=False
                )
        
        self.dataset_train = dataset_train
        self.dataset_val  = dataset_val
        self.loader_train = loader_train
        self.loader_val = loader_val

    def build_model(self, args):
        if args.arch == "eurnet":
            model_kwargs = args.get("model_kwargs", dict())
            model = backbones.EurNet(
                img_size=args.input_size,
                patch_size=args.patch_size,
                num_class=args.num_class, **model_kwargs
            )
        else:
            raise NotImplementedError

        print(model)
        input = torch.randn(1, 3, args.input_size, args.input_size)
        flops = FlopCountAnalysis(model, input).total()
        print("Number of GFLOPs: ", flops / 1e9)
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of million parameters: ", num_parameters / 1e6)

        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)
        model_without_ddp = model.module

        self.model = model
        self.model_without_ddp = model_without_ddp

        self.build_criterion(args)

    def build_criterion(self, args):
        """
        Determine criterion upon the usage of mixup.
        """
        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.num_class)

        if mixup_fn is not None:
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        print("Criterion: ", criterion)
        print("Mixup function: ", mixup_fn)
        self.criterion = criterion
        self.mixup_fn = mixup_fn

    def build_optimizer(self, args):
        # build optimizer
        accum_iter = args.get("accum_iter", 1)
        eff_batch_size = args.batch_size * accum_iter * dist_utils.get_world_size()
        self.args.lr = self.args.lr * eff_batch_size / 512.0
        self.args.start_lr = self.args.start_lr * eff_batch_size / 512.0
        self.args.min_lr = self.args.min_lr * eff_batch_size / 512.0

        print("Effective batch size: ", eff_batch_size)
        print("Actual lr: ", self.args.lr)
        print("Actual starting lr: ", self.args.start_lr)
        print("Actual minimum lr: ", self.args.min_lr)

        skip = self.model_without_ddp.no_weight_decay() \
            if hasattr(self.model_without_ddp, "no_weight_decay") else {}
        skip_keywords = self.model_without_ddp.no_weight_decay_keywords() \
            if hasattr(self.model_without_ddp, "no_weight_decay_keywords") else {}
        layer_decay = self.args.get("layer_decay", 1.0)
        if layer_decay == 1.0:
            parameters = misc.get_params_groups(self.model_without_ddp, skip, skip_keywords)
        else:
            num_layers = 12
            values = list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
            assigner = misc.LayerDecayValueAssigner(values)
            print("Assigned values = %s" % str(assigner.values))
            parameters = misc.get_parameter_groups_with_scale(self.model_without_ddp,
                                                              weight_decay=args.optimizer.params.weight_decay,
                                                              skip_list=skip, skip_keywords=skip_keywords,
                                                              get_num_layer=assigner.get_layer_id,
                                                              get_layer_scale=assigner.get_scale)

        optimizer_class = args.optimizer.name
        optimizer_params = args.optimizer.params
        optimizer = getattr(torch.optim, optimizer_class)(parameters, lr=self.args.lr, **optimizer_params)
        print("Optimizer: ", optimizer)
        self.optimizer = optimizer

        # build lr scheduler
        num_iter_per_epoch = len(self.loader_train) // accum_iter
        num_steps = int(self.args.epochs * num_iter_per_epoch)
        warmup_steps = int(self.args.warmup_epochs * num_iter_per_epoch)

        lr_scheduler = CosineLRScheduler(self.optimizer, t_initial=num_steps, lr_min=self.args.min_lr,
                                         warmup_lr_init=self.args.start_lr, warmup_t=warmup_steps, cycle_limit=1,
                                         t_in_epochs=False)
        print("LR Scheduler: ", lr_scheduler)
        self.lr_scheduler = lr_scheduler

    def build_scaler(self, args):
        """
        support for auto mix precision (amp)
        """
        use_fp16 = args.get("use_fp16", False)
        args.use_fp16 = use_fp16
        if args.use_fp16:
            print("auto mix precision enabled!")
        loss_scaler = NativeScaler(args.use_fp16)
        self.loss_scaler = loss_scaler

    def load_pretrain(self, args):
        print("Loading weight %s for fine-tuning ..." % (args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        state_dict = checkpoint["model"]

        # Delete relative_pos since we always re-init it
        relative_pos_keys = [k for k in state_dict.keys() if "relative_pos" in k]
        for k in relative_pos_keys:
            del state_dict[k]

        # Bicubic interpolate mismatched absolute_pos_embed if it is used
        absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
        for k in absolute_pos_embed_keys:
            absolute_pos_embed_pretrained = state_dict[k]
            absolute_pos_embed_current = self.model_without_ddp.state_dict()[k]
            _, L1, C1 = absolute_pos_embed_pretrained.size()
            _, L2, C2 = absolute_pos_embed_current.size()
            if C1 != C2:
                print("Error in loading %s, passing ..." % k)
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                    absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                        absolute_pos_embed_pretrained, size=(S2, S2), mode="bicubic")
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                    state_dict[k] = absolute_pos_embed_pretrained_resized

        if args.model_kwargs.get("head_init_scale", 1.0) !=1.0 :
            head_keys = [k for k in state_dict.keys() if "head" in k]
            for k in head_keys:
                del state_dict[k]
        else:
            # check classifier, if not match, then re-init classifier to zero
            head_bias_pretrained = state_dict['head.bias']
            Nc1 = head_bias_pretrained.shape[0]
            Nc2 = self.model_without_ddp.head.bias.shape[0]
            if (Nc1 != Nc2):
                if Nc1 == 21841 and Nc2 == 1000:
                    print("Loading ImageNet-22K weights to ImageNet-1K ...")
                    with open("./eurnet/datasets/map22kto1k.txt") as f:
                        map22kto1k = f.readlines()
                    map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
                    state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
                    state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
                else:
                    torch.nn.init.constant_(self.model_without_ddp.head.bias, 0.)
                    torch.nn.init.constant_(self.model_without_ddp.head.weight, 0.)
                    del state_dict['head.weight']
                    del state_dict['head.bias']
                    print("Error in loading classification head, re-init classification head to 0.")

        msg = self.model_without_ddp.load_state_dict(state_dict, strict=False)
        print("Loading message: ", msg)
        print("Loaded successfully from: ", args.pretrained)
        del checkpoint
        torch.cuda.empty_cache()

    def resume(self, args):
        if os.path.exists(args.pretrained):
            self.load_pretrain(args)

        to_restore = {"epoch": 0, "max_accuracy": 0.0}
        if isinstance(args.resume, str) and os.path.exists(args.resume):
            self.restart_from_checkpoint(args.resume, run_variables=to_restore, model=self.model_without_ddp,
                                         optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
                                         loss_scaler=self.loss_scaler)
        self.start_epoch = to_restore["epoch"]
        self.max_accuracy = to_restore["max_accuracy"]

    def set_epoch(self, epoch):
        self.loader_train.sampler.set_epoch(epoch)

    @property
    def custom_save_dict(self):
        return {
            'model': self.model_without_ddp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "loss_scaler": self.loss_scaler.state_dict(),
        }

    def run(self):
        """
        A standard running process for imagenet downstream tasks.
        """
        start_time = time.time()
        max_accuracy = self.max_accuracy if hasattr(self, "max_accuracy") else 0.0
        print(f"Start training for {self.args.epochs} epochs")
        self.model.train()
        for epoch in range(self.start_epoch, self.args.epochs):
            self.set_epoch(epoch)
            train_stats = self.train_one_epoch(epoch)
            val_stats = self.evaluate()
            if max_accuracy < val_stats["acc1"]:
                max_accuracy = val_stats["acc1"]
            print(f'Max accuracy: {max_accuracy:.2f}%')

            # save model
            save_dict = {
                'epoch': epoch + 1,
                'args': self.args.to_dict(),
                'max_accuracy': max_accuracy,
            }
            save_dict.update(self.custom_save_dict)
            self.save_model(save_dict, epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch}
            if dist_utils.is_main_process():
                with (Path(self.args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        self.metric = max_accuracy
        self.custom_end()

    def train_one_epoch(self, epoch):
        self.model.train(True)
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}/{}]'.format(epoch, self.args.epochs)
        log_interval = self.args.get("log_interval", 10)
        iter_per_epoch = self.args.get("iter_per_epoch", -1)
        iter_per_epoch = len(self.loader_train) if iter_per_epoch < 0 else iter_per_epoch

        accum_iter = self.args.get("accum_iter", 1)
        clip_grad = self.args.get("clip_grad", None)
        self.optimizer.zero_grad()

        for it, (samples, targets) in enumerate(metric_logger.log_every(self.loader_train, log_interval, header)):
            if it >= iter_per_epoch:
                break
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            if self.mixup_fn is not None:
                samples, targets = self.mixup_fn(samples, targets)

            with torch.cuda.amp.autocast(self.args.use_fp16):
                outputs = self.model(samples)
                loss = self.criterion(outputs, targets)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss = loss / accum_iter
            self.loss_scaler(loss, self.optimizer, parameters=self.model.parameters(),
                             update_grad=(it + 1) % accum_iter == 0, clip_grad=clip_grad)
            if (it + 1) % accum_iter == 0:
                self.optimizer.zero_grad()
                self.lr_scheduler.step_update((epoch * len(self.loader_train) + it) // accum_iter)

            torch.cuda.synchronize()
            metric_logger.update(loss=loss_value)
            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def evaluate(self):
        criterion = torch.nn.CrossEntropyLoss()

        metric_logger = misc.MetricLogger(delimiter="  ")
        header = 'Test:'
        # switch to evaluation mode
        self.model.eval()

        for batch in metric_logger.log_every(self.loader_val, 10, header):
            images, target = batch[0], batch[-1]
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(self.args.use_fp16):
                output = self.model(images)
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
