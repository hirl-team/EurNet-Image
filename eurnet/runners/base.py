import os

from addict import Dict as adict
import torch
import torch.backends.cudnn as cudnn

from eurnet.utils import misc
import eurnet.utils.dist as dist_utils


class BaseRunner(object):
    def __init__(self, args: adict, resume=True) -> None:
        super().__init__()
        self.args = args
        dist_utils.init_distributed_mode(self.args)
        self.local_rank = args.local_rank
        self.rank = dist_utils.get_rank()
        misc.fix_random_seeds(self.args.seed)
        cudnn.benchmark = True
        
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(args.to_dict().items())))
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.custom_initialize()

        self.build_dataset(self.args)
        self.build_model(self.args)
        self.build_optimizer(self.args)
        self.build_scaler(self.args)
        if resume:
            self.resume(self.args)

    def custom_initialize(self):
        """
        we use this function to perform auto-resume from save dir.
        """

        os.makedirs(self.args.output_dir, exist_ok=True)
        if self.args.autoresume and os.path.exists(os.path.join(self.args.output_dir, "latest.pth")):
            self.args.resume = os.path.join(self.args.output_dir, "latest.pth")

    def custom_end(self):
        pass

    def build_dataset(self, args):
        raise NotImplementedError

    def build_model(self, args):
        raise NotImplementedError

    def build_optimizer(self, args):
        raise NotImplementedError

    def build_scaler(self, args):
        raise NotImplementedError

    def resume(self, args):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError

    def save_model(self, save_dict, epoch):
        torch.save(save_dict, os.path.join(self.args.output_dir, 'latest.pth'))
        if  (epoch % self.args.save_interval == 0) and epoch:
            torch.save(save_dict, os.path.join(self.args.output_dir, f'checkpoint_{epoch:04}.pth'))

    def restart_from_checkpoint(self, ckp_path, run_variables=None, **kwargs):
        """
        Re-start from checkpoint; 
        load the key into the value of kwargs
        kwargs in the format: 
            { key (str): value(nn.Module)}
        """
        if not os.path.isfile(ckp_path):
            return
        print("Found checkpoint at {}".format(ckp_path))

        # open checkpoint file
        checkpoint = torch.load(ckp_path, map_location="cpu")

        # key is what to look for in the checkpoint file
        # value is the object to load
        # example: {'state_dict': model}
        for key, value in kwargs.items():
            if key in checkpoint and value is not None:
                try:
                    msg = value.load_state_dict(checkpoint[key], strict=False)
                    print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
                except TypeError:
                    try:
                        msg = value.load_state_dict(checkpoint[key])
                        print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                    except ValueError:
                        print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
            else:
                print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

        # re load variable important for the run
        if run_variables is not None:
            for var_name in run_variables:
                if var_name in checkpoint:
                    run_variables[var_name] = checkpoint[var_name]
    
    def log_metrics(self, metric_logger, output_dict):
        pass
