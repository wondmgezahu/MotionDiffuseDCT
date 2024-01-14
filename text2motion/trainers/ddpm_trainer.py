import torch
import torch.nn.functional as F
import random
import time
from models.transformer import MotionTransformer
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
from utils.utils import print_current_loss
from os.path import join as pjoin
import codecs as cs
import torch.distributed as dist
import numpy as np
from mmcv.runner import get_dist_info
from models.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)

from datasets import build_dataloader


class DDPMTrainer(object):

    def __init__(self, args, encoder):
        self.opt = args
        self.device = args.device
        self.encoder = encoder
        self.diffusion_steps = args.diffusion_steps
        sampler = 'uniform'
        beta_scheduler = 'linear'
        betas = get_named_beta_schedule(beta_scheduler, self.diffusion_steps)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE
        )
        self.sampler = create_named_schedule_sampler(sampler, self.diffusion)
        self.sampler_name = sampler

        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.to(self.device)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def util_mac(self,T):

        def get_dct_matrix(N, is_torch=True):
            dct_m = np.eye(N)
            for k in np.arange(N):
                for i in np.arange(N):
                    w = np.sqrt(2 / N)
                    if k == 0:
                        w = np.sqrt(1 / N)
                    dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
            idct_m = np.linalg.inv(dct_m)
            if is_torch:
                dct_m = torch.from_numpy(dct_m)
                idct_m = torch.from_numpy(idct_m)
            return dct_m, idct_m

        dct_m, idct_m = get_dct_matrix(T)
        dct_m_all = dct_m.float().to(self.device)
        idct_m_all = idct_m.float().to(self.device)

        return dct_m_all, idct_m_all

    def forward(self, batch_data, eval_mode=False):
        caption, motions, m_lens = batch_data
        motions = motions.detach().to(self.device).float()
        # breakpoint()
        self.caption = caption
        self.motions = motions
        x_start = motions
        # breakpoint()

        # breakpoint()


        B, T = x_start.shape[:2]
        print(x_start.shape)
        n_pre=40 # L elemnts of DCT
        # added dct
        dct_m_all,_=self.util_mac(T)
        x_start=torch.matmul(dct_m_all[:n_pre], x_start)
        # breakpoint()
        #end dct

        cur_len = torch.LongTensor([min(T, m_len) for m_len in m_lens]).to(self.device)
        t, _ = self.sampler.sample(B, x_start.device)
        # breakpoint()

        from calflops import calculate_flops
        # input_shape = (128, 196)  # define your input shape here
        # rgs=[x_start,t,cur_len,caption, ],
        flops, macss, paramss = calculate_flops(self.encoder,args = [x_start, t,cur_len,caption] ,include_backPropagation=True)
        print("FLOPs:%s  MACs:%s  Params:%s \n" %(flops, macss, paramss))
        breakpoint()

        output = self.diffusion.training_losses(
            model=self.encoder,
            x_start=x_start,
            t=t,
            model_kwargs={"text": caption, "length": cur_len}
        )
        # breakpoint()
        self.real_noise = output['target']
        self.fake_noise = output['pred']
        self.src_mask = torch.ones(B, n_pre).to(x_start.device)
        # breakpoint()
        # try:
        #     self.sr_mask = self.encoder.module.generate_src_mask(T, cur_len).to(x_start.device)
        # except:
        #     self.sr_mask = self.encoder.generate_src_mask(T, cur_len).to(x_start.device)

    def generate_batch(self, caption, m_lens, dim_pose):
        xf_proj, xf_out = self.encoder.encode_text(caption, self.device)

        B = len(caption)
        T = min(m_lens.max(), self.encoder.num_frames)

        # breakpoint()
        n_pre=40
        dct_m_all, idct_m_all = self.util_mac(T.cpu().numpy())
        # noise_dct=
        import time
        start_time=time.time()
        # breakpoint()

        output = self.diffusion.p_sample_loop(
            self.encoder,
            (B, T, dim_pose),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                'xf_proj': xf_proj,
                'xf_out': xf_out,
                'length': m_lens,
                'noise_dct':dct_m_all
            })
        
        # output = self.diffusion.ddim_sample_loop(
        #     self.encoder,
        #     (B, T, dim_pose),
        #     clip_denoised=False,
        #     progress=True,
        #     model_kwargs={
        #         'xf_proj': xf_proj,
        #         'xf_out': xf_out,
        #         'length': m_lens,
        #         'noise_dct':dct_m_all
        #     })
        
        # # breakpoint()
        # apply idct
        end_time = time.time()
        print('time it takes', end_time - start_time)
        output = torch.matmul(idct_m_all[:, :n_pre], output)
        # output = torch.matmul(idct_m_all, output)
        return output

    def generate(self, caption, m_lens, dim_pose, batch_size=1024):
        N = len(caption)
        cur_idx = 0
        # batch_size=4
        self.encoder.eval()

        all_output = []
        # breakpoint()
        while cur_idx < N:
            if cur_idx + batch_size >= N:
                batch_caption = caption[cur_idx:]
                batch_m_lens = m_lens[cur_idx:]
            else:
                batch_caption = caption[cur_idx: cur_idx + batch_size]
                batch_m_lens = m_lens[cur_idx: cur_idx + batch_size]
            output = self.generate_batch(batch_caption, batch_m_lens, dim_pose)
            B = output.shape[0]

            for i in range(B):
                all_output.append(output[i])
            cur_idx += batch_size
        # breakpoint()
        return all_output

    def backward_G(self):
        loss_mot_rec = self.mse_criterion(self.fake_noise, self.real_noise).mean(dim=-1)
        # breakpoint()
        loss_mot_rec = (loss_mot_rec * self.src_mask).sum() / self.src_mask.sum()
        self.loss_mot_rec = loss_mot_rec
        loss_logs = OrderedDict({})
        loss_logs['loss_mot_rec'] = self.loss_mot_rec.item()
        return loss_logs

    def update(self):
        self.zero_grad([self.opt_encoder])
        loss_logs = self.backward_G()
        self.loss_mot_rec.backward()
        self.clip_norm([self.encoder])
        self.step([self.opt_encoder])

        return loss_logs

    def to(self, device):
        if self.opt.is_train:
            self.mse_criterion.to(device)
        self.encoder = self.encoder.to(device)

    def train_mode(self):
        self.encoder.train()

    def eval_mode(self):
        self.encoder.eval()

    def save(self, file_name, ep, total_it):
        state = {
            'opt_encoder': self.opt_encoder.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        try:
            state['encoder'] = self.encoder.module.state_dict()
        except:
            state['encoder'] = self.encoder.state_dict()
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        if self.opt.is_train:
            self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
        self.encoder.load_state_dict(checkpoint['encoder'], strict=False) # True
        return checkpoint['ep'], checkpoint.get('total_it', 0)

    def train(self, train_dataset):
        rank, world_size = get_dist_info()
        self.to(self.device)
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.opt.lr)
        it = 0
        cur_epoch = 0
        
        
        
        # modufy the model loading
        # checkpoint = torch.load('/home/wondm/HumanMotion/MotionDiffuseMAC/text2motion/checkpoints/kit/kit_baseline_Mltgpu_8layers_1000/model/ckpt_e000.tar', map_location=self.device)
        # self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
        # self.encoder.load_state_dict(checkpoint['encoder'], strict=False)

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            cur_epoch, it = self.load(model_dir)

        start_time = time.time()

        train_loader = build_dataloader(
            train_dataset,
            samples_per_gpu=self.opt.batch_size,
            drop_last=True,
            workers_per_gpu=4,
            shuffle=True,
            dist=self.opt.distributed,
            num_gpus=len(self.opt.gpu_id))

        logs = OrderedDict()
        for epoch in range(cur_epoch, self.opt.num_epochs):
            self.train_mode()
            # breakpoint()
            for i, batch_data in enumerate(train_loader):
                # breakpoint()
                self.forward(batch_data)
                log_dict = self.update()
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v
                it += 1
                if it % self.opt.log_every == 0 and rank == 0:
                    mean_loss = OrderedDict({})
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, mean_loss, epoch, inner_iter=i)

                if it % self.opt.save_latest == 0 and rank == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if rank == 0:
                self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if epoch % self.opt.save_every_e == 0 and rank == 0:
                self.save(pjoin(self.opt.model_dir, 'ckpt_e%03d.tar' % (epoch)),
                          epoch, total_it=it)
