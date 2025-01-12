import os
from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import decode
import decode.utils
import decode.neuralfitter.train.live_engine
import torch
path = Path('')
import copy
from pathlib import Path
import decode.evaluation
import decode.neuralfitter
import decode.neuralfitter.coord_transform
import decode.neuralfitter.utils
import decode.simulation
from decode.neuralfitter.train.random_simulation import setup_random_simulation
from decode.neuralfitter.utils import log_train_val_progress
from decode.utils.checkpoint import CheckPoint

def setup_trainer(simulator_train, simulator_test, logger, model_out, ckpt_path, device, param):
    """Set model, optimiser, loss and schedulers"""
    models_available = {
        'SigmaMUNet': decode.neuralfitter.models.SigmaMUNet,
        'DoubleMUnet': decode.neuralfitter.models.model_param.DoubleMUnet,
        'SimpleSMLMNet': decode.neuralfitter.models.model_param.SimpleSMLMNet,
    }

    model = models_available[param.HyperParameter.architecture]
    # print("ch_in:%d" % param.HyperParameter.channels_in)
    model = model.parse(param)

    model_ls = decode.utils.model_io.LoadSaveModel(model,
                                                   output_file=model_out)

    model = model_ls.load_init()
    model = model.to(torch.device(device))

    # Small collection of optimisers
    """Checkpointing"""
    checkpoint = CheckPoint(path=ckpt_path)

    """Setup gradient modification"""
    grad_mod = param.HyperParameter.grad_mod

    """Log the model"""
    try:
        dummy = torch.rand((2, param.HyperParameter.channels_in,
                            *param.Simulation.img_size), requires_grad=False).to(
            torch.device(device))
        logger.add_graph(model, dummy)

    except:
        print("Did not log graph.")
        # raise RuntimeError("Your dummy input is wrong. Please update it.")

    """Transform input data, compute weight mask and target data"""
    frame_proc = decode.neuralfitter.scale_transform.AmplitudeRescale.parse(param)
    bg_frame_proc = None

    if param.HyperParameter.emitter_label_photon_min is not None:
        em_filter = decode.neuralfitter.em_filter.PhotonFilter(
            param.HyperParameter.emitter_label_photon_min)
    else:
        em_filter = decode.neuralfitter.em_filter.NoEmitterFilter()

    tar_frame_ix_train = (0, 0)
    tar_frame_ix_test = (0, param.TestSet.test_size)

    """Setup Target generator consisting possibly multiple steps in a transformation sequence."""
    tar_gen = decode.neuralfitter.utils.processing.TransformSequence(
        [
            decode.neuralfitter.target_generator.ParameterListTarget(
                n_max=param.HyperParameter.max_number_targets,
                xextent=param.Simulation.psf_extent[0],
                yextent=param.Simulation.psf_extent[1],
                ix_low=tar_frame_ix_train[0],
                ix_high=tar_frame_ix_train[1],
                squeeze_batch_dim=True),

            decode.neuralfitter.target_generator.DisableAttributes.parse(param),

            decode.neuralfitter.scale_transform.ParameterListRescale(
                phot_max=param.Scaling.phot_max,
                z_max=param.Scaling.z_max,
                bg_max=param.Scaling.bg_max)
        ])

    # setup target for test set in similar fashion, however test-set is static.
    tar_gen_test = copy.deepcopy(tar_gen)
    tar_gen_test.com[0].ix_low = tar_frame_ix_test[0]
    tar_gen_test.com[0].ix_high = tar_frame_ix_test[1]
    tar_gen_test.com[0].squeeze_batch_dim = False
    tar_gen_test.com[0].sanity_check()

    if param.Simulation.mode == 'acquisition':
        train_ds = decode.neuralfitter.dataset.SMLMLiveDataset(
            simulator=simulator_train,
            em_proc=em_filter,
            frame_proc=frame_proc,
            bg_frame_proc=bg_frame_proc,
            tar_gen=tar_gen, weight_gen=None,
            frame_window=param.HyperParameter.channels_in,
            pad=None, return_em=True)

        train_ds.sample(True)

    elif param.Simulation.mode == 'samples':
        train_ds = decode.neuralfitter.dataset.SMLMLiveSampleDataset(
            simulator=simulator_train,
            em_proc=em_filter,
            frame_proc=frame_proc,
            bg_frame_proc=bg_frame_proc,
            tar_gen=tar_gen,
            weight_gen=None,
            frame_window=param.HyperParameter.channels_in,
            return_em=False,
            ds_len=param.HyperParameter.pseudo_ds_size)

    test_ds = decode.neuralfitter.dataset.SMLMAPrioriDataset(
        simulator=simulator_test,
        em_proc=em_filter,
        frame_proc=frame_proc,
        bg_frame_proc=bg_frame_proc,
        tar_gen=tar_gen_test, weight_gen=None,
        frame_window=param.HyperParameter.channels_in,
        pad=None, return_em=True)

    test_ds.sample(True)

    """Set up post processor"""
    if param.PostProcessing is None:
        post_processor = decode.neuralfitter.post_processing.NoPostProcessing(xy_unit='px',
                                                                              px_size=param.Camera.px_size)

    elif param.PostProcessing == 'LookUp':
        post_processor = decode.neuralfitter.utils.processing.TransformSequence([

            decode.neuralfitter.scale_transform.InverseParamListRescale(
                phot_max=param.Scaling.phot_max,
                z_max=param.Scaling.z_max,
                bg_max=param.Scaling.bg_max),

            decode.neuralfitter.coord_transform.Offset2Coordinate.parse(param),

            decode.neuralfitter.post_processing.LookUpPostProcessing(
                raw_th=param.PostProcessingParam.raw_th,
                pphotxyzbg_mapping=[0, 1, 2, 3, 4, 9],
                xy_unit='px',
                px_size=param.Camera.px_size)
        ])

    elif param.PostProcessing in ('SpatialIntegration', 'NMS'):  # NMS as legacy support
        post_processor = decode.neuralfitter.utils.processing.TransformSequence([

            decode.neuralfitter.scale_transform.InverseParamListRescale(
                phot_max=param.Scaling.phot_max,
                z_max=param.Scaling.z_max,
                bg_max=param.Scaling.bg_max),

            decode.neuralfitter.coord_transform.Offset2Coordinate.parse(param),

            decode.neuralfitter.post_processing.SpatialIntegration(
                raw_th=param.PostProcessingParam.raw_th,
                xy_unit='px',
                px_size=param.Camera.px_size)
        ])

    else:
        raise NotImplementedError

    """Evaluation Specification"""
    matcher = decode.evaluation.match_emittersets.GreedyHungarianMatching.parse(param)

    return train_ds, test_ds, model, model_ls, grad_mod, post_processor, matcher, checkpoint

from typing import Union, Tuple
import numpy as np
from torch import distributions
from decode.simulation import psf_kernel
class LossFunc():
    def __init__(self, xextent: tuple, yextent: tuple, img_shape: tuple, device: Union[str, torch.device], psf):
        super().__init__()
        self._psf_loss = torch.nn.MSELoss(reduction='none')
        self._offset2coord = psf_kernel.DeltaPSF(xextent=xextent, yextent=yextent, img_shape=img_shape)
        self.device = device
        self._psf_img_gen = decode.simulation.Simulation(psf=psf)
        self.xextent = xextent
        self.yextent = xextent
        self.img_shape = img_shape

    def log(self, loss_val):
        return loss_val.mean().item(), {'gmm': loss_val[:, 0].mean().item(),
                                        'p': loss_val[:, 1].mean().item(),
                                        'bg': loss_val[:, 2].mean().item(),
                                        # 'img': loss_val[:, -1].mean().item()
                                        }

    def CELoss(self, P, em_tar,tar_mask) -> torch.Tensor:
        S = torch.zeros([len(em_tar),param.Simulation.img_size[0],param.Simulation.img_size[1]]).to(self.device)
        if tar_mask.sum():
            for i, tar in enumerate(em_tar):
                tar = tar.xyz_px.to(self.device)
                tar = torch.round(tar[:,[0,1]],decimals=0)
                tar = (tar.transpose(0,1)).int()
                tar = (tar[0],tar[1])
                S[i].index_put(tar, torch.ones(tar[0].size()).to(self.device))
        loss = 0
        loss += -(S * torch.log(P) + (1 - S) * torch.log(1 - P))
        loss = loss.sum(-1).sum(-1)
        return loss

    def Loss_Count(self, P, tar_mask):
        loss = 0
        prob_mean = P.sum(-1).sum(-1)
        prob_var = (P - P ** 2).sum(-1).sum(-1)
        loss += 1 / 2 * ((tar_mask.sum(-1) - prob_mean) ** 2) / prob_var + 1 / 2 * torch.log(2 * np.pi * prob_var)
        loss = loss * tar_mask.sum(-1)
        return loss

    def Loss_Loc(self, P, pxyz_mu, pxyz_sig, pxyz_tar, mask):
        batch_size = P.size(0)
        prob_normed = P / (P.sum(-1).sum(-1)[:, None, None])

        p_inds = tuple((P + 1).nonzero().transpose(1, 0))


        pxyz_mu = pxyz_mu[p_inds[0], :, p_inds[1], p_inds[2]]
        self._offset2coord._bin_ctr_x = self._offset2coord._bin_ctr_x.to(pxyz_mu.device)
        self._offset2coord._bin_ctr_y = self._offset2coord._bin_ctr_y.to(pxyz_mu.device)
        pxyz_mu[:, 1] = pxyz_mu[:, 1] + self._offset2coord.bin_ctr_x[p_inds[1]]
        pxyz_mu[:, 2] = pxyz_mu[:, 2] + self._offset2coord.bin_ctr_y[p_inds[2]]

        pxyz_mu = pxyz_mu.reshape(batch_size, 1, -1, 4)
        pxyz_sig = pxyz_sig[p_inds[0], :, p_inds[1], p_inds[2]].reshape(batch_size, 1, -1, 4)
        PXYZ = pxyz_tar.reshape(batch_size, -1, 1, 4).repeat_interleave(self.img_shape[0] * self.img_shape[1], 2)

        numerator = -1 / 2 * ((PXYZ - pxyz_mu) ** 2)
        denominator = (pxyz_sig ** 2)
        log_p_gauss_4d = (numerator / denominator).sum(3) - 1 / 2 * (torch.log(2 * np.pi * denominator[:, :, :, 0]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 1]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 2]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 3]))

        gauss_coef = prob_normed.reshape(batch_size, 1, self.img_shape[0] * self.img_shape[1])
        gauss_coef_logits = torch.log(gauss_coef)
        gauss_coef_logmax = torch.log_softmax(gauss_coef_logits, dim=2)
        gmm_log = torch.logsumexp(log_p_gauss_4d + gauss_coef_logmax, dim=2)

        return -(gmm_log * mask).sum(-1)

    def Loss_psf(self, psf_img, psf_gt):
        if psf_gt.dim() == 4:
            psf_gt = psf_gt[:, 2]
        loss = self._psf_loss(psf_img, psf_gt)
        loss = loss.sum(-1).sum(-1)
        return loss
    
    def get_psf_gt(self, em_tar):
        for i, em in enumerate(em_tar):
            tmp, _ = self._psf_img_gen.forward(em)
            tmp = tmp.to(self.device)
            if i==0:
                psf_gt = tmp
            else:
                psf_gt = torch.cat((psf_gt,tmp),dim=0)
        return psf_gt
    
    def norm(self, nobg):
        ret = []
        for tmp in nobg:
            tmp = (tmp - torch.min(tmp)) / (torch.max(tmp)-torch.min(tmp))
            ret.append(tmp)
        return torch.stack(ret,dim=0)

    def final_loss(self, output, target, nobg, em_tar=None):
        tar_param, tar_mask, tar_bg = target
        P = output[:, 0]
        pxyz_mu = output[:, 1:5]
        pxyz_sig = output[:, 5:9]
        bg_img = output[:, 9]
        # psf_img = output[:, -1]
        # nobg = self.norm(nobg)
        # psf_gt = self.get_psf_gt(em_tar)

        loss = torch.stack((self.Loss_Loc(P, pxyz_mu, pxyz_sig, tar_param, tar_mask),self.Loss_Count(P, tar_mask)),dim=1)
        loss = torch.cat((loss,self._psf_loss(bg_img, tar_bg).sum(-1).sum(-1).unsqueeze(1)),dim=1)
        # loss = torch.cat((loss,self.Loss_psf(psf_img, nobg).unsqueeze(1)*0.0001),dim=1)
        return loss

import torch
import time
from typing import Union

from tqdm import tqdm
from collections import namedtuple

from decode.neuralfitter.utils import log_train_val_progress
from decode.evaluation.utils import MetricMeter


def train(model, optimizer, loss, dataloader, grad_rescale, grad_mod, epoch, device, logger) -> float:

    model.train()
    tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.)  # progress bar enumeration
    t0 = time.time()
    loss_epoch = MetricMeter()
    
    for batch_num, (x, y_tar, weight, em_tar, nobg) in enumerate(tqdm_enum):  # model input (x), target (yt), weights (w)
        
        t_data = time.time() - t0

        x, y_tar, weight, nobg = ship_device([x, y_tar, weight, nobg], device)

        y_out = model(x)

        loss_val = loss.final_loss(y_out, y_tar, nobg, em_tar)
        # loss_val = loss(y_out, y_tar, weight)

        if grad_rescale:  # rescale gradients so that they are in the same order for the last layer
            weight, _, _ = model.rescale_last_layer_grad(loss_val, optimizer)
            loss_val = loss_val * weight

        optimizer.zero_grad()
        loss_val.mean().backward()

        if grad_mod:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.03, norm_type=2)

        optimizer.step()

        t_batch = time.time() - t0

        loss_mean, loss_cmp = loss.log(loss_val)  # compute individual loss components
        del loss_val
        loss_epoch.update(loss_mean)
        # tqdm_enum.set_description(f"E: {epoch} - t: {t_batch:.2} - t_dat: {t_data:.2} - L: {loss_mean:.3}")
        tqdm_enum.set_description(f"E: {epoch} - t: {t_batch:.2} - t_dat: {t_data:.2} - L: {loss_mean:.3} \
                                  Lgmm: {loss_cmp['gmm']:.3}, Lp: {loss_cmp['p']:.3}, Lbg: {loss_cmp['bg']:.3}")
        # tqdm_enum.set_description(f"E: {epoch} - t: {t_batch:.2} - t_dat: {t_data:.2} - L: {loss_mean:.3} \
        #                           Lgmm: {loss_cmp['gmm']:.3}, Lp: {loss_cmp['p']:.3}, Lbg: {loss_cmp['bg']:.3}, Limg: {loss_cmp['img']:.3},")
        # fig, axs = plt.subplots(1, 4)
        # axs[0].imshow(,cmap='gray')

        t0 = time.time()

    log_train_val_progress.log_train(loss_p_batch=loss_epoch.vals, loss_mean=loss_epoch.mean, logger=logger, step=epoch)

    return loss_epoch.mean

_val_return = namedtuple("network_output", ["loss", "x", "y_out", "y_tar", "weight", "em_tar"])

def test(model, loss, dataloader, epoch, device):

    x_ep, y_out_ep, y_tar_ep, weight_ep, em_tar_ep = [], [], [], [], []  # store things epoche wise (_ep)
    loss_cmp_ep = []

    model.eval()
    tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.)  # progress bar enumeration

    t0 = time.time()

    with torch.no_grad():
        for batch_num, (x, y_tar, weight, em_tar, nobg) in enumerate(tqdm_enum):

            x, y_tar, weight, nobg = ship_device([x, y_tar, weight, nobg], device)

            y_out = model(x)

            loss_val = loss.final_loss(y_out, y_tar, nobg, em_tar)

            # loss_val = loss(y_out, y_tar, weight)

            t_batch = time.time() - t0

            tqdm_enum.set_description(f"(Test) E: {epoch} - T: {t_batch:.2}")

            loss_cmp_ep.append(loss_val.detach().cpu())
            x_ep.append(x.cpu())
            y_out_ep.append(y_out.detach().cpu())

    loss_cmp_ep = torch.cat(loss_cmp_ep, 0)
    x_ep = torch.cat(x_ep, 0)
    y_out_ep = torch.cat(y_out_ep, 0)

    return loss_cmp_ep.mean(), _val_return(loss=loss_cmp_ep, x=x_ep, y_out=y_out_ep, y_tar=None, weight=None, em_tar=None)

def ship_device(x, device: Union[str, torch.device]):
    if x is None:
        return x

    elif isinstance(x, torch.Tensor):
        return x.to(device)

    elif isinstance(x, (tuple, list)):
        x = [ship_device(x_el, device) for x_el in x]  # a nice little recursion that worked at the first try
        return x

    elif device != 'cpu':
        raise NotImplementedError(f"Unsupported data type for shipping from host to CUDA device.")

if __name__ == '__main__' :
    from decode.utils import param_io
    param_file = 'network/loc/lsnr/lstm_network_9ch_3d_005/param_run.yaml'
    param = param_io.load_params(param_file)
    param.Meta.version = decode.utils.bookkeeping.decode_state()
    param = decode.utils.param_io.autoset_scaling(param)

    #@markdown > Set the path to the calibration file
    calibration_file = 'psfmod/spline_calibration_3dcal.mat' #@param {type:"string"}
    param.InOut.calibration_file = calibration_file

    #@markdown > Set the output directory(!), i.e. the folder in which you'll find the model during/after training. You may want to change this to a folder in your Google Drive, e.g. `gdrive/My Drive/[your_folder]`
    model_dir = './network/loc/lsnr/lstm_network_9ch_3d_005' #@param {type:"string"}

    #@markdown > Set the directory in which the checkpoints should be saved. This is useful if colab times out or crashes and you want to continue training. Unless you have reasons, you should use the same directory as for the model.
    ckpt_dir = './network/loc/lsnr/lstm_network_9ch_3d_005'  #@param {type:"string"}
    from_ckpt = False
    model_dir = Path(model_dir)

    if not model_dir.parents[0].is_dir():
        raise FileNotFoundError(f"The path to the directory of 'model_out' (and even its parent folder) could not be found.")
    else:
        if not model_dir.is_dir():
            model_dir.mkdir()
            print(f"Created directory, absolute path: {model_dir.resolve()}")

    model_out = Path(model_dir) / 'model.pt'
    ckpt_path = Path(ckpt_dir) /'ckpt.pt'
    
    param.InOut.experiment_out = str(model_dir)
    param.HyperParameter.batch_size = 24
    param.HyperParameter.channels_in = 9
    # param.Simulation.lifetime_avg = 5
    param_run_path = Path(model_out).parents[0] / 'param_run.yaml'
    param_io.save_params(param_run_path, param)
    import generic.random_simulation
    sim_train, sim_test = generic.random_simulation.setup_random_simulation(param)
    # sim_train, sim_test = decode.neuralfitter.train.live_engine.setup_random_simulation(param)
    simulator = sim_train
    from decode.neuralfitter.train import live_engine
    from decode.neuralfitter.utils import logger as logger_utils
    device = 'cuda'
    logger = [logger_utils.SummaryWriter(log_dir='logs', 
                                        filter_keys=["dx_red_mu", "dx_red_sig", 
                                                    "dy_red_mu", "dy_red_sig", 
                                                    "dz_red_mu", "dz_red_sig",
                                                    "dphot_red_mu", "dphot_red_sig",
                                                    "f1",
                                                    ]),
            logger_utils.DictLogger()]
    logger = logger_utils.MultiLogger(logger)
    ds_train, ds_test, model, model_ls, grad_mod, post_processor, matcher, ckpt = \
        setup_trainer(sim_train, sim_test, logger, model_out, ckpt_path, device, param)
    dl_train, dl_test = live_engine.setup_dataloader(param, ds_train, ds_test)



    import Choose_Device as Device
    import network.lstm_network_as_hsnr.Net.CNNLSTM as LS
    # model = IST(1, seq_len=param.HyperParameter.channels_in, initial_features=32, sigma_eps_default=0.2, model_dim=32, num_heads=4, depth=3).to(Device.device)
    model = LS.CNNBiLSTM(1, 10, seq_len=param.HyperParameter.channels_in, pad_convs=True, depth=2,initial_features=48, norm=None, norm_groups=None, sigma_eps_default=0.005).to(Device.device)

    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0006,weight_decay=0.1)
    psf = decode.utils.calibration_io.SMAPSplineCoefficient(
            calib_file=param.InOut.calibration_file).init_spline(
            xextent=param.Simulation.psf_extent[0],
            yextent=param.Simulation.psf_extent[1],
            img_shape=param.Simulation.img_size,
            device=param.Hardware.device_simulation,
            roi_size=param.Simulation.roi_size,
            roi_auto_center=param.Simulation.roi_auto_center
        )
    criterion = LossFunc(
        xextent=param.Simulation.psf_extent[0],
        yextent=param.Simulation.psf_extent[1],
        img_shape=param.Simulation.img_size,
        psf=psf,
        device=param.Hardware.device_simulation,
    )
    # criterion = decode.neuralfitter.loss.GaussianMMLoss(
    #     xextent=param.Simulation.psf_extent[0],
    #     yextent=param.Simulation.psf_extent[1],
    #     img_shape=param.Simulation.img_size,
    #     device=device,
    #     chweight_stat=param.HyperParameter.chweight_stat)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,gamma=0.9,step_size=10)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,cooldown=10,factor=0.5,mode='min',patience=20,threshold=0.0001,verbose=True)

    converges = False
    n = 0
    n_max = param.HyperParameter.auto_restart_param.num_restarts
    if from_ckpt:
        ckpt = decode.utils.checkpoint.CheckPoint.load(param.InOut.checkpoint_init)
        model.load_state_dict(ckpt.model_state)
        optimizer.load_state_dict(ckpt.optimizer_state)
        lr_scheduler.load_state_dict(ckpt.lr_sched_state)
        epoch0 = ckpt.step + 1
        model = model.train()
        print(f'Resuming training from checkpoint')
    else:
        epoch0 = 0 
    while not converges and n < n_max:
        n += 1
    
        conv_check = decode.neuralfitter.utils.progress.GMMHeuristicCheck(
            ref_epoch=1,
            emitter_avg=sim_train.em_sampler.em_avg,
            threshold=param.HyperParameter.auto_restart_param.restart_treshold,
        )
        
        for i in range(epoch0, param.HyperParameter.epochs):
            logger.add_scalar('learning/learning_rate', optimizer.param_groups[0]['lr'], i)
            
            if i >= 1:
                train_loss = train(
                    model=model,
                    optimizer=optimizer,
                    loss=criterion,
                    dataloader=dl_train,
                    grad_rescale=param.HyperParameter.moeller_gradient_rescale,
                    grad_mod=param.HyperParameter.grad_mod,
                    epoch=i,
                    device=torch.device(param.Hardware.device),
                    logger=logger
                )
            
            val_loss, test_out = test(model=model, loss=criterion, dataloader=dl_test,
                                                                        epoch=i,
                                                                        device=torch.device(param.Hardware.device))
            
            # if i >= 1:
            #   train_loss = decode.neuralfitter.train_val_impl.train(
            #       model=model,
            #       optimizer=optimizer,
            #       loss=criterion,
            #       dataloader=dl_train,
            #       grad_rescale=param.HyperParameter.moeller_gradient_rescale,
            #       grad_mod=param.HyperParameter.grad_mod,
            #       epoch=i,
            #       device=torch.device(param.Hardware.device),
            #       logger=logger
            #   )
            
            # val_loss, test_out = decode.neuralfitter.train_val_impl.test(model=model, loss=criterion, dataloader=dl_test,
            #                                                               epoch=i,
            #                                                               device=torch.device(param.Hardware.device))

            """Post-Process and Evaluate"""
            decode.neuralfitter.train.live_engine.log_train_val_progress.post_process_log_test(loss_cmp=test_out.loss, loss_scalar=val_loss,
                                                        x=test_out.x, y_out=test_out.y_out, y_tar=test_out.y_tar,
                                                        weight=test_out.weight, em_tar=ds_test.emitter,
                                                        px_border=-0.5, px_size=1.,
                                                        post_processor=post_processor, matcher=matcher, logger=logger,
                                                        step=i)


            if i >= 1:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step()

            model_ls.save(model, None)
            ckpt.dump(model.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict(),
                            log=logger.logger[1].log_dict, step=i)

            """Draw new samples Samples"""
            if param.Simulation.mode in 'acquisition':
                del ds_train._frames
                del ds_train._emitter
                del ds_train._bg_frames
                del ds_train._nobg_frames
                ds_train.sample(True)
            elif param.Simulation.mode != 'samples':
                raise ValueError
        break
    converges = True
    if converges:
        print("Training finished after reaching maximum number of epochs.")
    else:
        raise ValueError(f"Training aborted after {n_max} restarts. "
                        "You can try to reduce the learning rate by a factor of 2."
                        "\nIt is also possible that the simulated data is to challenging. "
                        "Check if your background and intensity values are correct "
                        "and possibly lower the average number of emitters.")