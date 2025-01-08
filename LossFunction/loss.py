from typing import Union, Tuple
import numpy as np
from torch import distributions
from decode.simulation import psf_kernel
import torch
import decode
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
        S = torch.zeros([len(em_tar),self.img_shape[0],self.img_shape[1]]).to(self.device)
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