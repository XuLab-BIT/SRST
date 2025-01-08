import torch
from decode.generic.emitter import EmitterSet
from abc import ABC, abstractmethod
from typing import Tuple
import decode

class LooseEmitterSet:
    """
    Related to the standard EmitterSet. However, here we do not specify a frame_ix but rather a (non-integer)
    initial point in time where the emitter starts to blink and an on-time.

    Attributes:
        xyz (torch.Tensor): coordinates. Dimension: N x 3
        intensity (torch.Tensor): intensity, i.e. photon flux per time unit. Dimension N
        id (torch.Tensor, int): identity of the emitter. Dimension: N
        t0 (torch.Tensor, float): initial blink event. Dimension: N
        ontime (torch.Tensor): duration in frame-time units how long the emitter blinks. Dimension N
        xy_unit (string): unit of the coordinates
    """

    def __init__(self, xyz: torch.Tensor, t0: torch.Tensor, lifetime_dist, intensity_dist, intensity_th,
                 xy_unit: str, px_size, id: torch.Tensor = None, sanity_check=True, blink_time = None):
        """

        Args:
            xyz (torch.Tensor): coordinates. Dimension: N x 3
            intensity (torch.Tensor): intensity, i.e. photon flux per time unit. Dimension N
            t0 (torch.Tensor, float): initial blink event. Dimension: N
            ontime (torch.Tensor): duration in frame-time units how long the emitter blinks. Dimension N
            id (torch.Tensor, int, optional): identity of the emitter. Dimension: N
            xy_unit (string): unit of the coordinates
        """

        """If no ID specified, give them one."""
        if id is None:
            id = torch.arange(xyz.shape[0])

        self.intensity_th = intensity_th
        self.n = xyz.size()[0]
        self.xyz = xyz
        self.xy_unit = xy_unit
        self.px_size = px_size
        self._phot = None
        self.intensity_dist = intensity_dist
        self.id = id
        self.t0 = t0
        self.lifetime_dist = lifetime_dist
        self.blink_time = blink_time
        self.ontime = self.lifetime_dist.rsample((self.n,))
        self.intensity = torch.clamp(self.intensity_dist.sample((self.n,)), self.intensity_th)

        if sanity_check:
            self.sanity_check()

    def sanity_check(self):

        """Check IDs"""
        if self.id.unique().numel() != self.id.numel():
            raise ValueError("IDs are not unique.")

        """Check xyz"""
        if self.xyz.dim() != 2 or self.xyz.size(1) != 3:
            raise ValueError("Wrong xyz dimension.")

        """Check intensity"""
        if (self.intensity < 0).any():
            raise ValueError("Negative intensity values encountered.")

        """Check timings"""
        if (self.ontime < 0).any():
            raise ValueError("Negative ontime encountered.")

    @property
    def te(self):  # end time
        return self.t0 + self.ontime

    def _distribute_framewise(self):
        """
        Distributes the emitters framewise and prepares them for EmitterSet format.

        Returns:
            xyz_ (torch.Tensor): coordinates
            phot_ (torch.Tensor): photon count
            frame_ (torch.Tensor): frame indices (the actual distribution)
            id_ (torch.Tensor): identities

        """
        """Repeat by full-frame duration"""

        # kick out everything that has no full frame_duration

        for i in range(3):
            
            self.ontime = self.lifetime_dist.rsample((self.n,))
            self.intensity = torch.clamp(self.intensity_dist.sample((self.n,)), self.intensity_th)

            frame_start = torch.floor(self.t0).long()
            frame_last = torch.floor(self.te).long()
            frame_count = (frame_last - frame_start).long()

            frame_count_full = frame_count - 2
            ontime_first = torch.max(torch.min(self.te - self.t0, frame_start + 1 - self.t0), torch.ones_like(self.te))
            ontime_last = torch.max(torch.min(self.te - self.t0, self.te - frame_last), torch.ones_like(self.te))
            # ontime_first = torch.min(self.te - self.t0, frame_start + 1 - self.t0)
            # ontime_last = torch.min(self.te - self.t0, self.te - frame_last)

            blink_ix = self.blink_time >= i
            ix_full = frame_count_full >= 0
            ix_full = ix_full & blink_ix
            ix_full = ix_full.bool()
            xyz_ = self.xyz[ix_full, :]
            flux_ = self.intensity[ix_full]
            id_ = self.id[ix_full]
            frame_start_full = frame_start[ix_full]
            frame_dur_full_clean = frame_count_full[ix_full]

            xyz_ = xyz_.repeat_interleave(frame_dur_full_clean + 1, dim=0)
            phot_ = flux_.repeat_interleave(frame_dur_full_clean + 1, dim=0)  # because intensity * 1 = phot
            id_ = id_.repeat_interleave(frame_dur_full_clean + 1, dim=0)
            # because 0 is first occurence
            frame_ix_ = frame_start_full.repeat_interleave(frame_dur_full_clean + 1, dim=0) \
                        + decode.generic.utils.cum_count_per_group(id_) + 1

            """First frame"""
            # first
            xyz_ = torch.cat((xyz_, self.xyz[blink_ix]), 0)
            phot_ = torch.cat((phot_, self.intensity[blink_ix] * ontime_first[blink_ix]), 0)
            id_ = torch.cat((id_, self.id[blink_ix]), 0)
            frame_ix_ = torch.cat((frame_ix_, frame_start[blink_ix]), 0)

            # last (only if frame_last != frame_first
            ix_with_last = frame_last >= frame_start + 1
            ix_with_last = ix_with_last & blink_ix
            ix_with_last = ix_with_last.bool()
            xyz_ = torch.cat((xyz_, self.xyz[ix_with_last]))
            phot_ = torch.cat((phot_, self.intensity[ix_with_last] * ontime_last[ix_with_last]), 0)
            id_ = torch.cat((id_, self.id[ix_with_last]), 0)
            frame_ix_ = torch.cat((frame_ix_, frame_last[ix_with_last]))

            blink_time = self.lifetime_dist.rsample((self.n,))
            self.t0 = self.te + blink_time + 1

            if i == 0:
                re_xyz = xyz_
                re_phot = phot_
                re_id = id_
                re_frame_ix = frame_ix_
            else:
                re_xyz = torch.cat((re_xyz, xyz_))
                re_phot = torch.cat((re_phot, phot_), 0)
                re_id = torch.cat((re_id, id_), 0)
                re_frame_ix = torch.cat((re_frame_ix, frame_ix_), 0)

        return re_xyz, re_phot, re_frame_ix, re_id
    
    def return_emitterset(self):
        """
        Returns EmitterSet with distributed emitters. The ID is preserved such that localisations coming from the same
        fluorophore will have the same ID.

        Returns:
            EmitterSet
        """

        xyz_, phot_, frame_ix_, id_ = self._distribute_framewise()
        return EmitterSet(xyz_, phot_, frame_ix_.long(), id_.long(), xy_unit=self.xy_unit, px_size=self.px_size)


def at_least_one_dim(*args) -> None:
    """Make tensors at least one dimensional (inplace)"""
    for arg in args:
        if arg.dim() == 0:
            arg.unsqueeze_(0)


def same_shape_tensor(dim, *args) -> bool:
    """Test if tensors are of same size in a certain dimension."""
    for i in range(args.__len__() - 1):
        if args[i].size(dim) == args[i + 1].size(dim):
            continue
        else:
            return False

    return True


def same_dim_tensor(*args) -> bool:
    """Test if tensors are of same dimensionality"""
    for i in range(args.__len__() - 1):
        if args[i].dim() == args[i + 1].dim():
            continue
        else:
            return False

    return True