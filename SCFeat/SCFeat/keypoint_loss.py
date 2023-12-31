import torch
import torch.nn as nn
import torch.nn.functional as F 
from .preprocess_utils import *
from torch.distributions import Categorical, Bernoulli

class KeypointLoss(nn.Module):
    '''
        Keypoints loss
    '''
    def __init__(self, configs, device=None):
        super(KeypointLoss, self).__init__()
        self.__lossname__ = 'KeypointLoss'
        
        self.config = configs

        self.unfold_size    = self.config['grid_size']
        self.t_base         = self.config['temperature_base']
        self.t_max          = self.config['temperature_max']
        # self.reward         = getattr(self, self.config['epipolar_reward'])
        self.good_reward    = self.config['good_reward']
        self.bad_reward     = self.config['bad_reward']
        self.kp_penalty     = self.config['kp_penalty']

    def point_distribution(self, logits):
        proposal_dist = Categorical(logits=logits) # bx1x(h//g)x(w//g)x(g*g)
        proposals     = proposal_dist.sample() # bx1x(h//g)x(w//g)
        proposal_logp = proposal_dist.log_prob(proposals) # bx1x(h//g)x(w//g)

        # accept_logits = select_on_last(logits, proposals).squeeze(-1)
        accept_logits = torch.gather(logits, dim=-1, index=proposals[..., None]).squeeze(-1) # bx1x(h//g)x(w//g)

        accept_dist    = Bernoulli(logits=accept_logits)
        accept_samples = accept_dist.sample() # bx1x(h//g)x(w//g)
        accept_logp    = accept_dist.log_prob(accept_samples) # for accepted points, equals to sigmoid() then log(); for denied, (1-sigmoid).log
        accept_mask    = accept_samples == 1.

        logp = proposal_logp + accept_logp

        return proposals, accept_mask, logp

    def point_sample(self, kp_map):
        kpmap_unfold = unfold(kp_map, self.unfold_size)
        proposals, accept_mask, logp = self.point_distribution(kpmap_unfold)

        b, _, h, w = kp_map.shape
        grids_org = gen_grid(h_min=0, h_max=h-1, w_min=0, w_max=w-1, len_h=h, len_w=w)
        grids_org = grids_org.reshape(h, w, 2)[None, :, :, :].repeat(b, 1, 1, 1).to(kp_map)
        grids_org = grids_org.permute(0,3,1,2) # bx2xhxw
        grids_unfold = unfold(grids_org, self.unfold_size) # bx2x(h//g)x(w//g)x(g*g)

        kps = grids_unfold.gather(dim=4, index=proposals.unsqueeze(-1).repeat(1,2,1,1,1))
        return kps.squeeze(4).permute(0,2,3,1), logp, accept_mask
    
    def random_choice_idx(self, len, size):
        rand = np.random.RandomState(1234)
        if len > size:
            idx = rand.choice(len, size, replace=False)
        else:
            idx = rand.choice(len, size, replace=True)

        return idx

    def compute_fundamental(self, x1, x2):
        '''
        Estimating fundamental matrix
        '''
        x1 = x1.detach().cpu().numpy()
        x2 = x2.detach().cpu().numpy()
        n = x1.shape[1]
        if x2.shape[1] != n:
            raise ValueError("Number of points don't match.")
        A = np.zeros((n, 9))
        for i in range(n):
            A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
                    x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
                    x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i]]
        U, S, V = np.linalg.svd(A)
        F = V[-1].reshape(3, 3)
        U, S, V = np.linalg.svd(F)
        S[2] = 0
        F = np.dot(U, np.dot(np.diag(S), V))

        return F / F[2, 2]

    def get_fundamental_loss(self, F_gt, good, coord1_ori, coord2_ori, iters=100):
        '''
        Estimating fundamental matrix error
        '''
        B, _, _ = good.shape
        assert B == 1
        gi = good.int() # toInt()
        ngi = torch.nonzero(gi) # nonZero()
        if len(ngi) < 8: # assert len > 8
            return None
        loss = [] # initialize fundamental error vector
        for i in range(iters):
            
            idx = self.random_choice_idx(len(ngi)-1, 8) # select 8 indices randomly
            x1 = torch.zeros(8, 2)
            x2 = torch.zeros(8, 2)
            k = 0
            for id in idx:
                x1[k] = coord1_ori[0,ngi[id][1]]
                x2[k] = coord2_ori[0,ngi[id][2]]
                k = k + 1
            x1 = homogenize(x1).transpose(0,1)
            x2 = homogenize(x2).transpose(0,1)
            F_e = torch.Tensor(self.compute_fundamental(x1, x2)).cuda()
            tmp_loss = F.smooth_l1_loss(F_e, F_gt) # Smooth L1 Loss
        
            loss.append(tmp_loss)

        try:
            loss = torch.min(torch.Tensor(loss).cuda()) # minimize
        except:
            loss = None
        return loss

    @ torch.no_grad()
    def my_reward(self, inputs, outputs, coord1, coord2, reward_thr, rescale_thr):
        '''
        New reward with fundamental matrix error
        '''
        coord1_h = homogenize(coord1).transpose(1, 2) #bx3xm
        coord2_h = homogenize(coord2).transpose(1, 2) #bx3xn
        fmatrix = inputs['F1']
        fmatrix2 = inputs['F2']

        # compute the distance of the points in the second image
        epipolar_line = fmatrix.bmm(coord1_h) # Bx3xn
        epipolar_line_ = epipolar_line / torch.clamp(
            torch.norm(epipolar_line[:, :2, :], p=2, dim=1, keepdim=True), min=1e-8)
        epipolar_dist = torch.abs(epipolar_line_.transpose(1, 2)@coord2_h) #bxmxn
       
        # compute the distance of the points in the first image
        epipolar_line2 = fmatrix2.bmm(coord2_h)
        epipolar_line2_ = epipolar_line2 / torch.clamp(
            torch.norm(epipolar_line2[:, :2, :], p=2, dim=1, keepdim=True), min=1e-8)
        epipolar_dist2 = torch.abs(epipolar_line2_.transpose(1, 2)@coord1_h) #bxnxm
        epipolar_dist2 = epipolar_dist2.transpose(1,2) #bxmxn

        if rescale_thr:
            b, _, _ = epipolar_dist.shape
            dist1 = epipolar_dist.detach().reshape(b, -1).mean(1,True)
            dist2 = epipolar_dist2.detach().reshape(b,-1).mean(1,True)
            dist_ = torch.cat([dist1, dist2], dim=1)
            scale1 = dist1/dist_.min(1,True)[0].clamp(1e-6)
            scale2 = dist2/dist_.min(1,True)[0].clamp(1e-6)
            thr1 = reward_thr*scale1
            thr2 = reward_thr*scale2
            thr1 = thr1.reshape(b,1,1)
            thr2 = thr2.reshape(b,1,1)
        else:
            thr1 = reward_thr
            thr2 = reward_thr
            scale1 = epipolar_dist2.new_tensor(1.) 
            scale2 = epipolar_dist2.new_tensor(1.)
        
        good = (epipolar_dist<thr1) & (epipolar_dist2<thr2)
        b,m,n = good.shape
        fundamental_loss = None
        for i in range(b):
            tmp_loss = self.get_fundamental_loss(inputs['F2'][i,:,:].unsqueeze(0), good[i,:,:].unsqueeze(0), coord1[i,:,:].unsqueeze(0), coord2[i,:,:].unsqueeze(0))
            if fundamental_loss is None:
                fundamental_loss = tmp_loss
            elif tmp_loss is not None:
                fundamental_loss += tmp_loss

        reward = None
        if fundamental_loss is None:
            reward = self.good_reward*good + self.bad_reward*(~good)
        else:
            fundamental_loss = F.tanh(fundamental_loss)
            reward = (1 - fundamental_loss)*good - fundamental_loss*(~good)

        return reward, scale1, scale2, fundamental_loss

    @ torch.no_grad()
    def posfeat_reward(self, inputs, outputs, coord1, coord2, reward_thr, rescale_thr):
        '''
        PoSFeat reward
        '''
        coord1_h = homogenize(coord1).transpose(1, 2) #bx3xm
        coord2_h = homogenize(coord2).transpose(1, 2) #bx3xn
        fmatrix = inputs['F1']
        fmatrix2 = inputs['F2']

        # compute the distance of the points in the second image
        epipolar_line = fmatrix.bmm(coord1_h)
        epipolar_line_ = epipolar_line / torch.clamp(
            torch.norm(epipolar_line[:, :2, :], p=2, dim=1, keepdim=True), min=1e-8)
        epipolar_dist = torch.abs(epipolar_line_.transpose(1, 2)@coord2_h) #bxmxn

        # compute the distance of the points in the first image
        epipolar_line2 = fmatrix2.bmm(coord2_h)
        epipolar_line2_ = epipolar_line2 / torch.clamp(
            torch.norm(epipolar_line2[:, :2, :], p=2, dim=1, keepdim=True), min=1e-8)
        epipolar_dist2 = torch.abs(epipolar_line2_.transpose(1, 2)@coord1_h) #bxnxm
        epipolar_dist2 = epipolar_dist2.transpose(1,2) #bxmxn

        if rescale_thr:
            b, _, _ = epipolar_dist.shape
            dist1 = epipolar_dist.detach().reshape(b, -1).mean(1,True)
            dist2 = epipolar_dist2.detach().reshape(b,-1).mean(1,True)
            dist_ = torch.cat([dist1, dist2], dim=1)
            scale1 = dist1/dist_.min(1,True)[0].clamp(1e-6)
            scale2 = dist2/dist_.min(1,True)[0].clamp(1e-6)
            thr1 = reward_thr*scale1
            thr2 = reward_thr*scale2
            thr1 = thr1.reshape(b,1,1)
            thr2 = thr2.reshape(b,1,1)
        else:
            thr1 = reward_thr
            thr2 = reward_thr
            scale1 = epipolar_dist2.new_tensor(1.) 
            scale2 = epipolar_dist2.new_tensor(1.) 

        good = (epipolar_dist<thr1) & (epipolar_dist2<thr2)
        reward = self.good_reward*good + self.bad_reward*(~good)
        return reward, scale1, scale2

    def forward(self, inputs, outputs, processed):
        preds1 = outputs['preds1']
        preds2 = outputs['preds2']

        kp_map1, kp_map2 = preds1['local_point'], preds2['local_point']
        
        xf1, xf2 = preds1['local_map'], preds2['local_map']
        
        b,c,h4,w4 = xf1.shape
        _, _, h, w = kp_map1.shape
        
        temperature = min(self.t_base + outputs['epoch'], self.t_max)
        
        coord1, logp1, accept_mask1 = self.point_sample(kp_map1) # bx(h//g)x(w//g)x2 bx1x(h//g)x(w//g) bx1x(h//g)x(w//g)
        coord2, logp2, accept_mask2 = self.point_sample(kp_map2)
        coord1 = coord1.reshape(b,-1,2)
        coord2 = coord2.reshape(b,-1,2)

        coord1_n = normalize_coords(coord1, h, w) # bx((h//g)*(w//g))x2
        coord2_n = normalize_coords(coord2, h, w)

        feat1 = sample_feat_by_coord(xf1, coord1_n, self.config['loss_distance']=='cos') #bxmxc
        feat2 = sample_feat_by_coord(xf2, coord2_n, self.config['loss_distance']=='cos') #bxnxc

        # matching
        if self.config['match_grad']:
            costs = 1-feat1@feat2.transpose(1,2) # bxmxn 0-2
        else:
            with torch.no_grad():
                costs = 1-feat1@feat2.transpose(1,2) # bxmxn 0-2
        affinity = -temperature * costs

        cat_I = Categorical(logits=affinity)
        cat_T = Categorical(logits=affinity.transpose(1,2))

        dense_p     = cat_I.probs * cat_T.probs.transpose(1,2)
        dense_logp  = cat_I.logits + cat_T.logits.transpose(1,2)

        if self.config['cor_detach']:
            sample_p = dense_p.detach()
        else:
            sample_p = dense_p

        reward, scale1, scale2, fundamental_loss = self.my_reward(inputs, outputs, coord1, coord2, **self.config['reward_config'])
        # reward, scale1, scale2 = self.posfeat_reward(inputs, outputs, coord1, coord2, **self.config['reward_config'])

        kps_logp = logp1.reshape(b,1,-1).transpose(1,2) + logp2.reshape(b,1,-1) # bxmxn
        sample_plogp = sample_p * (dense_logp + kps_logp)
        accept_mask = accept_mask1.reshape(b,1,-1).transpose(1,2) * accept_mask2.reshape(b,1,-1) # bxmxn
        
        reinforce   = (reward[accept_mask] * sample_plogp[accept_mask]).sum()
        # kp_penalty  = self.kp_penalty * (logp1[accept_mask1].sum()+logp2[accept_mask2].sum())
        if fundamental_loss is None:
            kp_penalty  = self.kp_penalty * (logp1[accept_mask1].sum()+logp2[accept_mask2].sum())
        else:
            kp_penalty  = self.kp_penalty * fundamental_loss * (logp1[accept_mask1].sum()+logp2[accept_mask2].sum())

        loss = -reinforce - kp_penalty

        sample_p_detach = sample_p.detach()
        components = {
            'reinforce':    reinforce.detach(), 
            'kp_penalty':   kp_penalty.detach(), 
            'scale1':       scale1, 
            'scale2':       scale2, 
            'cor minmax':   sample_p_detach.view(b,-1).max(-1)[0].min(), 
            'cor minmean':  sample_p_detach.view(b,-1).mean(-1).min(), 
            'cor max':      sample_p_detach.max(), 
            'cor mean':     sample_p_detach.mean(), 
            'cor summin':   torch.min(sample_p_detach.sum(1).min(), sample_p_detach.sum(2).min()), 
            'cor summax':   torch.max(sample_p_detach.sum(1).max(), sample_p_detach.sum(2).max()),
            'n_kps':        (accept_mask1.detach().reshape(b,1,-1).sum(-1) + accept_mask2.detach().reshape(b,1,-1).sum(-1)).float().mean(),
            'n_pairs':      sample_p.detach().sum(-1).sum(-1).mean(),
            'temperature':  sample_p_detach.new_tensor(temperature)
            }
        return loss, components