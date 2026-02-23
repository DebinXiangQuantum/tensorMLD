from args import args
import torch
import time
import sys
from pymatching import Matching
import numpy as np
from torch.optim.lr_scheduler import StepLR
from os.path import abspath, dirname, exists
sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import MADE, Data, mod2


def forward(n_s, m, van, syndrome, device, dtype, k=1):
    condition = syndrome*2-1
    x = (van.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k) + 1)/2
    x = x[:, m:m+int(2*k)]
    return x


def training(ni, k, epoch, batch, lr, device, van, optimizer, scheduler, sampler, save_path=None):
    for l in range(1, epoch+1):
        dets, obvs = sampler.sample(shots=batch, separate_observables=True)

        s = torch.hstack((torch.tensor(dets)*1.0, torch.tensor(obvs)*1.0)).to(device).to(torch.float32)
        logp = van.log_prob((s*2-1))
    
        loss = torch.mean((-logp), dim=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if optimizer.state_dict()['param_groups'][0]['lr'] > 0.0002 :
            scheduler.step()
        
        if (l+1) % 1000 == 0 or l<=100:
                
                t0 = time.time()
                lconf = forward(n_s=10000, m=ni-k, van=van, syndrome=torch.tensor(dets)*1.0, device=device, dtype=dtype, k=k/2)
                # print(lconf[0])
                aclo = (torch.tensor(obvs)*1.0).to(device).to(dtype)
                # print(aclo[0])
                logical_error_rate = torch.count_nonzero((aclo-lconf).sum(1))/10000
                t1 = time.time()
                # print('decoding time:', t1-t0)
                # print('mw:', lmw)
                print('\r epoch: {}, loss: {}, gnd_ler: {}'.format(l, loss.cpu().item(), logical_error_rate.cpu().item()), end='')

        if save_path is not None:

            torch.save(van, save_path)

def decoding(net, dets, obvs):
    lconf = forward(n_s=dets.shape[0], m=ni-k, van=net, syndrome=torch.tensor(dets)*1.0, device=device, dtype=dtype, k=k/2)
    aclo = (torch.tensor(obvs)*1.0).to(device).to(dtype)
    logical_error_rate = torch.count_nonzero((aclo-lconf).sum(1))/10000
    return logical_error_rate


if __name__ == '__main__':
    from module import qcc_circuit
    
    
    epoch = 150000
    lr = 0.001
    batch  = 10000
    error_rate = 0.01
    seed = 0
    dtype = torch.float32
    device = 'cuda:0'
    


    qccc = qcc_circuit(error_rate = error_rate)
    dem = qccc.detector_error_model(flatten_loops=True, decompose_errors=False)
    sampler = qccc.compile_detector_sampler(seed=seed)
    ni = dem.num_detectors + dem.num_observables
    k = dem.num_observables
    print(ni)
    van = MADE(n=ni, depth=4, width=30, residual=False).to(device).to(dtype)
    optimizer = torch.optim.Adam(van.parameters(), lr=lr)#, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.9)

    training(ni=ni, k=k, epoch=epoch, batch=batch, lr=lr, device=device, 
        van=van, optimizer=optimizer, scheduler=scheduler, sampler=sampler, 
        save_path=None)
        
    
    
