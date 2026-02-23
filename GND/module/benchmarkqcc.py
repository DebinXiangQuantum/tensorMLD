import torch
import re
from ldpc import bposd_decoder
import numpy as np
import stim
from pymatching import Matching
import time
import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)).strip('benchmarkqcc'))
# print(abspath(dirname(__file__)))
from .qcc_circuit import create_bivariate_bicycle_codes, build_circuit


class Convert:
    def __init__(self, 
                 DEM_flatten=stim.DetectorErrorModel,
                 dtype=int, 
                 device='cpu') -> None:
        self.dtype = dtype
        self.device = device
        self.DEM_flatten = DEM_flatten
        self.dem_str = str(self.DEM_flatten).split('\n')
        self.num_detectors = DEM_flatten.num_detectors
        self.num_ems = DEM_flatten.num_errors

    def stabilizer(self):
        dem_str = self.dem_str[:self.num_ems]
        stabilizer = torch.zeros(
            (self.num_detectors, self.num_ems), dtype=self.dtype)
        for col, line in enumerate(dem_str):
            for part in line.split():
                if part.startswith('D'):
                    row = int(part[1:])
                    stabilizer[row, col] = 1
        return stabilizer

    def logical(self):
        dem_str = self.dem_str[:self.num_ems]
        # Determine the number of unique logical operators
        unique_logical_operators = set()
        for line in dem_str:
            for part in line.split():
                if part.startswith('L'):
                    unique_logical_operators.add(part)
        num_logical_operators = len(unique_logical_operators)
        logical = torch.zeros((num_logical_operators, self.num_ems), dtype=self.dtype)
        for col, line in enumerate(dem_str):
            for part in line.split():
                if part.startswith('L'):
                    logical_index = int(part[1:])
                    logical[logical_index, col] = 1  
        return logical

    def pro_list(self):
        float_numbers = []
        for i, line in enumerate(self.dem_str):
            if i < self.num_ems:
                match = re.search(r'\((.*?)\)', line)
                if match:
                    float_number = float(match.group(1))
                    float_numbers.append(float_number)
            else:
                break
        return float_numbers
    

def bposd(circuit, num_samples, seed = 0, L = 0 ):
    samples, logical_samples = circuit.compile_detector_sampler(
        seed=seed).sample(num_samples, separate_observables=True)
    logical_samples = logical_samples[:,L:L+1]
    dem = circuit.detector_error_model(flatten_loops=True, decompose_errors=False)

    convertion = Convert(dem)
    stabilizer = convertion.stabilizer()
    logical = convertion.logical()
    pro_list = convertion.pro_list()
    pcm = stabilizer.numpy()
    pcm_l = logical.numpy()
    bpd=bposd_decoder(
        pcm,#the parity check matrix
        channel_probs=pro_list, #assign error_rate to each qubit. This will override "error_rate" input variable
        max_iter=10000, #the maximum number of iterations for BP)
        bp_method="ms",
        ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
        osd_method="osd0", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
        # osd_order=7 #the osd search depth
        )
    decode_em = []
    fail_num = 0
    t0 = time.perf_counter()
    for i in range(samples.shape[0]):
        bpd.decode(samples[i])
        residual_error=bpd.osdw_decoding
        decode_em.append(residual_error)
        a=((pcm_l[L]@residual_error + logical_samples[i])%2).any() # decode fail
        if a: 
            fail_num += 1 
            print("osd tirals", i, "Fail")
        elif not a:
            print("osd tirals", i, "success")
    t_gap = time.perf_counter() - t0
    logical_error_rate = fail_num/samples.shape[0]
    return logical_error_rate, decode_em, t_gap, samples, logical_samples, pcm, pcm_l


def find_columns(pcm, indices_array):
    num_columns = pcm.shape[1]
    result = np.zeros((1, num_columns), dtype=int)
    for row in indices_array:
        if row[1] == -1:
            row_index = row[0]
            for col in range(num_columns):
                if np.sum(pcm[:, col]) == 1 and pcm[row_index, col] == 1:
                    result[0, col] = 1
        else:
            row_index_1, row_index_2 = row
            for col in range(num_columns):
                if pcm[row_index_1, col] == 1 and pcm[row_index_2, col] == 1 and np.sum(pcm[:, col]) == 2:
                    result[0, col] = 1
    return result


def matching(circuit, num_samples, seed = 0):

    samples, logical_samples = circuit.compile_detector_sampler(
                            seed=seed).sample(num_samples, separate_observables=True)
    dem = circuit.detector_error_model(flatten_loops=True, decompose_errors=True)
    cv = Convert(DEM_flatten=dem)
    pcm = cv.stabilizer().numpy()
    pcm.shape
    pcm_l = cv.logical().numpy()
    matcher = Matching.from_detector_error_model(dem)
    predictions = matcher.decode_batch(samples)
    num_errors = 0
    decode_ems = []
    t0 = time.perf_counter()
    for shot in range(num_samples):

        decode_edges = matcher.decode_to_edges_array(samples[shot])
        decode_em = find_columns(pcm=pcm,indices_array=decode_edges)
        decode_ems.append(decode_em)
        actual_for_shot = logical_samples[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    t_gap = time.perf_counter() - t0
    logical_error_rate = num_errors/num_samples
    return logical_error_rate, decode_ems, t_gap, samples, logical_samples, pcm, pcm_l


def qcc_circuit(error_rate = 0.003,
        l = 5, 
        m = 6, 
        A_x_pows = [0], 
        A_y_pows = [1,2], 
        B_x_pows = [1,4], 
        B_y_pows = [1], 
        rounds = 2,
        **kwargs):
    # error_rate = 0.01
    # l = 6
    # m = 6
    # A_x_pows = [1,2]
    # A_y_pows = [3]
    # B_x_pows = [3]
    # B_y_pows = [1,2]
    # rounds = 2

    code, A_list, B_list = create_bivariate_bicycle_codes(l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows)
    circuit = build_circuit(code, A_list, B_list, 
                        p=error_rate, # physical error rate
                        num_repeat=rounds, # usually set to code distance
                        z_basis=True,   # whether in the z-basis or x-basis
                        use_both=True, # whether use measurement results in both basis to decode one basis
                       )
    return circuit


if __name__ == '__main__':  
    p = 0.008
    num_samples = 10000
    seed = 0
    
    qccc = qcc_circuit(error_rate = p)
    dem = qccc.detector_error_model(flatten_loops=True, decompose_errors=False)
    # print(dem)


    # L = 7
    # logical_error_rate_qcc, decoded_em, timeused, syndrome_samples, logical_obs_samples, pcm, pcm_obs = bposd(circuit=qccc,num_samples=num_samples,seed=seed, L=L)
    # assert decoded_em[0].shape[0] == qccc.detector_error_model(flatten_loops=True, decompose_errors=False).num_errors # num_EM
    # print(f"QCC_logical_error_rate_qubit{L} = \n", logical_error_rate_qcc)
    # print("QCC_time_used = \n", timeused)