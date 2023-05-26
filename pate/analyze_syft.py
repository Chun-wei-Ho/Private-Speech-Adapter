import syft.frameworks.torch.dp.pate as pate

import numpy as np

import os

import argparse

import aggregation

import sys

import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_eps", type=float, required=True)
    parser.add_argument("--counts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--moment", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.09)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--n_chunks", type=int, default=2)
    args = parser.parse_args()

    input_mat = np.load(args.counts_file)
    input_mat = input_mat[:, args.chunk_id::args.n_chunks, :]
    teacher_preds = input_mat.argmax(axis=-1)

    n_iter = 35
    noise_eps_range = [1e-10, 10]
    pbar = tqdm.tqdm(range(n_iter))
    for i in pbar:
        noise_eps = np.mean(noise_eps_range)
        indices = aggregation.noisy_max(input_mat, 1/noise_eps)
        sys.stdout = open(os.devnull, 'w')
        data_dependent_privacy, data_independent_privacy = \
            pate.perform_analysis(teacher_preds, \
                                indices, noise_eps, \
                                delta=args.delta, moments=args.moment, beta=args.beta, nclass=input_mat.shape[-1])
        sys.stdout = sys.__stdout__
        if data_dependent_privacy < args.target_eps:
            noise_eps_range[1] = noise_eps
        else:
            noise_eps_range[0] = noise_eps
        pbar.set_description(f"epsilon={data_dependent_privacy}, lap_scale={1/noise_eps}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "lap_scale"), 'w') as f:
        f.write(str(1/noise_eps))
    with open(os.path.join(args.output_dir, "data_dependent_privacy"), 'w') as f:
        f.write(str(data_dependent_privacy))
    with open(os.path.join(args.output_dir, "data_independent_privacy"), 'w') as f:
        f.write(str(data_independent_privacy))
