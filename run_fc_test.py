import argparse
import os
import torchvision
import numpy as np
import random

import benchmark_test
import poison_crafting.craft_poisons_fc


def get_random_base_indices(dataset: str, poison_class: int, num_poisons: int):
    if dataset.lower() == "cifar10":
        full_train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
        all_idx_for_poison_class = list(np.where(np.array(full_train_dataset.targets) == poison_class)[0])

        assert num_poisons < len(all_idx_for_poison_class), (f'num_poisons={num_poisons}, all_idx_for_poison_class={len(all_idx_for_poison_class)}')
        temp_random = random.Random(0)  # choose the same base indices every time for now
        indices = temp_random.sample(all_idx_for_poison_class, num_poisons)

        # print(all_idx_for_poison_class[:10])
        # print(indices)    # this should never change
        # print(random.randint(0, 5))   # this should keep changing
    else:
        raise NotImplementedError()

    assert len(indices) == num_poisons
    return indices


def main(args):
    # Determine paths
    poisons_path = os.path.join(args.output_path, "poisons", args.model, args.dataset, args.pretrain_dataset, str(args.target_id), str(args.poison_class))
    output_path = os.path.join(args.output_path, "logs", str(args.dist_rank))

    # Generate the poisons
    fc_args = [
        '--model', args.model,
        '--dataset', args.dataset,
        '--pretrain_dataset', args.pretrain_dataset,
        '--model_path', args.model_path,
        '--poison_setups', args.poison_setups,
        '--poisons_path', poisons_path,
        '--output', output_path,
        '--target_img_idx', str(args.target_id),
        '--base_indices', *[str(i) for i in get_random_base_indices(dataset=args.dataset, poison_class=args.poison_class, num_poisons=args.num_poisons)]
    ]

    poison_crafting.craft_poisons_fc.main2(fc_args)

    # Run modified benchmark_test
    bench_args = [
        '--model', args.model,
        '--model_path', args.model_path,
        '--poisons_path', poisons_path,
        '--dataset', args.dataset,
        '--output', output_path,
        '--target_img_idx', str(args.target_id),
    ]

    benchmark_test.main2(bench_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch poison benchmarking")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pretrain_dataset", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--poison_setups", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--num_poisons", type=int)
    parser.add_argument("--target_id", type=int)
    parser.add_argument("--poison_class", type=int)
    parser.add_argument("--dist_rank", type=int)

    args = parser.parse_args()
    main(args)
