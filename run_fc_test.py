import argparse
from types import SimpleNamespace
import os

import benchmark_test
import poison_crafting.craft_poisons_fc


def main(args):
    # Determine paths
    poisons_path = os.path.join(args.output_path, "poisons", args.model, args.dataset, args.pretrain_dataset, str(args.target_img_idx), str(args.poison_class))
    output_path = os.path.join(args.output_path, str(args.dist_rank))

    # Generate the random base indices
    # TODO

    # Generate the poisons
    fc_args = SimpleNamespace()
    fc_args.model = args.model
    fc_args.dataset = args.dataset
    fc_args.pretrain_dataset = args.pretrain_dataset
    fc_args.model_path = args.model_path
    fc_args.poison_setups = args.poison_setups
    fc_args.poisons_path = poisons_path

    fc_args.target_img_idx = args.target_id
    fc_args.base_indices = []   # TODO

    poison_crafting.craft_poisons_fc.main(args)

    # Run modified benchmark_test
    bench_args = SimpleNamespace()
    bench_args.model = args.model
    bench_args.model_path = args.model_path
    bench_args.poisons_path = poisons_path
    bench_args.dataset = args.dataset
    bench_args.output = output_path

    benchmark_test.main(bench_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch poison benchmarking")

    parser.add_argument("--model", str, required=True)
    parser.add_argument("--dataset", str, required=True)
    parser.add_argument("--pretrain_dataset", str, required=True)
    parser.add_argument("--model_path", str, required=True)
    parser.add_argument("--poison_setups", str, required=True)
    parser.add_argument("--output_path", str, required=True)

    parser.add_argument("--num_poisons", type=int)
    parser.add_argument("--target_id", type=int)
    parser.add_argument("--poison_class", type=int)
    parser.add_argument("--dist_rank", type=int)

    args = parser.parse_args()
    main(args)
