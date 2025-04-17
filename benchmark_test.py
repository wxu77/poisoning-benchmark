############################################################
#
# benchmark_test.py
# Code to execute benchmark tests
# Developed as part of Poison Attack Benchmarking project
# June 2020
#
############################################################
import argparse
import os

import poison_test
from learning_module import now, model_paths, set_defaults


def main(args):
    """Main function to run a benchmark test
    input:
        args:       Argparse object that contains all the parsed values
    return:
        void
    """

    print(now(), "benchmark_test.py running.")
    out_dir = args.output
    if not args.from_scratch:
        print(
            f"Testing poisons from {args.poisons_path}, in the transfer learning "
            f"setting...\n"
        )

        ####################################################
        #          Transfer learning
        print("Transfer learning test:")
        print(args)

        # white-box attack
        args.output = args.output
        args.model = args.model
        args.model_path = args.model_path
        poison_test.main(args)

        # black box attacks
        # args.output = os.path.join(out_dir, "ffe-bb")

        # args.model = models[1]
        # args.model_path = model_paths[args.dataset]["blackbox"][0]
        # poison_test.main(args)

        # args.model_path = model_paths[args.dataset]["blackbox"][1]
        # args.model = models[2]
        # poison_test.main(args)

    # else:
    #     print(
    #         f"Testing poisons from {args.poisons_path}, in the from scratch training "
    #         f"setting...\n"
    #     )

    #     ####################################################
    #     #           From Scratch Training (fst)
    #     args.model_path = None
    #     args.output = os.path.join(out_dir, "fst")

    #     if args.dataset.lower() == "cifar10":
    #         print(f"From Scratch testing for {args.dataset}")
    #         args.model = "resnet18"
    #         poison_test.main(args)

    #         args.model = "MobileNetV2"
    #         poison_test.main(args)

    #         args.model = "VGG11"
    #         poison_test.main(args)

        # else:
        #     print(f"From Scratch testing for {args.dataset}")
        #     args.model = "vgg16"
        #     poison_test.main(args)



def main2(passed_args):
    parser = argparse.ArgumentParser(description="PyTorch poison benchmarking")
    parser.add_argument(
        "--from_scratch", action="store_true", help="Train from scratch with poisons?"
    )
    parser.add_argument(
        "--poisons_path", type=str, required=True, help="where are the poisons?"
    )
    parser.add_argument(
        "--model", type=str, required=True,
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="dataset")
    parser.add_argument(
        "--output", default="output_default", type=str, help="output subdirectory"
    )
    parser.add_argument(
        "--target_img_idx", type=int, required=True,
    )

    args = parser.parse_args(passed_args)
    set_defaults(args)
    main(args)
