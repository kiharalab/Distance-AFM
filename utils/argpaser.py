import argparse
import torch
import os


def argparser():
    parser = argparse.ArgumentParser()
    # for input file
    parser.add_argument(
        "--target_file",
        required=True,
        type=str,
        help="File of targets for training",
    )
    parser.add_argument(
        "--emd_file",
        required=True,
        type=str,
        help="AlphaFold-Multimer result .pkl file containing 'representations'",
    )
    parser.add_argument(
        "--dist_info",
        required=True,
        type=str,
        help="distance constraint file",
    )
    parser.add_argument(
        "--window_info",
        required=True,
        type=str,
        help="window info to specify different domains",
    )
    parser.add_argument(
        "--initial_pdb",
        required=True,
        type=str,
        help="Starting structure for overfitting",
    )
    parser.add_argument(
        "--fasta_file",
        default=None,
        type=str,
        help="the path of fasta sequence (optional, not used by the multimer pipeline)",
    )
    parser.add_argument(
        "--output_dir",
        default="./example_output",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        help="model directory if load model from checkpoints",
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=10000,
        help="Maximum sequnce length, larger proteins are clipped",
    )
    parser.add_argument(
        "--epochs", default=20000, type=int, help="Total number of training epochs."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout probability"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="L2 Regularization"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for training")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Num of workers in dataloader"
    )
    parser.add_argument(
        "--resnest_blocks",
        type=int,
        nargs="+",
        default=[2, 1, 1, 1],
        help="ResNeSt blocks, each block has 3 conv",
    )
    parser.add_argument(
        "--af2", default=True, action="store_true", help="whether use af2 embeddings"
    )
    parser.add_argument(
        "--val_epochs",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--embed",
        type=str,
        default="msa_transformer",
        help="Options: onehot | tape | onehot_tape | msa_transformer",
    )
    parser.add_argument("--device_id", type=int, default=0, help="cude device id")
    parser.add_argument(
        "--seed", type=int, default=999, help="random seed for initialization"
    )
    parser.add_argument("--ipa_depth", type=int, default=8, help="depth of ipd block")
    parser.add_argument(
        "--point_scale", type=int, default=20, help="point scale for translations"
    )
    parser.add_argument(
        "--dist", default=1, type=int, help="whether to use distance constraint"
    )
    # parser.add_argument("--dist_window", default=1, type=int, help='whether to use distance constraint')
    parser.add_argument(
        "--dist_range",
        default=0,
        type=int,
        help="whether to use distance range constraint",
    )
    parser.add_argument("--domain_relative", default=1, type=int)
    parser.add_argument("--angle_loss", type=int, default=1)
    parser.add_argument(
        "--dist_weight",
        type=float,
        default=0.5,
        help="adjustable for weight of distance loss",
    )
    parser.add_argument(
        "--loose_dist",
        type=int,
        default=1,
        help="if loose the weight of distance loss when it is smaller than 1",
    )
    parser.add_argument(
        "--use_checkpoint",
        type=int,
        default=1,
        help="if using torch checkpoint to save GPU RAM",
    )
    parser.add_argument(
        "--config_preset",
        type=str,
        default="model_1_multimer_v3",
        help="config preset for training",
    )
    parser.add_argument(
        "--selected_chains", type=str, default=None, help="selected chains for training"
    )
    parser.add_argument(
        "--using_fafe", type=int, default=0, help="whether to use fafe loss"
    )
    parser.add_argument(
        "--weight_file", type=str, default=None, help="weight file for training"
    )
    parser.add_argument(
        "--rigid_body_improve",
        type=int,
        default=0,
        help="whether to use rigid body improvement",
    )
    # parser.add_argument("--use_pretrained", type=int, default=1, help="whether to use previous trained model")
    parser.add_argument(
        "--use_estimation",
        type=int,
        default=0,
        help="whether to use estimation for training",
    )
    parser.add_argument(
        "--start_pdb",
        type=str,
        default=None,
        help="start structure for coarse tuning (e.g. <TARGET>_start_0.pdb from generate_start_structures.py)"
    )
    parser.add_argument(
        "--coarse_tuning_epochs",
        type=int,
        default=3000,
        help="number of coarse tuning epochs trained towards --start_pdb",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    args.device_id = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cuda = True if torch.cuda.is_available() else False

    args.n_gpu = 1
    return args
