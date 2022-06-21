import os
import argparse
net = "vgg16"
part = "test_t"
start_epoch = 7
max_epochs = 12
output_dir = "result/"
dataset = "mskda_bdd"

def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train a Fast R-CNN network")
    parser.add_argument(
        "--model_prefix",
        dest="model_prefix",
        help="directory to trained model",
        default=" ",
        type=str,
    )
    parser.add_argument(
        "--net", dest="net", help="vgg16, res101", default="vgg16", type=str
    )

    parser.add_argument(
        "--gpus", dest="gpus", help="gpu number", default="3", type=str
    )

    args = parser.parse_args()
    return args

args = parse_args()
net=args.net
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

for i in range(start_epoch, max_epochs + 1):
    model_dir = args.model_prefix+"mskda_bdd_{}.pth".format(
        i
    )
    command = "python eval/test_msda.py --cuda --gc --lc --vis --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
        part, net, dataset, model_dir, output_dir, i
    )
    os.system(command)
