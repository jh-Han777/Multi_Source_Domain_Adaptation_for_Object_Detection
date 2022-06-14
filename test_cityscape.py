import os
import argparse
net = "vgg16"
part = "test_t"
start_epoch = 7
max_epochs = 12
output_dir = "result/"
dataset = "cityscape"

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
model_prefix=args.model_prefix+"/cityscape_"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

for i in range(start_epoch, max_epochs + 1):
    model_dir = args.model_prefix+"cityscape_{}.pth".format(
        i
    )
    # command = "python eval/test_SW_ICR_CCR.py --cuda --gc --lc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
    #     part, net, dataset, model_dir, output_dir, i
    # )
    command = "python eval/test_strong_weak.py --cuda --gc --lc --vis --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
        part, net, dataset, model_dir, output_dir, i
    )
    os.system(command)
