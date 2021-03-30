from argparse import ArgumentParser
from mnist.train import train
from mnist.export import export

parser = ArgumentParser()
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("-e", "--epochs", type=int, default=3)
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("-s", "--save", action="store_true")
args = parser.parse_args()

if args.train: train(args.output, epochs=args.epochs)
if args.save: export(args.output)