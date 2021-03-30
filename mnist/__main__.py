from argparse import ArgumentParser
from mnist.train import train


parser = ArgumentParser()
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("-e", "--epochs", type=int, default=3)
args = parser.parse_args()

train(args.output, epochs=args.epochs)