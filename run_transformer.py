from models.transformer import build_transformer
from main import get_args_parser
from torchinfo import summary
import argparse

parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
args.enc_layers = 4
args.dec_layers = 4
print(args)
model = build_transformer(args)
summary(model, input_size=(2, 256, 20, 30), depth=100)
