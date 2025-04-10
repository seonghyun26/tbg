import os
import argparse

parser = argparse.ArgumentParser(description='Train TBG model')
parser.add_argument('--date', type=str, default="debug", help='Date for the experiment')

args = parser.parse_args()

print("azsdf")
print(args.date)