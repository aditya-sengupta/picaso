import argparse

parser = argparse.ArgumentParser()
parser.add_argument('sweep')
parser.add_argument('cloudy')
parser.add_argument('save')

args = parser.parse_args()
print(args.sweep, bool(args.do_cloudy))