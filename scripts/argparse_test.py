import argparse

parser = argparse.ArgumentParser()
parser.add_argument('sweep')
parser.add_argument('do_cloudy')
parser.add_argument('do_irradiated')
parser.add_argument('full_save')

args = parser.parse_args()
print(args.sweep, bool(args.do_cloudy))