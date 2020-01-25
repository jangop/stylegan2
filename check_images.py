import argparse
import os

from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--mode', type=str, default='RGB')
parser.add_argument('--action', type=str, default='remove')

args = parser.parse_args()

def remove(path):
    os.remove(path)

if args.action == 'remove':
    action = remove
else:
    raise Exception()

for filename in tqdm(os.listdir(args.path)):
    full_path = os.path.join(args.path, filename)
    try:
        image = Image.open(full_path)
        image.verify()
        assert(image.mode == args.mode)
    except Exception as e:
        print('Removing {}...'.format(filename))
        action(full_path)
        print (e)
