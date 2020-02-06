import argparse
import math
import os
from enum import Flag, auto

import numpy as np
import skimage.measure
from PIL import Image
from tqdm import tqdm

import imagehash


class Offense(Flag):
    CORRUPT = auto()
    MODE = auto()
    DUPLICATE = auto()
    SIZE = auto()
    ENTROPY = auto()


class Overseer():
    def __init__(self, base_dir, mode, action, hash_function, side_length,
                 aux_dir, entropy_threshold, check_entropy, check_hash,
                 soft_hash, soft_hash_treshold, thumbnail_size,
                 create_thumbnails):
        self._base_dir = base_dir
        self._mode = mode
        self._action = action
        self._hash_function = hash_function
        self._side_length = side_length
        self._aux_dir = aux_dir
        self._entropy_threshold = entropy_threshold
        self._check_entropy = check_entropy
        self._check_hash = check_hash
        self._soft_hash = soft_hash
        self._soft_hash_threshold = soft_hash_treshold
        self._thumbnail_size = thumbnail_size
        self._create_thumbnails = create_thumbnails

        self.offenses = {}
        self.hashes = {}
        self.entropies = {}

        self.n_files_inspected = 0
        self.n_offending_files = 0

    def _add_offense(self, obj, offense):
        if obj in self.offenses:
            self.offenses[obj] = self.offenses[obj] | offense
        else:
            self.offenses[obj] = offense

    def inspect(self):
        for filename in tqdm(os.listdir(self._base_dir),
                             desc='Inspecting images'):
            self.n_files_inspected += 1
            full_path = os.path.join(self._base_dir, filename)
            # Verify image consistency.
            try:
                image = Image.open(full_path)
                image.verify()
            except Exception as e:
                print('UNSPECIFIC EXCEPTION')
                print(type(e))
                print('{} did not open or verify'.format(filename))
                self._add_offense(filename, Offense.CORRUPT)
                continue

            # Verify correct mode.
            if image.mode != self._mode:
                self._add_offense(filename, Offense.MODE)
                continue

            # Verify correct size.
            width, height = image.size
            if width != self._side_length or height != self._side_length or width != height:
                self._add_offense(filename, Offense.SIZE)
                continue

            # Determine perceptual hash.
            if self._check_hash:
                try:
                    if args.hash == 'dhash':
                        image = Image.open(full_path)
                        image.load()
                        width, height = image.size
                        center = width / 2
                        quarter = center / 2
                        crop = image.crop((center - quarter, center - quarter,
                                           center + quarter, center + quarter))
                        crop_hash = self._hash_function(crop)
                        self.hashes[crop_hash] = self.hashes.get(
                            crop_hash, []) + [filename]
                except OSError as e:
                    self._add_offense(filename, Offense.CORRUPT)

            # Determine entropy.
            if self._check_entropy:
                try:
                    image = Image.open(full_path)
                    image.load()
                    entropy = skimage.measure.shannon_entropy(image)
                    self.entropies[filename] = entropy
                    if entropy < self._entropy_threshold:
                        self._add_offense(filename, Offense.ENTROPY)
                except OSError as e:
                    self._add_offense(filename, Offense.CORRUPT)

        # Find (near) duplicates.
        if self._soft_hash:
            soft_hashes = {}
            for hash_value, filenames in self.hashes.items():
                new = True
                for soft_hash_value, soft_filenames in soft_hashes.items():
                    if (soft_hash_value -
                            hash_value) < self._soft_hash_threshold:
                        soft_hashes[soft_hash_value] += filenames
                        new = False
                        break
                if new:
                    soft_hashes[hash_value] = filenames
            self._hashes = soft_hashes

        for hash_value, filenames in self.hashes.items():
            if len(filenames) > 1:
                first = True
                for filename in filenames:
                    if first:
                        first = False
                        continue
                    self._add_offense(filename, Offense.DUPLICATE)

    def log(self):
        self.n_offending_files = len(self.offenses)
        print('{} offending files out of {} total, i.e. {} %'.format(
            self.n_offending_files, self.n_files_inspected,
            self.n_offending_files / self.n_files_inspected * 100))

        if self._aux_dir is not None:
            entropy_path = os.path.join(self._aux_dir, 'entropy.txt')
            with open(entropy_path, 'w') as entropy_handle:
                for filename, entropy in tqdm(self.entropies.items(),
                                              desc='Logging entropy'):
                    entropy_handle.write('{} {}\n'.format(filename, entropy))

                    if self._create_thumbnails and entropy < self._entropy_threshold:
                        full_path = os.path.join(self._base_dir, filename)
                        image = Image.open(full_path)
                        image.load()
                        image = image.resize(
                            (self._thumbnail_size, self._thumbnail_size))
                        image.save(
                            os.path.join(self._aux_dir, 'low-entropy',
                                         '{}-{}.jpg'.format(entropy,
                                                            filename)))

            hash_path = os.path.join(self._aux_dir, 'hashes.txt')
            with open(hash_path, 'w') as hash_handle:
                for hash_value, filenames in tqdm(self.hashes.items(),
                                                  desc='Logging hashes'):
                    n_filenames = len(filenames)
                    if n_filenames < 2:
                        continue
                    hash_handle.write(
                        str(hash_value) + ' ' + ' '.join(filenames) + '\n')
                    if self._create_thumbnails:
                        n_cols = min(16, n_filenames)
                        n_rows = math.ceil(n_filenames / n_cols)

                        grid = np.zeros((n_rows * self._thumbnail_size,
                                         n_cols * self._thumbnail_size, 3),
                                        dtype='uint8')
                        for i_filename, filename in enumerate(filenames):
                            i_row = int(i_filename / n_cols)
                            i_col = i_filename % n_cols
                            full_path = os.path.join(self._base_dir, filename)
                            image = Image.open(full_path)
                            image.load()
                            image = image.resize(
                                (self._thumbnail_size, self._thumbnail_size))
                            grid[i_row * self._thumbnail_size:(i_row + 1) *
                                 self._thumbnail_size, i_col *
                                 self._thumbnail_size:(i_col + 1) *
                                 self._thumbnail_size, :] = np.asarray(image)
                        image = Image.fromarray(grid)
                        image.save(
                            os.path.join(self._aux_dir, 'duplicates',
                                         '{}.jpg'.format(hash_value)))

    def sweep(self):
        for filename in tqdm(self.offenses, desc='Sweeping'):
            full_path = os.path.join(self._base_dir, filename)
            #            print('Handling {}...'.format(filename))
            #            print(self.offenses[filename])
            self._action(full_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--mode', type=str, default='RGB')
    parser.add_argument('--action', type=str, default='list')
    parser.add_argument('--hash', type=str, default='dhash')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--aux', type=str, default='/tmp/checks')
    parser.add_argument('--entropy', type=float, default=3.0)
    parser.add_argument('--check-entropy', action='store_true')
    parser.add_argument('--check-hash', action='store_true')
    parser.add_argument('--soft-hash', action='store_true')
    parser.add_argument('--soft-hash-treshold', type=int, default=8)
    parser.add_argument('--thumbnail-size', type=int, default=256)
    parser.add_argument('--create-thumbnails', action='store_true')

    args = parser.parse_args()

    if args.aux is not None:
        os.makedirs(args.aux, exist_ok=True)
        if args.create_thumbnails:
            os.makedirs(os.path.join(args.aux, 'duplicates'), exist_ok=True)
            os.makedirs(os.path.join(args.aux, 'low-entropy'), exist_ok=True)

    if args.action == 'remove':

        def action(path):
            os.remove(path)
    elif args.action == 'list':

        def action(path):
            pass
    else:
        parser.error('TODO')

    if args.hash == 'dhash':
        hash_function = imagehash.dhash
    else:
        parser.error('TODO')

    overseer = Overseer(args.path, args.mode, action, hash_function, args.size,
                        args.aux, args.entropy, args.check_entropy,
                        args.check_hash, args.soft_hash,
                        args.soft_hash_treshold, args.thumbnail_size,
                        args.create_thumbnails)
    overseer.inspect()
    overseer.log()
    overseer.sweep()
