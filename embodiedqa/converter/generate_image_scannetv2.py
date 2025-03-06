# Modified from https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py # noqa
import os
import struct
import zlib
from argparse import ArgumentParser
from functools import partial
# import imageio
import imageio.v2 as imageio
import mmengine
import numpy as np

COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}

COMPRESSION_TYPE_DEPTH = {
    -1: 'unknown',
    0: 'raw_ushort',
    1: 'zlib_ushort',
    2: 'occi_ushort'
}


class RGBDFrame:
    """Class for single ScanNet RGB-D image processing."""

    def load(self, file_handle):
        """Load basic information of a given RGBD frame."""
        self.camera_to_world = np.asarray(struct.unpack(
            'f' * 16, file_handle.read(16 * 4)),
                                          dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = b''.join(
            struct.unpack('c' * self.color_size_bytes,
                          file_handle.read(self.color_size_bytes)))
        self.depth_data = b''.join(
            struct.unpack('c' * self.depth_size_bytes,
                          file_handle.read(self.depth_size_bytes)))

    def decompress_depth(self, compression_type):
        """Decompress the depth data."""
        assert compression_type == 'zlib_ushort'
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        """Decompress the RGB image data."""
        assert compression_type == 'jpeg'
        return imageio.imread(self.color_data)


class SensorData:
    """Class for single ScanNet scene processing.

    Single scene file contains multiple RGB-D images.
    """

    def __init__(self, filename,output_path, fast=False):
        self.output_path = output_path
        self.version = 4
        self.load(filename, fast)

    def load(self, filename, fast):
        """Load a single scene data with multiple RGBD frames."""
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = b''.join(
                struct.unpack('c' * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(struct.unpack(
                'f' * 16, f.read(16 * 4)),
                                              dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack(
                'f' * 16, f.read(16 * 4)),
                                              dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack(
                'f' * 16, f.read(16 * 4)),
                                              dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack(
                'f' * 16, f.read(16 * 4)),
                                              dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack(
                'i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack(
                'i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]
            num_frames = struct.unpack('Q', f.read(8))[0]
            self.num_frames = num_frames
            self.frames = []
            if fast:
                index = list(range(num_frames))[::10]
            else:
                index = list(range(num_frames))
            self.index = index
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                if i in index:
                    # self.frames.append(frame)
                    self.export_depth_images(self.output_path, frame, i)
                    self.export_color_images(self.output_path, frame, i)
                    self.export_poses(self.output_path, frame, i)
    def export_depth_images(self, output_path, frame, index):
        """Export depth images to the output path."""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        depth_data = frame.decompress_depth(
            self.depth_compression_type)
        # depth = np.fromstring(depth_data, dtype=np.uint16).reshape(
        #     self.depth_height, self.depth_width)
        depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
            self.depth_height, self.depth_width)
        imageio.imwrite(
            os.path.join(output_path,
                            self.index_to_str(index) + '.png'), depth)

    def export_color_images(self, output_path, frame, index):
        """Export RGB images to the output path."""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        color = frame.decompress_color(
            self.color_compression_type)
        imageio.imwrite(
            os.path.join(output_path,
                            self.index_to_str(index) + '.jpg'), color)


    @staticmethod
    def index_to_str(index):
        """Convert the sample index to string."""
        return str(index).zfill(5)

    @staticmethod
    def save_mat_to_file(matrix, filename):
        """Save a matrix to file."""
        with open(filename, 'w') as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt='%f')

    def export_poses(self, output_path, frame, index):
        """Export camera poses to the output path."""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.save_mat_to_file(
            frame.camera_to_world,
            os.path.join(output_path,
                            self.index_to_str(index) + '.txt'))

    def export_intrinsics(self, output_path):
        """Export the intrinsic matrix to the output path."""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.save_mat_to_file(self.intrinsic_color,
                              os.path.join(output_path, 'intrinsic.txt'))

    def export_depth_intrinsics(self, output_path):
        """Export the depth intrinsic matrix to the output path."""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.save_mat_to_file(self.intrinsic_depth,
                              os.path.join(output_path, 'depth_intrinsic.txt'))


def process_scene(path, fast, idx):
    """Process single ScanNet scene.

    Extract RGB images, poses and camera intrinsics.
    """
    if 'scan_test' in path:
        output_path = os.path.join('posed_images_test', idx)
    else:
        output_path = os.path.join('posed_images', idx)
    try:
        if not os.path.isfile(os.path.join(output_path, 'depth_intrinsic.txt')):
            data = SensorData(os.path.join(path, idx, f'{idx}.sens'),output_path, fast)
            data.export_intrinsics(output_path)
            data.export_depth_intrinsics(output_path)
        else:
            print(f'skip {output_path}!')
    except:
        print(f'{idx}: this some error in {path}')
        if os.path.isfile(os.path.join(path, idx, f'{idx}.sens')):
            os.remove(os.path.join(path, idx, f'{idx}.sens'))


def process_directory(path, fast, nproc):
    """Process the files in a directory with parallel support."""
    mmengine.track_parallel_progress(func=partial(process_scene, path, fast),
                                     tasks=os.listdir(path),
                                     nproc=nproc)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_folder',
                        default='./data/scannet',
                        help='folder of the dataset.')
    parser.add_argument('--nproc', type=int, default=4)
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    if args.dataset_folder is not None:
        os.chdir(args.dataset_folder)

    # process train and val scenes
    if os.path.exists('scans'):
        process_directory('scans', args.fast, args.nproc)
    if os.path.exists('scans_test'):
        process_directory('scans_test', args.fast, args.nproc)
