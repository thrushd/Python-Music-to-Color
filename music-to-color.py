import argparse
import numpy as np
import math
from PIL import Image, ImageDraw
from pathlib import Path
import soundfile as sf
import cv2
from matplotlib import cm
import scipy.signal


def get_arguments():
    """
    Obtain and return the arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('file', type=str, help='path to the video file')
    parser.add_argument('-s', '--size', nargs=2, type=int, default=[1000, 1000], help='size of the image, in width height (1920 1080)')

    return parser.parse_args()


def generate_image(file_path, output_path, width, height):
    """
    Takes an audio file and generates an image.
    """
    # get audio info
    data, samplerate = sf.read(file_path)

    # create image and save
    im = draw_bars(data, width, height)
    # im = draw_circles(data, width, height)
    im.save(str(output_path / file_path.parts[-1][:-4]) + '.png')


def get_samples(data, num_samples):
    decimation = math.floor(len(data) / num_samples)

    samples = np.empty([0, 0])

    for i in range(num_samples):
        samples = np.append(samples, data[i * decimation][0])

    # shift up
    samples = samples + abs(min(samples))
    # scale
    samples = samples * (360 / max(samples))

    return samples


def get_colors(sound_samples):
    # TODO: add other modes for colors
    # get colors
    # filtered_samples = scipy.signal.medfilt(sound_samples)
    filtered_samples = sound_samples

    color_list = []
    for sample in filtered_samples:
        color_list.append((int(sample), 100, 100))

    return color_list


def draw_bars(data, width, height):
    samples = get_samples(data, width)
    colors = get_colors(samples)
    im = Image.new('HSV', (width, 1))
    im.putdata(colors)
    im = im.resize((width, height))
    im = im.convert('RGB')
    return im


def draw_circles(data, width, height):

    scale = 2
    line_thickness = 10

    width = scale * width
    height = scale * height

    radius = math.floor(min([width, height]) / 2)  # for non square images

    samples = get_samples(data, radius // line_thickness)
    colors = get_colors(samples)

    im = np.zeros((width, height, 3), np.uint8)
    # center = (int(height/2), int(width/2))
    center = (width // 2, height // 2)

    # TODO: fix issue with rectangular images
    for i in range(len(colors)):
        cv2.circle(im, center, i*line_thickness, color=colors[i], thickness=line_thickness)

    im = Image.fromarray(im, mode='HSV')
    im = im.resize((width // scale, height // scale), Image.ANTIALIAS)
    im = im.convert('RGBA')

    # make black pixels transparent
    pixels = im.getdata()
    new_pixels = []
    for pixel in pixels:
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
            new_pixels.append((0, 0, 0, 0))
        else:
            new_pixels.append(pixel)

    im.putdata(new_pixels)

    return im


if __name__ == '__main__':
    args = get_arguments()
    # output_path = Path('output')
    # generate_image(Path(args.file), output_path, args.size[0], args.size[1])
