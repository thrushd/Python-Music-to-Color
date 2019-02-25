import argparse
import numpy as np
import math
from PIL import Image, ImageFilter
from pathlib import Path
import soundfile as sf
import cv2
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import signal


def get_arguments():
    """
    Obtain and return the arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('file', type=str, help='path to the video file')
    parser.add_argument('-s', '--size', nargs=2, type=int, default=[1000, 1000], help='size of the image, in width height (1920 1080)')
    parser.add_argument('-c', '--colormap', type=str, default='viridis', help='A matplotlib colormap name')

    return parser.parse_args()


def generate_image(file_path, output_path, width, height, colormap):
    """
    Takes an audio file and generates an image.
    """
    # get audio info
    data, samplerate = sf.read(file_path)

    # create image and save
    im = draw_bars(data, width, height, colormap)
    # im = draw_circles(data, width, height)
    # im = im.filter(ImageFilter.GaussianBlur(radius=5))
    im.save(str(output_path / file_path.parts[-1][:-4]) + '_' + colormap + '.png')


def get_abs_samples(data, num_samples):
    decimation = math.floor(len(data) / num_samples)

    samples = np.empty([0, 0])

    for i in range(num_samples):
        samples = np.append(samples, data[i * decimation][0])

    # shift up
    # samples = samples + abs(min(samples))
    samples = abs(samples)
    # scale
    samples = samples * (255 / max(samples))

    return samples.astype(int)


def get_colormap(color_map_name):
    """
    Get an array of colormap RBGA tuples from a matplotlib colormap name.
    :param color_map_name: String of the supported colormap name.
    :return: A list of RBGA tuples.
    """
    colors = cm.get_cmap(color_map_name)

    color_map_array = []

    for i in range(colors.N):
        r = int(colors(i)[0]*256)
        g = int(colors(i)[1]*256)
        b = int(colors(i)[2]*256)

        color_map_array.append((r, g, b, 255))

    return color_map_array


def get_colors(sound_samples, colormap):
    # TODO: add other modes for colors
    # get colors
    color_map = get_colormap(colormap)
    color_list = []
    for sample in sound_samples:
        color_list.append(color_map[int(sample)])

    return color_list


def draw_bars(data, width, height, colormap):
    samples = get_abs_samples(data, width)
    colors = get_colors(samples, colormap)

    # plt.plot(samples)
    # plt.show()

    im = Image.new('RGBA', (width, 1))
    im.putdata(colors)
    im = im.resize((width, height))
    return im


def draw_circles(data, width, height):

    scale = 2
    line_thickness = 10

    width = scale * width
    height = scale * height

    radius = math.floor(min([width, height]) / 2)  # for non square images

    samples = get_abs_samples(data, radius // line_thickness)
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
    output_path = Path('output')
    # generate_image(Path(args.file), output_path, args.size[0], args.size[1], args.colormap)

    width = args.size[0]
    height = args.size[1]

    data, sample_rate = sf.read(Path(args.file))

    left_channel = data[:, 0]

    window_width = height*2  # sampling window

    sample_width = len(left_channel)//width
    windows = np.linspace(0, len(left_channel)-sample_width, num=width, dtype=int)

    color_map = get_colormap(args.colormap)

    i = 0
    pixels = []

    for window_start in windows:

        samples = left_channel[window_start:window_start + window_width]  # Get chunk

        sample_fft = np.fft.fft(samples)  # get fft
        mag = np.abs(sample_fft)  # get magnitudes
        mag = mag[0:height]  # only positives
        # print(type(np.max(mag)))

        max_mag = float(np.max(mag))

        if max_mag != 0.0:
            mag = mag * (255 / max_mag)  # scale

        mag = mag.astype(int)  # cast as int

        # freqs = np.fft.fftfreq(window, d=1/sample_rate)  # get frequencies
        # freqs = freqs[0:height]  # only positives

        color_list = []
        for sample in mag:
            # print(sample)
            color_list.append(color_map[int(sample)])

        pixels.extend(color_list)
        i += 1

    im = Image.new('RGBA', (height, width))
    print(len(pixels))
    print(width*height)
    im.putdata(pixels)
    im.save('spectrum.png')

    # plt.plot(freqs, mag)
    # plt.show()
    # print(len(mag))
    # print(len(freqs))
    # print('end')

    # matplotlib
    # Pxx, freqs, bins, im = plt.specgram(data[:, 0], Fs=samplerate, NFFT=4096, cmap=plt.get_cmap('magma'))
    # plt.show()

    # scipy
    # frequencies, times, spectrogram = signal.spectrogram(data[:, 0], samplerate)
    # # plt.imshow(spectrogram)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
