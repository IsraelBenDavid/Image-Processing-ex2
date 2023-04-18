import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from imageio import imread
import skimage.color

# ############## CONSTANTS ############### #

GREATEST_VALUE = 255
GRAYSCALE = 1
CHANGE_RATE_FILENAME = "change_rate.wav"
CHANGE_SAMPLES_FILENAME = "change_samples.wav"
DX_CONV = [[0.5, 0, -0.5]]
DY_CONV = [[0.5], [0], [-0.5]]


def DFT(signal):
    """
    perform fourier transform on the signal
    :param signal: sigal vector
    :return: fourier transform of the signal
    """
    assert signal is not None
    if signal.size == 0:
        return np.array([])
    return np.matmul(_get_dft_matrix(signal.size), signal)


def IDFT(fourier_signal):
    """
    perform inverse fourier transform on the fourier_signal
    :param fourier_signal: sigal vector
    :return: inverse fourier transform of the fourier_signal
    """
    assert fourier_signal is not None
    if fourier_signal.size == 0:
        return np.array([])
    return np.matmul(_get_idft_matrix(fourier_signal.size), fourier_signal)


def _get_dft_matrix(size):  # returns DFT matrix of size "size"
    assert size > 0
    dft_matrix = np.mgrid[0:size, 0:size]  # create vector of indexes matrix
    dft_matrix = dft_matrix[0] * dft_matrix[1]  # multiply element by element
    dft_matrix = dft_matrix * (-1j * np.pi) * 2  # it's like -2 * pi * ux
    dft_matrix = dft_matrix / size  # divide by N
    dft_matrix = np.e ** dft_matrix  # e^((-2 * pi * ux)/N)
    return dft_matrix


def _get_idft_matrix(size):  # returns inverse DFT matrix of size "size"
    assert size > 0
    return np.linalg.inv(_get_dft_matrix(size))


def DFT2(image):
    """
    perform fourier transform on the image
    :param image: grayscale image matrix
    :return: fourier transform of the image
    """
    assert image is not None
    assert image.ndim == 2 or image.ndim == 3
    fourier_image = np.apply_along_axis(DFT, 0, image)
    fourier_image = np.apply_along_axis(DFT, 1, fourier_image)
    return fourier_image


def IDFT2(fourier_image):
    """
    perform inverse fourier transform on the fourier_image
    :param fourier_image: fourier 2D matrix
    :return: inverse fourier transform of the fourier_image (an image)
    """
    assert fourier_image is not None
    assert fourier_image.ndim == 2 or fourier_image.ndim == 3
    image = np.apply_along_axis(IDFT, 0, fourier_image)
    image = np.apply_along_axis(IDFT, 1, image)
    return image


def change_rate(filename, ratio):
    """
    changes the rate of the given wav
    :param filename:
    :param ratio:
    :return:
    """
    audio = scipy.io.wavfile.read(filename)
    scipy.io.wavfile.write(CHANGE_RATE_FILENAME, int(audio[0] * ratio), audio[1].astype(np.int16))


def change_samples(filename, ratio):
    """
    changes the rate of the given wav
    :param filename:
    :param ratio:
    :return:
    """
    audio = scipy.io.wavfile.read(filename)

    scipy.io.wavfile.write(CHANGE_SAMPLES_FILENAME, audio[0], resize(audio[1], ratio))


def resize(data, ratio):
    """
    resizes the data by cutting the higher frequencies
    :param data: the data
    :param ratio: ratio of resize
    :return: new data
    """
    if ratio == 1:
        return data
    fourier_data = DFT(data)
    fourier_data = np.fft.fftshift(fourier_data)

    if ratio >= 1:
        fourier_data = _resize_crop(fourier_data, ratio)
    else:
        fourier_data = _resize_add(fourier_data, ratio)

    fourier_data = np.fft.ifftshift(fourier_data)
    fourier_data = IDFT(fourier_data)
    return np.real(fourier_data).astype(np.int16)


def _resize_crop(fourier_data, ratio):
    new_size = int(fourier_data.size / ratio)
    center = int(fourier_data.size / 2)
    start = center - int(new_size / 2)
    end = start + new_size
    return fourier_data[start:end]


def _resize_add(data, ratio):
    total_add = int((data.size / ratio) - data.size)
    side_add = int(total_add / 2)
    if total_add % 2 == 0:
        return np.concatenate(([0] * side_add, data, [0] * side_add))
    return np.concatenate(([0] * side_add, data, [0] * (side_add + 1)))


def resize_spectrogram(data, ratio):
    """
    resizes the data by cutting spectrogram
    :param data: the data
    :param ratio: ratio of resize
    :return: new data
    """
    if ratio == 1:
        return data
    mat = stft(data)
    mat = np.apply_along_axis(resize, 0, mat.T, ratio)  # todo: is it ok?
    return np.real(istft(mat.T)).astype(np.int16)


def resize_vocoder(data, ratio):
    """
    resizes the data by applying phase_vocoder on the spectrogram
    :param data: the data
    :param ratio: ratio of resize
    :return: new data
    """
    if ratio == 1:
        return data
    mat = stft(data)
    mat = phase_vocoder(mat, ratio)
    return np.real(istft(mat)).astype(np.int16)


def conv_der(im):
    """
    calculates the magnitude of the image derivatives by using convolution
    :param im: the image
    :return: magnitude
    """
    dx = convolve2d(im, DX_CONV, "same")  # get the dx and cut the last two extra cols
    dy = convolve2d(im, DY_CONV, "same")  # get the dy and cut the last two extra rows
    return _magnitude_calc(dx, dy)


def _magnitude_calc(dx, dy):
    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)


def fourier_der(im):
    """
    calculates the magnitude of the image derivatives by using fourier transform
    :param im: the image
    :return: magnitude
    """
    fourier_shifted_im = np.fft.fftshift(DFT2(im))

    fourier_shifted_dx = _fourier_der_calc(fourier_shifted_im, axis=0)
    fourier_shifted_dy = _fourier_der_calc(fourier_shifted_im, axis=1)

    dx = IDFT2(np.fft.ifftshift(fourier_shifted_dx)).real
    dy = IDFT2(np.fft.ifftshift(fourier_shifted_dy)).real

    return _magnitude_calc(dx, dy)


def _fourier_der_calc(fourier_shifted_im, axis=0):
    im = [fourier_shifted_im, fourier_shifted_im.T][axis]
    n = im.shape[1]
    if im.shape[1] % 2 == 0:
        mul = np.arange(-1 * n // 2, n // 2) * 2j * np.pi
    else:
        mul = np.arange(-1 * (n // 2), (n // 2) + 1) * 2j * np.pi
    mul = mul / n
    return [im * mul, (im*mul).T][axis]


# ############## from sol1.py ############# #

def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    image = imread(filename)
    image = image.astype(np.float64)
    image /= GREATEST_VALUE
    if representation == GRAYSCALE:
        image = skimage.color.rgb2gray(image)
    return image


# ############## from ex2_helper.py ############### #

def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec

