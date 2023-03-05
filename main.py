import numpy as np
import wave, struct  # Khong su dung
from scipy.io import wavfile
from IPython.display import Audio
import matplotlib.pyplot as plt

import librosa

FOLDER_FILE = 'D:/My_Study_Side/Nam3-ky6/KHDL/thi_giua_ky/data/'


def cach1(self):
    wav_file = wave.open(
        f'' + FOLDER_FILE + '(Silenced)_318622__matucha__roomtone_aircon_01.wav', 'rb')
    print(wav_file)
    binary_data = wav_file.readframes(wav_file.getnframes())
    print(binary_data)


def read_file_audio1(file_name, type):
    sample_rate, samples = wavfile.read(file_name)
    print(type)
    # print(samples)
    print("Tần số lấy mẫu: " + str(sample_rate))
    count = 0
    for _ in samples:
        count += 1
    print(count)


def read_file_audio2(file_name):
    # print("Librosa")
    samples, samples_rate = librosa.load(file_name)
    count = 0
    for _ in samples:
        count += 1
    # print(samples)
    # print(samples_rate)
    return samples, samples_rate, count


# Import đường dẫn file
AUDIO_FILE_SILENCE = FOLDER_FILE + "(Silenced)_318622__matucha__roomtone_aircon_01.wav"
AUDIO_FILE_SPEECH = FOLDER_FILE + "(Speech)519189__inspectorj__request-42-hmm-i-dont-know.wav"

# read_file_audio1(AUDIO_FILE_SILENCE, "SILENCE")
# read_file_audio1(AUDIO_FILE_SPEECH, "SPEECH")

sl_samples, sl_rate, sl_signal_length = read_file_audio2(AUDIO_FILE_SILENCE)
sp_samples, sp_rate, sp_signal_length = read_file_audio2(AUDIO_FILE_SPEECH)

# Chuyền sang integer - Không thành công: all mảng bằng 0
# print("SL_sample before")
# print(sl_samples)
# # sl_samples = [int(item) for item in sl_samples]
# print("SL_sample after")
# print(sl_samples)
# print("SP_sample")
# print(sp_samples)
# sp_samples = [int(item) for item in sp_samples]

# Import frame
# 1 frame : 20ms, 1s - sl_rate => frame [0:no.sl_rate/50]
# sl_tile = sl_rate // 50
# sp_tile = sp_rate // 50
#
# sl_frame = []
# sp_frame = []


# def getAllFrame(signal_length):
#     i = 0
#     frame = []
#     while (i + sl_tile) <= signal_length - 1:
#         frame.append(sl_samples[i:(i + sl_tile)])
#         i += sl_tile
#     return frame


'''
# Chia frame
sl_frame = getAllFrame(sl_signal_length)
#print(len(sl_frame))
sp_frame = getAllFrame(sp_signal_length)
#print(len(sp_frame))
'''

# Define frame size and hop length
sl_frame_size = int(sl_rate * 0.02)  # 20ms của cả 2 signal
sl_hop_length = int(sl_frame_size / 2)

sp_frame_size = int(sp_rate * 0.02)  # 20ms của cả 2 signal
sp_hop_length = int(sp_frame_size / 2)


def split_frame_and_cal_ste(data, frame_size, hop_length):
    frames = librosa.util.frame(data, frame_length=frame_size, hop_length=hop_length)

    # Calculate STE for each frame
    ste = np.sum(frames ** 2, 0) / frame_size  # số 0 có nghĩa axis =0
    return ste


sl_ste = split_frame_and_cal_ste(sl_samples, sl_frame_size, sl_hop_length)
sp_ste = split_frame_and_cal_ste(sp_samples, sp_frame_size, sp_hop_length)


# print("SL STE")
# print(min(sl_ste))
#
# print("SP STE")
# print(max(sp_ste))

def normalize_ste_frame(ste_frame):
    MIN = min(ste_frame)
    MAX = max(ste_frame)
    new_ste_frame = []
    for i in ste_frame:
        new_ste_frame.append((i - MIN) / (MAX - MIN))
    return new_ste_frame


# Split audio into frames
normalize_sl_ste = normalize_ste_frame(sl_ste)
normalize_sp_ste = normalize_ste_frame(sp_ste)

print("length Slience STE")
print(len(normalize_sl_ste))
print("length Speech STE")
print(len(normalize_sp_ste))

threshold = 0.005

sl_right_classified = []
sl_fail_classified = []
sp_right_classified = []
sp_fail_classified = []


def cal_cost_function():
    print("begin")
    cost_function = 0
    count = 0
    threshold = min_threshold = 0.0001  # Chọn threshold nhỏ nhất với cost_function lớn nhất
    min_cost = len(normalize_sl_ste) + len(normalize_sp_ste)
    while (threshold < 0.01):
        count += 1
        # print("c= " + str(count))
        for sl_ste in normalize_sl_ste:
            if sl_ste >= threshold:
                cost_function += 1
                sl_fail_classified.append(sl_ste)
            else:
                sl_right_classified.append(sl_ste)
        for sp_ste in normalize_sp_ste:
            if sp_ste >= threshold:
                sp_right_classified.append(sp_ste)
            else:
                cost_function += 1
                sp_fail_classified.append(sp_ste)
        # print("cost function: " + str(cost_function))

        if min_cost > cost_function:  # Nếu min_cost mà timf thấy số nhỏ hơn: chọn
            min_threshold = threshold
            min_cost = cost_function
            print("threshold=" + str(threshold))
            print("cost =" + str(cost_function))
            print("sl_fail" + str(len(sl_fail_classified)))
            print("sp_fail" + str(len(sp_fail_classified)))
        threshold += 0.0001

    return min_threshold


min_threshold = cal_cost_function()
print("End function")
print(min_threshold)
# Plot
plt.subplot(2, 1, 1)
plt.plot(normalize_sl_ste)  # , label="silence"
plt.subplot(2, 1, 2)
plt.plot(normalize_sp_ste)  # , label="speech"
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()
# plt.title("DEMO")
