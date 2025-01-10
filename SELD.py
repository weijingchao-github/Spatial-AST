import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)

import copy
import csv
import threading
import time
from collections import deque

import numpy as np
import rospy
import scipy
import torch
import torch.nn.functional as F
from audio_common_msgs.msg import AudioDataStamped
from spatial_ast import build_AST


def make_index_label_dict(label_csv):
    index_label_lookup_dict = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_label_lookup_dict[str(line_count)] = row["display_name"]
            line_count += 1
    return index_label_lookup_dict


class SpatialAST:
    def __init__(self):
        # preparation
        self.device = torch.device("cuda")
        self.buffer_max_lenth = 100
        self.left_channel_audio_buffer = deque(maxlen=self.buffer_max_lenth)
        self.right_channel_audio_buffer = deque(maxlen=self.buffer_max_lenth)
        label_csv_path = os.path.join(
            os.path.dirname(__file__),
            "AudioSet/metadata/class_labels_indices_subset.csv",
        )
        self.index_label_lookup_dict = make_index_label_dict(label_csv_path)
        self.classifier_threshold = 0.5
        # model init
        self.model = build_AST(
            num_classes=355, drop_path_rate=0.1, num_cls_tokens=3
        )  # num of params (M): 85.96
        checkpoint = torch.load(
            os.path.join(os.path.dirname(__file__), "weights/finetuned.pth"),
            map_location="cpu",
        )["model"]
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.to(self.device)
        self.model.eval()
        # ROS init
        rospy.Subscriber(
            "/audio/audio_stamped",
            AudioDataStamped,
            self._process_audio_msg,
            queue_size=10,
        )
        # loop
        self.thread_running = True
        self.loop_thread = threading.Thread(target=self._detect)
        self.loop_thread.start()

    def _process_audio_msg(self, audio_stamped_msg: AudioDataStamped):
        audio_data = np.frombuffer(
            audio_stamped_msg.audio.data, dtype=np.int16
        )  # 之所以是np.int16，因为音频数据是双字节的
        # 确保音频数据是双声道
        assert len(audio_data) % 2 == 0
        left_channel_audio = audio_data[::2]
        right_channel_audio = audio_data[1::2]
        num_samples = int(len(left_channel_audio) * 32000 / 48000)
        downsampled_left_channel_audio = (
            scipy.signal.resample(left_channel_audio, num_samples)
            .astype(np.int16)
            .tolist()
        )
        downsampled_right_channel_audio = (
            scipy.signal.resample(right_channel_audio, num_samples)
            .astype(np.int16)
            .tolist()
        )
        self.left_channel_audio_buffer.append(downsampled_left_channel_audio)
        self.right_channel_audio_buffer.append(downsampled_right_channel_audio)

    def _detect(self):
        while self.thread_running:
            if (
                len(self.left_channel_audio_buffer) < self.buffer_max_lenth
                and len(self.right_channel_audio_buffer) < self.buffer_max_lenth
            ):
                time.sleep(0.1)
                continue

            # audio process
            left_channel_waveform = np.concatenate(
                list(self.left_channel_audio_buffer), dtype=np.float64
            )
            right_channel_waveform = np.concatenate(
                list(self.right_channel_audio_buffer), dtype=np.float64
            )
            # normalize audio
            left_channel_waveform /= 32768
            right_channel_waveform /= 32768
            # numpy to tensor
            left_channel_waveform = (
                torch.from_numpy(left_channel_waveform).reshape(1, -1).float()
            )
            right_channel_waveform = (
                torch.from_numpy(right_channel_waveform).reshape(1, -1).float()
            )
            # Pad audio samples into 10s long
            left_channel_waveform_padding = 32000 * 10 - left_channel_waveform.shape[1]
            if left_channel_waveform_padding > 0:
                left_channel_waveform = F.pad(
                    left_channel_waveform,
                    (0, left_channel_waveform_padding),
                    "constant",
                    0,
                )
            elif left_channel_waveform_padding < 0:
                left_channel_waveform = left_channel_waveform[:, : 32000 * 10]
            right_channel_waveform_padding = (
                32000 * 10 - right_channel_waveform.shape[1]
            )
            if right_channel_waveform_padding > 0:
                right_channel_waveform = F.pad(
                    right_channel_waveform,
                    (0, right_channel_waveform_padding),
                    "constant",
                    0,
                )
            elif right_channel_waveform_padding < 0:
                right_channel_waveform = right_channel_waveform[:, : 32000 * 10]
            # Compose double channel audio
            double_channel_waveform = torch.vstack(
                (left_channel_waveform, right_channel_waveform)
            )
            # import soundfile as sf

            # sf.write(
            #     os.path.join(os.path.dirname(__file__), "double_channel_waveform.wav"),
            #     double_channel_waveform.T,
            #     32000,
            #     subtype="DOUBLE",
            # )
            # import sys

            # sys.exit(0)
            # inference
            with torch.no_grad():
                double_channel_waveform = double_channel_waveform.unsqueeze(0).to(
                    self.device
                )
                # print(double_channel_waveform.cpu().numpy().tolist()[0][0][0:6400])
                classifier, distance, azimuth, elevation = self.model(
                    double_channel_waveform
                )
                classifier = torch.sigmoid(classifier).squeeze(0).cpu().numpy()
                distance = torch.softmax(distance, dim=1).squeeze(0).cpu().numpy()
                azimuth = torch.softmax(azimuth, dim=1).squeeze(0).cpu().numpy()
                elevation = torch.softmax(elevation, dim=1).squeeze(0).cpu().numpy()
            classifier_result_array = np.where(classifier > self.classifier_threshold)[
                0
            ]
            distance_result = np.argmax(distance) * 0.5
            azimuth_result = np.argmax(azimuth)
            elevation_result = np.argmax(elevation)
            if len(classifier_result_array) != 0:
                for classifier_result in classifier_result_array:
                    sigmoid_value = classifier[classifier_result]
                    label = self.index_label_lookup_dict[str(classifier_result)]
                    print(f"sigmoid_value: {sigmoid_value}, label: {label}")
                print(f"distance: {distance_result}")
                print(f"azimuth: {azimuth_result}")
                print(f"elevation: {elevation_result}")
            else:
                print("No sound event detected.")
                max_sigmoid_value = np.max(classifier)
                print(f"max sigmoid_value: {max_sigmoid_value}")
            print("********************************")

            time.sleep(5)

    def thread_shutdown(self):
        self.thread_running = False
        self.loop_thread.join()


def main():
    rospy.init_node("binaural_spatial_sound_perception")
    seld_model = SpatialAST()
    try:
        rospy.spin()
    finally:
        seld_model.thread_shutdown()


if __name__ == "__main__":
    main()
