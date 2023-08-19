# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Updated to account for UI changes from https://github.com/rkfg/audiocraft/blob/long/app.py
# also released under the MIT license.

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import random
import subprocess as sp
from tempfile import NamedTemporaryFile
import time
import typing as tp
import warnings

import numpy as np
import torch
import torchaudio
import gradio as gr

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import AudioGen, MultiBandDiffusion


MODEL = None  # Last used model
INTERRUPTING = False
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call


def _call_nostderr(*args, **kwargs):
    # Avoid ffmpeg vomiting on the logs.
    kwargs['stderr'] = sp.DEVNULL
    kwargs['stdout'] = sp.DEVNULL
    _old_call(*args, **kwargs)


sp.call = _call_nostderr
# Preallocating the pool of processes.
pool = ProcessPoolExecutor(4)
pool.__enter__()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break


file_cleaner = FileCleaner()


def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out


def load_model(version='facebook/audiogen-medium'):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        MODEL = AudioGen.get_pretrained(version)


def load_diffusion():
    global MBD
    print("loading MBD")
    MBD = MultiBandDiffusion.get_mbd_musicgen()


def _do_predictions(texts, prompt_sounds, prompt_duration, duration, progress=False, seed=4242, **gen_kwargs):
    MODEL.set_generation_params(duration=duration, **gen_kwargs)
    be = time.time()
    processed_prompt_sounds = []
    prompt_sample_rates = []
    for prompt_sound in prompt_sounds:
        if prompt_sound is None:
            processed_prompt_sounds.append(None)
            prompt_sample_rates.append(None)
        else:
            prompt_sound, prompt_sample_rate = torchaudio.load(prompt_sound)
            if prompt_sound.dim() == 1:
                prompt_sound = prompt_sound[None]
            prompt_sound = prompt_sound[..., :min(int(prompt_sample_rate * prompt_duration), prompt_sound.shape[-1] - 1)]
            processed_prompt_sounds.append(prompt_sound)
            prompt_sample_rates.append(prompt_sample_rate)

    seed_everything(seed)

    if any(m is not None for m in processed_prompt_sounds):
        if len(processed_prompt_sounds) > 1:
            print("Using only first prompt audio, multiple not supported!")
        outputs = MODEL.generate_continuation(
            prompt=processed_prompt_sounds[0],
            prompt_sample_rate=prompt_sample_rates[0],
            descriptions=texts,
            progress=progress,
        )
    else:
        outputs = MODEL.generate(texts, progress=progress)
    if USE_DIFFUSION:
        outputs_diffusion = MBD.tokens_to_wav(outputs[1])
        outputs = torch.cat([outputs[0], outputs_diffusion], dim=0)
    outputs = outputs.detach().cpu().float()
    pending_videos = []
    out_wavs = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            pending_videos.append(pool.submit(make_waveform, file.name))
            out_wavs.append(file.name)
            file_cleaner.add(file.name)
    out_videos = [pending_video.result() for pending_video in pending_videos]
    for video in out_videos:
        file_cleaner.add(video)
    print("batch finished", len(texts), time.time() - be)
    print("Tempfiles currently stored: ", len(file_cleaner.files))
    return out_videos, out_wavs



def predict_full(model, decoder, text, prompt_sound, duration, prompt_duration, topk, topp, temperature, cfg_coef, seed, progress=gr.Progress()):
    global INTERRUPTING
    global USE_DIFFUSION
    INTERRUPTING = False
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    if decoder == "MultiBand_Diffusion":
        USE_DIFFUSION = True
        load_diffusion()
    else:
        USE_DIFFUSION = False
    load_model(model)

    def _progress(generated, to_generate):
        progress((min(generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    MODEL.set_custom_progress_callback(_progress)

    videos, wavs = _do_predictions(
        [text], [prompt_sound], prompt_duration, duration, progress=True, seed=seed,
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef)
    if USE_DIFFUSION:
        return videos[0], wavs[0], videos[1], wavs[1]
    return videos[0], wavs[0], None, None


def toggle_audio_src(choice):
    if choice == "mic":
        return gr.update(source="microphone", value=None, label="Microphone")
    else:
        return gr.update(source="upload", value=None, label="File")


def toggle_diffusion(choice):
    if choice == "MultiBand_Diffusion":
        return [gr.update(visible=True)] * 2
    else:
        return [gr.update(visible=False)] * 2


def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # AudioGen
            This is your private demo for [AudioGen](https://github.com/facebookresearch/audiocraft/blob/main/docs/AUDIOGEN.md),
            a simple and controllable model for audio generation
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", interactive=True)
                    with gr.Column():
                        radio = gr.Radio(["file", "mic"], value="file",
                                         label="Condition on a prompt_sound (optional) File or Mic")
                        prompt_sound = gr.Audio(source="upload", type="filepath", label="File",
                                          interactive=True, elem_id="prompt_sound-input")
                with gr.Row():
                    submit = gr.Button("Submit")
                    # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Row():
                    model = gr.Radio(["facebook/audiogen-medium"], label="Model", value="facebook/audiogen-medium", interactive=True)
                with gr.Row():
                    decoder = gr.Radio(["Default"], label="Decoder", value="Default", interactive=False)
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=120, value=10, label="Duration", interactive=True)
                    prompt_duration = gr.Slider(minimum=0.01, maximum=10, value=1, label="Prompt Duration", interactive=True)
                with gr.Row():
                    seed = gr.Number(label="Seed", value=42, interactive=True, precision=0)
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
            with gr.Column():
                output = gr.Video(label="Generated Audio")
                audio_output = gr.Audio(label="Generated Audio (wav)", type='filepath')
        submit.click(predict_full, inputs=[model, decoder, text, prompt_sound, duration, prompt_duration, topk, topp, temperature, cfg_coef, seed], outputs=[output, audio_output])
        radio.change(toggle_audio_src, radio, [prompt_sound], queue=False, show_progress=False)

        interface.queue().launch(**launch_kwargs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    # Show the interface
    ui_full(launch_kwargs)
