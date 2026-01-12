import torch
import soundfile as sf
import os
import sys
import tempfile
import argparse
import threading
import queue
import unicodedata
import re

project_root = os.path.dirname(os.path.abspath(__file__))
local_repo_root = os.path.join(project_root, "ChatTTS")
if os.path.isdir(os.path.join(local_repo_root, "ChatTTS")) and local_repo_root not in sys.path:
    sys.path.insert(0, local_repo_root)

import ChatTTS
from tools.seeder import TorchSeedContext

DEFAULT_TEXT = "Bonjour, ceci est un test de synthèse vocale en français avec ChatTTS."

VOICE_PRESETS = {
    "默认": 2,
    "男声1": 2222,
    "女声1": 1111,
    "男声2": 4444,
    "女声2": 3333,
    "随机": None,
}

_chat = None


def _get_chat():
    global _chat
    if _chat is not None:
        return _chat
    chat = ChatTTS.Chat()
    chat.load(compile=False)
    _chat = chat
    return _chat


def _sample_speaker_embedding(seed):
    chat = _get_chat()
    if seed is None:
        return chat.sample_random_speaker()
    with TorchSeedContext(int(seed)):
        return chat.sample_random_speaker()


def _normalize_french_text(text: str) -> str:
    text = (
        text.replace("œ", "oe")
        .replace("Œ", "OE")
        .replace("æ", "ae")
        .replace("Æ", "AE")
    )
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def _french_pronunciation_hint(text: str) -> str:
    t = text
    rules = [
        (r"(?i)\beau\b", "oh"),
        (r"(?i)eau", "oh"),
        (r"(?i)au", "oh"),
        (r"(?i)oi", "wah"),
        (r"(?i)ou", "oo"),
        (r"(?i)gn", "ny"),
        (r"(?i)\bje\b", "zhuh"),
        (r"(?i)\bj'", "zh'"),
        (r"(?i)\bj", "zh"),
        (r"(?i)ch", "sh"),
        (r"(?i)qu", "k"),
        (r"(?i)\bce\b", "suh"),
        (r"(?i)\bci\b", "see"),
        (r"(?i)\bde\b", "duh"),
        (r"(?i)\bdes\b", "day"),
        (r"(?i)\ben\b", "ahn"),
        (r"(?i)\bun\b", "uhn"),
    ]
    for pat, rep in rules:
        t = re.sub(pat, rep, t)
    return t


def _synthesize_to_file(
    text,
    voice_name,
    speed,
    normalize_french: bool = True,
    pronunciation_hint: bool = False,
):
    chat = _get_chat()
    seed = VOICE_PRESETS.get(voice_name, 2)
    spk_emb = _sample_speaker_embedding(seed)
    if normalize_french:
        text = _normalize_french_text(text)
    if pronunciation_hint:
        text = _french_pronunciation_hint(text)
    speed = int(speed)
    if speed < 1:
        speed = 1
    if speed > 10:
        speed = 10
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        prompt=f"[speed_{speed}]",
        spk_emb=spk_emb,
    )
    wavs = chat.infer(text, skip_refine_text=True, params_infer_code=params_infer_code)
    if not wavs:
        raise RuntimeError("No audio generated.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    sf.write(tmp.name, wavs[0], 24000)
    return tmp.name


def gui():
    import tkinter as tk
    from tkinter import ttk
    import winsound

    root = tk.Tk()
    root.title("ChatTTS 离线 TTS")
    root.minsize(720, 420)

    main_frame = ttk.Frame(root, padding=12)
    main_frame.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)

    text_label = ttk.Label(main_frame, text="文本")
    text_label.grid(row=0, column=0, sticky="w")

    text_box = tk.Text(main_frame, height=6, wrap="word")
    text_box.grid(row=1, column=0, sticky="nsew", pady=(6, 12))
    main_frame.rowconfigure(1, weight=1)
    text_box.insert("1.0", DEFAULT_TEXT)

    controls = ttk.Frame(main_frame)
    controls.grid(row=2, column=0, sticky="ew")
    controls.columnconfigure(3, weight=1)

    voice_label = ttk.Label(controls, text="语音")
    voice_label.grid(row=0, column=0, sticky="w")
    voice_var = tk.StringVar(value="默认")
    voice_combo = ttk.Combobox(
        controls,
        textvariable=voice_var,
        state="readonly",
        values=list(VOICE_PRESETS.keys()),
        width=12,
    )
    voice_combo.grid(row=0, column=1, padx=(8, 18), sticky="w")

    speed_label = ttk.Label(controls, text="语速")
    speed_label.grid(row=0, column=2, sticky="w")
    speed_var = tk.IntVar(value=5)
    speed_scale = ttk.Scale(
        controls,
        from_=1,
        to=10,
        orient="horizontal",
        command=lambda v: speed_var.set(int(float(v) + 0.5)),
    )
    speed_scale.set(5)
    speed_scale.grid(row=0, column=3, sticky="ew", padx=(8, 8))
    speed_value = ttk.Label(controls, textvariable=speed_var, width=3)
    speed_value.grid(row=0, column=4, sticky="e")

    normalize_var = tk.BooleanVar(value=True)
    normalize_checkbox = ttk.Checkbutton(
        controls,
        text="法语字符规范化（去除重音）",
        variable=normalize_var,
    )
    normalize_checkbox.grid(row=1, column=0, columnspan=5, sticky="w", pady=(10, 0))

    hint_var = tk.BooleanVar(value=False)
    hint_checkbox = ttk.Checkbutton(
        controls,
        text="法语发音辅助（实验）",
        variable=hint_var,
    )
    hint_checkbox.grid(row=2, column=0, columnspan=5, sticky="w", pady=(6, 0))

    actions = ttk.Frame(main_frame)
    actions.grid(row=3, column=0, sticky="ew", pady=(12, 0))
    actions.columnconfigure(1, weight=1)

    status_var = tk.StringVar(value="就绪")
    status_label = ttk.Label(actions, textvariable=status_var)
    status_label.grid(row=0, column=0, sticky="w")

    play_btn = ttk.Button(actions, text="Play")
    play_btn.grid(row=0, column=2, sticky="e")

    result_queue = queue.Queue()
    last_audio_path = {"path": None}

    def run_synthesis():
        try:
            text = text_box.get("1.0", "end").strip()
            voice = voice_var.get()
            speed = speed_var.get()
            normalize_french = bool(normalize_var.get())
            pronunciation_hint = bool(hint_var.get())
            wav_path = _synthesize_to_file(
                text,
                voice,
                speed,
                normalize_french=normalize_french,
                pronunciation_hint=pronunciation_hint,
            )
            result_queue.put(("ok", wav_path))
        except Exception as e:
            result_queue.put(("err", str(e)))

    def poll_queue():
        try:
            kind, payload = result_queue.get_nowait()
        except queue.Empty:
            root.after(100, poll_queue)
            return

        play_btn.state(["!disabled"])
        if kind == "ok":
            last_audio_path["path"] = payload
            status_var.set("播放中…")
            winsound.PlaySound(payload, winsound.SND_FILENAME | winsound.SND_ASYNC)
            status_var.set("就绪")
        else:
            status_var.set(f"错误: {payload}")

        root.after(100, poll_queue)

    def on_play():
        play_btn.state(["disabled"])
        status_var.set("生成中…")
        t = threading.Thread(target=run_synthesis, daemon=True)
        t.start()

    def on_close():
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
        root.destroy()

    play_btn.configure(command=on_play)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, poll_queue)
    root.mainloop()


def main():
    print("----------------------------------------------------------------")
    print("ChatTTS French Demo Initialization")
    print("----------------------------------------------------------------")
    
    # Check GPU
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA is NOT available. Running on CPU (this will be slow).")
    
    print("\nInitializing ChatTTS model...")
    chat = _get_chat()
    
    print("Loading models (this may download files from HuggingFace on first run)...")
    try:
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    text_french = DEFAULT_TEXT
    
    print(f"\nGenerating audio for text: '{text_french}'")
    print("Note: ChatTTS is optimized for English/Chinese. French quality may vary.")
    
    try:
        wavs = chat.infer([text_french])
        
        if wavs and len(wavs) > 0:
            output_file = "output_french.wav"
            sf.write(output_file, wavs[0], 24000)
            print(f"\nSuccess! Audio saved to: {os.path.abspath(output_file)}")
        else:
            print("No audio generated.")
            
    except Exception as e:
        print(f"Error during inference: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.gui:
        gui()
    else:
        main()
