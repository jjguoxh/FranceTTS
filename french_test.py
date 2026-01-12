import soundfile as sf
import os
import sys
import tempfile
import threading
import queue
import unicodedata
import re

import numpy as np

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

import torch

_chat = None


def _get_chat():
    global _chat
    if _chat is not None:
        return _chat
    chat = ChatTTS.Chat()
    print("Loading ChatTTS model...")
    # Try to use CUDA if available
    if torch.cuda.is_available():
        print(f"CUDA is available. Loading on {torch.cuda.get_device_name(0)}")
        # ChatTTS load might handle device automatically, but we can be explicit or rely on it.
        # compile=True optimizes speed but takes time to startup.
        # Let's try compile=False first but ensure it uses GPU.
        # Check ChatTTS source: it usually moves to device if we don't specify, or we can move manually.
        chat.load(compile=False) 
    else:
        print("CUDA not available. Loading on CPU (slow).")
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
    progress_callback=None,
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
    chunks = _split_text_for_tts(text)
    if not chunks:
        raise RuntimeError("Empty text.")
    
    # Process chunks one by one to update progress
    wavs = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i + 1, total_chunks)
        
        # infer returns a list of wavs (usually 1 per input text)
        chunk_wavs = chat.infer(
            [chunk],
            skip_refine_text=True,
            params_infer_code=params_infer_code,
            split_text=False,
        )
        if chunk_wavs and len(chunk_wavs) > 0:
            wavs.append(chunk_wavs[0])
            
    if not wavs:
        raise RuntimeError("No audio generated.")
    
    if len(wavs) > 1:
        silence = np.zeros(int(0.12 * 24000), dtype=np.float32)
        merged = []
        for i, w in enumerate(wavs):
            if w is None or len(w) == 0:
                continue
            merged.append(w.astype(np.float32, copy=False))
            if i != len(wavs) - 1:
                merged.append(silence)
        if not merged:
            raise RuntimeError("No audio generated.")
        wav = np.concatenate(merged)
    else:
        wav = wavs[0]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    sf.write(tmp.name, wav, 24000)
    return tmp.name


def _split_text_for_tts(text: str, max_chars: int = 260):
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # Split into paragraphs (separated by 2+ newlines)
    parts = [p.strip() for p in re.split(r"\n{2,}", t) if p.strip()]
    
    cleaned_parts = []
    for p in parts:
        # Within a paragraph, replace single newlines with spaces to avoid broken sentences
        p = p.replace("\n", " ")
        # Collapse multiple spaces
        p = re.sub(r"[ \t]+", " ", p)
        if p:
            cleaned_parts.append(p)
    parts = cleaned_parts

    sentences = []
    for p in parts:
        ss = re.split(r"(?<=[\.\!\?\:\;])\s+", p)
        for s in ss:
            s = s.strip()
            if s:
                sentences.append(s)
    if not sentences:
        sentences = [t.strip()] if t.strip() else []
    chunks = []
    buf = ""
    for s in sentences:
        if not buf:
            buf = s
            continue
        if len(buf) + 1 + len(s) <= max_chars:
            buf = f"{buf} {s}"
        else:
            chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    final_chunks = []
    for c in chunks:
        if len(c) <= max_chars:
            final_chunks.append(c)
            continue
        for i in range(0, len(c), max_chars):
            seg = c[i : i + max_chars].strip()
            if seg:
                final_chunks.append(seg)
    return final_chunks


def gui():
    import tkinter as tk
    from tkinter import ttk
    from tkinter import filedialog, messagebox
    from tkinter.scrolledtext import ScrolledText
    import winsound

    root = tk.Tk()
    root.title("ChatTTS 离线 TTS")
    root.minsize(720, 420)

    main_frame = ttk.Frame(root, padding=12)
    main_frame.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)

    notebook = ttk.Notebook(main_frame)
    notebook.grid(row=0, column=0, sticky="nsew")
    main_frame.rowconfigure(0, weight=1)

    text_tab = ttk.Frame(notebook, padding=8)
    notebook.add(text_tab, text="输入文本")
    text_tab.columnconfigure(0, weight=1)
    text_tab.rowconfigure(1, weight=1)

    text_label = ttk.Label(text_tab, text="文本")
    text_label.grid(row=0, column=0, sticky="w")

    text_box = tk.Text(text_tab, height=6, wrap="word")
    text_box.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
    text_box.insert("1.0", DEFAULT_TEXT)

    pdf_tab = ttk.Frame(notebook, padding=8)
    notebook.add(pdf_tab, text="PDF")
    pdf_tab.columnconfigure(0, weight=1)
    pdf_tab.rowconfigure(1, weight=1)

    pdf_state = {"doc": None, "path": None, "page_index": 0, "page_count": 0, "page_text": ""}
    pdf_img_ref = {"img": None}

    pdf_top = ttk.Frame(pdf_tab)
    pdf_top.grid(row=0, column=0, sticky="ew")
    pdf_top.columnconfigure(6, weight=1)

    page_var = tk.StringVar(value="未加载 PDF")
    import_btn = ttk.Button(pdf_top, text="导入 PDF")
    prev_btn = ttk.Button(pdf_top, text="上一页", state="disabled")
    next_btn = ttk.Button(pdf_top, text="下一页", state="disabled")
    extract_text_btn = ttk.Button(pdf_top, text="提取本页文本", state="disabled")
    page_label = ttk.Label(pdf_top, textvariable=page_var)

    import_btn.grid(row=0, column=0, sticky="w")
    prev_btn.grid(row=0, column=1, padx=(10, 0), sticky="w")
    next_btn.grid(row=0, column=2, padx=(6, 0), sticky="w")
    page_label.grid(row=0, column=3, padx=(10, 0), sticky="w")
    extract_text_btn.grid(row=0, column=4, padx=(10, 0), sticky="w")

    pdf_body = ttk.PanedWindow(pdf_tab, orient="horizontal")
    pdf_body.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

    preview_frame = ttk.Frame(pdf_body)
    text_frame = ttk.Frame(pdf_body)
    pdf_body.add(preview_frame, weight=2)
    pdf_body.add(text_frame, weight=3)

    preview_frame.rowconfigure(0, weight=1)
    preview_frame.columnconfigure(0, weight=1)
    text_frame.rowconfigure(0, weight=1)
    text_frame.columnconfigure(0, weight=1)

    canvas = tk.Canvas(preview_frame, background="white", highlightthickness=1)
    vbar = ttk.Scrollbar(preview_frame, orient="vertical", command=canvas.yview)
    hbar = ttk.Scrollbar(preview_frame, orient="horizontal", command=canvas.xview)
    canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    vbar.grid(row=0, column=1, sticky="ns")
    hbar.grid(row=1, column=0, sticky="ew")

    page_text_view = ScrolledText(text_frame, wrap="word")
    page_text_view.grid(row=0, column=0, sticky="nsew")
    # page_text_view.configure(state="disabled")

    controls = ttk.Frame(main_frame)
    controls.grid(row=1, column=0, sticky="ew", pady=(12, 0))
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
    actions.grid(row=2, column=0, sticky="ew", pady=(12, 0))
    actions.columnconfigure(1, weight=1)

    status_var = tk.StringVar(value="就绪")
    status_label = ttk.Label(actions, textvariable=status_var)
    status_label.grid(row=0, column=0, sticky="w")

    play_btn = ttk.Button(actions, text="Play")
    play_btn.grid(row=0, column=2, sticky="e")
    
    progress_bar = ttk.Progressbar(actions, orient="horizontal", mode="determinate", length=200)
    progress_bar.grid(row=0, column=1, padx=(10, 10), sticky="ew")
    
    result_queue = queue.Queue()
    last_audio_path = {"path": None}

    def _pdf_set_text(text: str):
        # page_text_view.configure(state="normal")
        page_text_view.delete("1.0", "end")
        page_text_view.insert("1.0", text or "")
        # page_text_view.configure(state="disabled")

    def _pdf_extract_text(page):
        txt = page.get_text("text") or ""
        if txt.strip():
            return txt, {"method": "text", "text_len": len(txt), "images": None, "words": None, "blocks": None}

        blocks = None
        words = None
        try:
            blocks = page.get_text("blocks") or []
        except Exception:
            blocks = None
        try:
            words = page.get_text("words") or []
        except Exception:
            words = None

        parts = []
        if blocks:
            for b in blocks:
                if len(b) >= 5 and isinstance(b[4], str):
                    s = b[4].strip()
                    if s:
                        parts.append(s)
        if not parts and words:
            parts = [w[4] for w in words if len(w) >= 5 and isinstance(w[4], str) and w[4].strip()]

        txt2 = "\n".join(parts).strip()
        if txt2:
            return txt2, {
                "method": "blocks/words",
                "text_len": len(txt2),
                "images": None,
                "words": (len(words) if isinstance(words, list) else None),
                "blocks": (len(blocks) if isinstance(blocks, list) else None),
            }

        images = None
        try:
            images = page.get_images(full=True) or []
        except Exception:
            images = None
        return "", {
            "method": "none",
            "text_len": 0,
            "images": (len(images) if isinstance(images, list) else None),
            "words": (len(words) if isinstance(words, list) else None),
            "blocks": (len(blocks) if isinstance(blocks, list) else None),
        }

    def _pdf_render_page():
        try:
            import fitz
        except Exception:
            messagebox.showerror("缺少依赖", "请先安装 pymupdf：pip install pymupdf -i https://pypi.org/simple")
            return

        doc = pdf_state["doc"]
        if doc is None:
            return
        idx = pdf_state["page_index"]
        page = doc.load_page(idx)

        txt, info = _pdf_extract_text(page)
        pdf_state["page_text"] = txt
        pdf_state["page_diag"] = info
        _pdf_set_text(txt)

        zoom = 1.5
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        ppm_path = pdf_state.get("ppm_path")
        if not ppm_path:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ppm")
            ppm_path = tmp.name
            tmp.close()
            pdf_state["ppm_path"] = ppm_path
        with open(ppm_path, "wb") as f:
            f.write(pix.tobytes("ppm"))
        img = tk.PhotoImage(file=ppm_path)
        pdf_img_ref["img"] = img

        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=img)
        canvas.configure(scrollregion=(0, 0, img.width(), img.height()))

        page_var.set(f"第 {idx + 1} / {pdf_state['page_count']} 页")
        prev_btn.configure(state=("normal" if idx > 0 else "disabled"))
        next_btn.configure(state=("normal" if idx < pdf_state["page_count"] - 1 else "disabled"))
        extract_text_btn.configure(state="normal")

    def _pdf_open(path: str):
        try:
            import fitz
        except Exception:
            messagebox.showerror("缺少依赖", "请先安装 pymupdf：pip install pymupdf -i https://pypi.org/simple")
            return
        try:
            doc = fitz.open(path)
        except Exception as e:
            messagebox.showerror("打开失败", str(e))
            return
        pdf_state["doc"] = doc
        pdf_state["path"] = path
        pdf_state["page_index"] = 0
        pdf_state["page_count"] = doc.page_count
        _pdf_render_page()

    def on_import_pdf():
        path = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if not path:
            return
        _pdf_open(path)

    def on_extract_page_text():
        if pdf_state["doc"] is None:
            messagebox.showwarning("提示", "请先导入 PDF")
            return
        current_text = page_text_view.get("1.0", "end").strip()
        last_extracted = (pdf_state.get("page_text") or "").strip()
        if current_text and last_extracted and current_text != last_extracted:
            ok = messagebox.askyesno("确认覆盖", "检测到你已编辑文本，重新提取将覆盖，是否继续？")
            if not ok:
                return
        try:
            doc = pdf_state["doc"]
            idx = pdf_state["page_index"]
            page = doc.load_page(idx)
            txt, info = _pdf_extract_text(page)
        except Exception as e:
            messagebox.showerror("提取失败", str(e))
            return
        pdf_state["page_text"] = txt
        pdf_state["page_diag"] = info
        _pdf_set_text(txt)
        if not txt.strip():
            images = info.get("images")
            blocks = info.get("blocks")
            words = info.get("words")
            extra = []
            if images is not None:
                extra.append(f"图片数={images}")
            if blocks is not None:
                extra.append(f"blocks={blocks}")
            if words is not None:
                extra.append(f"words={words}")
            extra_s = ("，" + "，".join(extra)) if extra else ""
            messagebox.showinfo("未检测到文本层", f"当前页未提取到文本，可能是扫描版 PDF（只有图片，没有文字层）{extra_s}")

    def on_prev_page():
        if pdf_state["doc"] is None:
            return
        if pdf_state["page_index"] <= 0:
            return
        pdf_state["page_index"] -= 1
        _pdf_render_page()

    def on_next_page():
        if pdf_state["doc"] is None:
            return
        if pdf_state["page_index"] >= pdf_state["page_count"] - 1:
            return
        pdf_state["page_index"] += 1
        _pdf_render_page()

    import_btn.configure(command=on_import_pdf)
    prev_btn.configure(command=on_prev_page)
    next_btn.configure(command=on_next_page)
    extract_text_btn.configure(command=on_extract_page_text)

    def _get_active_text_for_play():
        current = notebook.select()
        if current == str(pdf_tab):
            if pdf_state["doc"] is None:
                return None, "请先导入 PDF"
            t = page_text_view.get("1.0", "end").strip()
            if not t:
                info = pdf_state.get("page_diag") or {}
                images = info.get("images")
                blocks = info.get("blocks")
                words = info.get("words")
                if images and (blocks == 0 or blocks is None) and (words == 0 or words is None):
                    return None, "本页未检测到文本层（可能是扫描版 PDF）"
                return None, "本页未提取到可读文本"
            return t, None
        t = text_box.get("1.0", "end").strip()
        if not t:
            return None, "请输入文本"
        return t, None

    def run_synthesis():
        try:
            text, err = _get_active_text_for_play()
            if err is not None:
                result_queue.put(("err", err))
                return
            voice = voice_var.get()
            speed = speed_var.get()
            normalize_french = bool(normalize_var.get())
            pronunciation_hint = bool(hint_var.get())
            
            def on_progress(current, total):
                result_queue.put(("progress", (current, total)))
                
            wav_path = _synthesize_to_file(
                text,
                voice,
                speed,
                normalize_french=normalize_french,
                pronunciation_hint=pronunciation_hint,
                progress_callback=on_progress,
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
        
        if kind == "progress":
            current, total = payload
            status_var.set(f"生成中 ({current}/{total})...")
            progress_bar["maximum"] = total
            progress_bar["value"] = current
            # Keep play button disabled while progressing
            play_btn.state(["disabled"])
            # Re-schedule poll immediately to catch up
            root.after(10, poll_queue)
            return
            
        if kind == "ok":
            last_audio_path["path"] = payload
            status_var.set("播放中…")
            progress_bar["value"] = 0
            winsound.PlaySound(payload, winsound.SND_FILENAME | winsound.SND_ASYNC)
            status_var.set("就绪")
        else:
            status_var.set(f"错误: {payload}")
            progress_bar["value"] = 0

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
        try:
            doc = pdf_state.get("doc")
            if doc is not None:
                doc.close()
        except Exception:
            pass
        try:
            ppm_path = pdf_state.get("ppm_path")
            if ppm_path:
                os.unlink(ppm_path)
        except Exception:
            pass
        root.destroy()

    play_btn.configure(command=on_play)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, poll_queue)
    root.mainloop()


if __name__ == "__main__":
    gui()
