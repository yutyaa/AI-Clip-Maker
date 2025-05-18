import threading, queue, pathlib, os, subprocess, tempfile
from tkinter import (
    Tk, Button, Label, filedialog, StringVar, Checkbutton, BooleanVar,
    Entry, IntVar, Frame
)
import platform
import clipmaker
import whisper

OUT_DIR = pathlib.Path("output")
OUT_DIR.mkdir(exist_ok=True)

def open_file(path):
    p = os.path.abspath(path)
    if platform.system()=="Windows":
        os.startfile(p)
    elif platform.system()=="Darwin":
        subprocess.run(["open",p])
    else:
        subprocess.run(["xdg-open",p])

class App:
    def __init__(self, root: Tk):
        self.root = root
        root.title("AI Clip Maker")
        root.geometry("600x560")

        self.video_path = None
        self.running = False
        self.log_q = queue.Queue()
        self.whisper_model = None
        self.result_clips = []

        Button(root, text="Выбрать видео", width=18, command=self.choose_video).pack(pady=4)

        self.sub_en = BooleanVar(value=True)
        Checkbutton(root, text="Добавить субтитры", var=self.sub_en).pack(anchor="w", padx=10)

        Label(root, text="Сколько клипов:").pack(anchor="w", padx=10)
        self.clip_count = IntVar(value=3)
        Entry(root, textvariable=self.clip_count, width=6).pack(anchor="w", padx=10)

        Label(root, text="Длительность (сек):").pack(anchor="w", padx=10)
        self.clip_dur = IntVar(value=20)
        Entry(root, textvariable=self.clip_dur, width=6).pack(anchor="w", padx=10)

        self.btn_start = Button(root, text="Старт", width=16, state="disabled", command=self.start)
        self.btn_start.pack(pady=6)

        self.clips_frame = Frame(root)

        self.btn_desc = Button(root, text="Сгенерировать описания",
                               state="disabled", command=self.gen_all)
        self.btn_desc.pack(pady=4)

        Button(root, text="Открыть output/", command=lambda: open_file(OUT_DIR)).pack(pady=4)

        self.status = StringVar(value="Ожидание выбора видео…")
        Label(root, textvariable=self.status, justify="left", anchor="w",
              wraplength=580, relief="sunken", bg="#fafafa")\
            .pack(fill="x", padx=8, pady=4)

        root.after(200, self.poll_log)

    def choose_video(self):
        p = filedialog.askopenfilename(filetypes=[("Видео","*.mp4;*.mov;*.mkv;*.webm")])
        if p:
            self.video_path = pathlib.Path(p)
            self.status.set(f"Выбрано: {self.video_path.name}")
            self.btn_start.config(state="normal")

    def start(self):
        if not self.video_path or self.running:
            return
        self.running = True
        self.btn_start.config(state="disabled")
        self.btn_desc.config(state="disabled")
        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def run_pipeline(self):
        clipmaker.main(
            str(self.video_path), None,
            clip_count=self.clip_count.get(),
            clip_duration=self.clip_dur.get(),
            use_subtitles=self.sub_en.get(),
            log=self.log_q.put
        )
        self.running = False
        self.btn_start.config(state="normal")
        self.show_result_clips()

    def show_result_clips(self):
        for w in self.clips_frame.winfo_children():
            w.destroy()
        self.result_clips = sorted(
            OUT_DIR.glob("clip_*_sub.mp4" if self.sub_en.get() else "clip_*_raw.mp4")
        )
        if not self.result_clips:
            self.status.set("⚠ Итоговых клипов не найдено.")
            return

        for idx, clip in enumerate(self.result_clips, start=1):
            name = f"🎬 Клип {idx}"
            desc_file = clip.with_name(clip.stem.replace("_sub","").replace("_raw","") + "_description.txt")
            if desc_file.exists():
                name += " 📝"
            btn = Button(self.clips_frame, text=name, anchor="w", width=50,
                         command=lambda p=clip: open_file(p))
            btn.pack(fill="x", pady=2)
        self.clips_frame.pack(fill="both", expand=True, padx=10)
        self.btn_desc.config(state="normal")

    def gen_all(self):
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        self.btn_desc.config(state="disabled")
        if not self.whisper_model:
            self.log_q.put("Загрузка whisper-модели…")
            self.whisper_model = whisper.load_model("tiny")

        subtitle_map = {}
        for idx, clip in enumerate(self.result_clips, start=1):
            srt = clip.with_suffix(".srt")
            if srt.exists():
                text = srt.read_text(encoding="utf-8")
            else:
                wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                subprocess.call(
                    ["ffmpeg","-y","-i",str(clip),wav],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                res = self.whisper_model.transcribe(wav, fp16=False, language=clipmaker.CFG["LANG"])
                text = " ".join(seg["text"].strip() for seg in res["segments"])
                os.remove(wav)
            subtitle_map[f"Клип {idx}"] = text.strip()

        self.log_q.put("📡 Отправляем единый запрос через api…")
        clipmaker.generate_descriptions_json(subtitle_map, self.log_q.put)

        out = "\n".join(f"{k}: {v}" for k,v in subtitle_map.items() if False)  # placeholder
        # реальные описания уже в files description_i.txt

        self.log_q.put("✅ Описания сохранены в description_i.txt")
    def poll_log(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                cur = self.status.get()
                self.status.set(cur + "\n" + msg if cur else msg)
        except queue.Empty:
            pass
        finally:
            self.root.after(200, self.poll_log)

if __name__=="__main__":
    root = Tk()
    root.iconbitmap("icon.ico")
    App(root)
    root.mainloop()
