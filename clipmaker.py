import os, cv2, toml, tempfile, subprocess, pathlib, traceback, json
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip
from pydub import AudioSegment
import whisper
from datetime import datetime
from typing import Optional
import requests

CFG = toml.load("config.toml")
OUT = "output"
os.makedirs(OUT, exist_ok=True)
LOGFILE = os.path.join(OUT, "log.txt")

def write_log(msg: str):
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def detect_audio_peaks(path: str):
    audio = AudioSegment.from_file(str(path))
    return {i//1000 for i in range(0,len(audio),1000)
            if audio[i:i+1000].dBFS > CFG["AUDIO_THRESHOLD"]}

def detect_motion_peaks(path: str):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, prev = cap.read()
    idx, peaks = 0, set()
    while ret:
        ret, frame = cap.read()
        if not ret: break
        g1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.mean(cv2.absdiff(g1,g2)) > CFG["FRAME_DIFF_THRESHOLD"]:
            peaks.add(int(idx//fps))
        prev, idx = frame, idx+1
    cap.release()
    return peaks

def extract_clip(src, start, end, name):
    clip = VideoFileClip(str(src)).subclip(start, end)
    out = pathlib.Path(OUT, f"clip_{name}.mp4")
    clip.write_videofile(str(out), codec="libx264", audio_codec="aac", logger=None)
    return str(out)

def add_subtitles(video_path, model, *, log=write_log):
    vp = pathlib.Path(video_path)
    base = vp.stem
    srt = pathlib.Path(OUT, f"{base}.srt")
    wav = pathlib.Path(OUT, f"{base}.wav")
    sub_mp4 = pathlib.Path(OUT, f"{base}_sub.mp4")

    subprocess.run(["ffmpeg","-y","-i",str(vp),str(wav)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    segments = model.transcribe(str(wav), fp16=False, language=CFG["LANG"])["segments"]
    def fmt(t):
        h,m = divmod(int(t),3600)
        m,s = divmod(m,60)
        ms = int((t-int(t))*1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    with open(srt,"w",encoding="utf-8") as f:
        for i,seg in enumerate(segments,1):
            f.write(f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n{seg['text'].strip()}\n\n")
    vf = f"subtitles='{srt.as_posix()}:charenc=UTF-8'"
    subprocess.run(["ffmpeg","-y","-i",str(vp),"-vf",vf,str(sub_mp4)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not sub_mp4.exists():
        raise RuntimeError("FFmpeg не создал видео с субтитрами")
    wav.unlink(missing_ok=True); srt.unlink(missing_ok=True)
    return str(sub_mp4)

def ask_openrouter(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {CFG['OPENROUTER_KEY']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model":"mistralai/mistral-7b-instruct:free",
        "messages":[{"role":"user","content":prompt}],
        "max_tokens":1500,"temperature":0.7
    }
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                         headers=headers,json=payload,timeout=60)
    if resp.status_code!=200:
        raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.text}")
    data = resp.json()
    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"No choices: {data}")
    return data["choices"][0]["message"]["content"].strip()

def generate_descriptions_json(sub_map: dict, log=write_log) -> dict:
    """
    sub_map: {'Клип 1': subtitle_text_or_empty, ...}
    """
    prompt = (
        "Ты — маркетолог и эксперт по TikTok.\n"
        "Для каждого клипа:\n"
        "- Если есть субтитры, придумай 1-2 предложения + 2–4 хештега.\n"
        "- Если субтитров нет, предложи 3–5 популярных хештегов.\n"
        "Верни строго JSON вида:\n"
        "{\n  \"Клип 1\": \"...\",\n  \"Клип 2\": \"...\"\n}\n\n"
        "Субтитры:\n"
    )
    for k, txt in sub_map.items():
        safe = txt.replace("“","\"").replace("”","\"")[:1000]
        prompt += f"{k}: {safe}\n\n"

    resp = ask_openrouter(prompt)
    clean = resp.replace("“","\"").replace("”","\"")
    try:
        data = json.loads(clean)
    except:
        data = {}
        for line in clean.splitlines():
            if line.startswith("Клип"):
                key,val = line.split(":",1)
                data[key.strip()] = val.strip().strip('",')

    # сохраняем каждый description_i.txt как строку
    for key, desc in data.items():
        idx = int(key.split()[1]) - 1
        text = desc if isinstance(desc,str) else json.dumps(desc, ensure_ascii=False)
        with open(os.path.join(OUT, f"description_{idx}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

    return data

def main(video: str,
         extra: Optional[str]=None,
         clip_count: int=3,
         clip_duration: int=20,
         use_subtitles: bool=True,
         log=write_log):
    # очистка
    for f in os.listdir(OUT):
        if f!="log.txt": os.remove(os.path.join(OUT,f))

    try:
        log(f"▶ Старт: {video}")
        peaks = sorted(detect_audio_peaks(video) | detect_motion_peaks(video))
        sel,last = [], -9999
        for s in peaks:
            if s-last>=10:
                sel.append(s); last=s
        if not sel:
            log("⚠ Пиков не найдено"); return

        log(f"Найдено {len(sel)} пиков, берём {clip_count}")
        model = whisper.load_model("base")
        sub_map = {}
        for i,sec in enumerate(sel[:clip_count],1):
            start = max(0,sec-1); end = start+clip_duration
            log(f"→ Клип {i}: {start}s–{end}s")
            raw = extract_clip(video, start, end, f"{i-1}_raw")
            final = add_subtitles(raw, model) if use_subtitles else raw
            if use_subtitles:
                srt = pathlib.Path(final).with_suffix(".srt")
                sub_map[f"Клип {i}"] = srt.read_text(encoding="utf-8") if srt.exists() else ""
            else:
                sub_map[f"Клип {i}"] = ""
    except Exception as e:
        log("❌ "+str(e))
        log(traceback.format_exc())
