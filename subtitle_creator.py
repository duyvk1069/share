import os
import subprocess
import whisper
import requests
import time
from dotenv import load_dotenv
from tqdm import tqdm
import sys
import shutil
import re
import math
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    print("[WARNING] Thư viện gTTS chưa được cài đặt. Chức năng tạo thuyết minh sẽ bị bỏ qua.")
    print("          Để cài đặt, chạy: pip install gTTS")
    GTTS_AVAILABLE = False

# --- Lấy đường dẫn thư mục chứa script ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Thư mục làm việc của script: {SCRIPT_DIR}")
print(f"Thư mục làm việc hiện tại (CWD): {os.getcwd()}")

# --- Config ---
# !!! THAY ĐỔI ĐƯỜNG DẪN NÀY THÀNH FILE VIDEO CỦA BẠN !!!
VIDEO_FILE = r"D:\test_make_video\video.mp4"
CROP_BOTTOM_PIXELS = 0
# ORIGINAL_AUDIO_VOLUME = 0.3 # Không cần nữa

# --- Giới hạn Atempo ---
ATEMPO_MIN = 0.5
ATEMPO_MAX = 100.0

# --- Đường dẫn File/Thư mục ---
AUDIO_FILE = os.path.join(SCRIPT_DIR, "audio_whisper.wav")
FULL_AUDIO_FILE = os.path.join(SCRIPT_DIR, "original_full_audio.mp3")
DEMUCS_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "demucs_output") # Thư mục gốc cho output Demucs
BACKGROUND_AUDIO_DEFAULT_NAME = "no_vocals.wav" # Tên file nền cần tìm

SEGMENTS_ZH_FILE = os.path.join(SCRIPT_DIR, "segments_zh.txt")
SRT_FILE = os.path.join(SCRIPT_DIR, "subtitle_vi.srt")
TTS_SEGMENTS_DIR = os.path.join(SCRIPT_DIR, "tts_audio_segments_raw")
TTS_ADJUSTED_DIR = os.path.join(SCRIPT_DIR, "tts_audio_segments_adjusted")

OUTPUT_VIDEO_FILE_SOFTSUB = os.path.join(SCRIPT_DIR, "video_output_vi_softsub.mp4")
if CROP_BOTTOM_PIXELS > 0:
    OUTPUT_VIDEO_FILE_HARDSUB = os.path.join(SCRIPT_DIR, f"video_output_vi_hardsub_cropped{CROP_BOTTOM_PIXELS}.mp4")
else:
    OUTPUT_VIDEO_FILE_HARDSUB = os.path.join(SCRIPT_DIR, "video_output_vi_hardsub.mp4")
OUTPUT_VIDEO_FILE_NARRATED = os.path.join(SCRIPT_DIR, "video_output_narrated_bg_synced_vi.mp4")


print(f"File video gốc: {VIDEO_FILE}")
print(f"Cắt bỏ pixel từ dưới lên (Hardsub): {CROP_BOTTOM_PIXELS}")
print(f"File âm thanh tạm (Whisper): {AUDIO_FILE}")
print(f"File âm thanh gốc đầy đủ (Demucs): {FULL_AUDIO_FILE}")
print(f"Thư mục output Demucs: {DEMUCS_OUTPUT_DIR}")
print(f"File segment gốc: {SEGMENTS_ZH_FILE}")
print(f"File phụ đề SRT: {SRT_FILE}")
print(f"Thư mục TTS gốc: {TTS_SEGMENTS_DIR}")
print(f"Thư mục TTS đã chỉnh tốc độ: {TTS_ADJUSTED_DIR}")
print(f"File video output (softsub): {OUTPUT_VIDEO_FILE_SOFTSUB}")
print(f"File video output (hardsub): {OUTPUT_VIDEO_FILE_HARDSUB}")
print(f"File video output (Nền + Thuyết minh Sync): {OUTPUT_VIDEO_FILE_NARRATED}")


MODEL_SIZE = "medium"
LANGUAGE_CODE = "zh"

# --- Load Gemini API Key ---
dotenv_path = os.path.join(SCRIPT_DIR, '.env')
if os.path.exists(dotenv_path): load_dotenv(dotenv_path=dotenv_path); print("Đã tải .env")
elif load_dotenv(): print("Đã tải .env (mặc định)")
else: print("[WARNING] Không tìm thấy file .env.")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY: print("[ERROR] Không tìm thấy GEMINI_API_KEY."); exit(1)


# --- Các hàm tiện ích ---
def format_timestamp(seconds):
    assert seconds >= 0, "Thời gian không thể âm"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000; milliseconds %= 3_600_000
    minutes = milliseconds // 60_000; milliseconds %= 60_000
    seconds = milliseconds // 1_000; milliseconds %= 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def get_audio_duration(file_path):
    try:
        ffprobe_executable = os.path.join(os.path.dirname(get_ffmpeg_path()), "ffprobe")
        if not os.path.exists(ffprobe_executable):
             ffprobe_executable = "ffprobe"
        command = [ffprobe_executable, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        duration_str = subprocess.check_output(command).strip().decode('utf-8')
        return float(duration_str)
    except subprocess.CalledProcessError as e: print(f"[ERROR] ffprobe lỗi khi lấy thời lượng file {file_path}: {e}"); return None
    except FileNotFoundError: print("[ERROR] Lệnh 'ffprobe' không tìm thấy."); return None
    except Exception as e: print(f"[ERROR] Lỗi không xác định khi lấy thời lượng file {file_path}: {e}"); return None

def get_ffmpeg_path():
    ffmpeg_env_path = os.getenv("FFMPEG_PATH")
    if ffmpeg_env_path and os.path.isfile(ffmpeg_env_path): return ffmpeg_env_path
    ffmpeg_in_path = shutil.which("ffmpeg")
    if ffmpeg_in_path: return ffmpeg_in_path
    current_dir_ffmpeg = os.path.join(os.getcwd(), "ffmpeg.exe")
    script_dir_ffmpeg = os.path.join(SCRIPT_DIR, "ffmpeg.exe")
    if os.path.isfile(current_dir_ffmpeg): return current_dir_ffmpeg
    if os.path.isfile(script_dir_ffmpeg): return script_dir_ffmpeg
    print("[WARNING] Không tìm thấy đường dẫn ffmpeg cụ thể, thử gọi 'ffmpeg'.")
    return "ffmpeg"


# --- Các hàm xử lý chính ---

def extract_audio(video_path, audio_path):
    print("\n[1a] Tách âm thanh (mono 16k cho Whisper)...")
    start_time = time.time()
    if not os.path.exists(video_path): print(f"[ERROR] Không tìm thấy file video: {video_path}"); return False
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    command = [get_ffmpeg_path(), "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
    try:
        print(f"   Đang chạy: {' '.join(command)}"); result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0: print(f"[ERROR] Không tạo được file âm thanh mono.\n{result.stderr}"); return False
        end_time = time.time(); print(f"[OK] Đã lưu âm thanh (mono 16k): {audio_path} ({end_time - start_time:.2f} giây)"); return True
    except subprocess.CalledProcessError as e: print(f"[ERROR] Lỗi ffmpeg tách audio mono:\n{e.stderr}"); return False
    except FileNotFoundError: print("[ERROR] Không tìm thấy 'ffmpeg'."); return False
    except Exception as e: print(f"[ERROR] Lỗi tách audio mono: {e}"); return False

def extract_full_audio(video_path, full_audio_path):
    print(f"\n[1b] Tách âm thanh gốc đầy đủ (cho Demucs): {full_audio_path}...")
    start_time = time.time()
    if not os.path.exists(video_path): print(f"[ERROR] Không tìm thấy file video: {video_path}"); return None
    os.makedirs(os.path.dirname(full_audio_path), exist_ok=True)
    wav_path = os.path.splitext(full_audio_path)[0] + ".wav"
    command_copy = [get_ffmpeg_path(), "-y", "-i", video_path, "-vn", "-acodec", "copy", full_audio_path]
    command_wav = [get_ffmpeg_path(), "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", wav_path]
    final_extracted_path = full_audio_path
    try:
        print(f"   Thử sao chép codec audio: {' '.join(command_copy)}")
        result = subprocess.run(command_copy, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if not os.path.exists(final_extracted_path) or os.path.getsize(final_extracted_path) == 0: raise ValueError("Sao chép codec thất bại.")
        print(f"[OK] Đã tách âm thanh gốc (copy codec): {final_extracted_path}")
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"[INFO] Sao chép codec lỗi ({e}), thử mã hóa sang WAV...")
        final_extracted_path = wav_path
        try:
            print(f"   Đang mã hóa lại audio sang WAV: {' '.join(command_wav)}")
            result = subprocess.run(command_wav, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if not os.path.exists(final_extracted_path) or os.path.getsize(final_extracted_path) == 0: print(f"[ERROR] Mã hóa lại WAV thất bại.\n{result.stderr}"); return None
            print(f"[OK] Đã tách âm thanh gốc (mã hóa WAV): {final_extracted_path}")
        except subprocess.CalledProcessError as e2: print(f"[ERROR] Lỗi ffmpeg mã hóa WAV:\n{e2.stderr}"); return None
        except FileNotFoundError: print("[ERROR] Không tìm thấy 'ffmpeg'."); return None
        except Exception as e2: print(f"[ERROR] Lỗi mã hóa WAV: {e2}"); return None
    end_time = time.time(); print(f"   (Thời gian tách audio đầy đủ: {end_time - start_time:.2f} giây)");
    return final_extracted_path

def separate_vocals_demucs(input_audio_path, output_dir):
    print(f"\n[1c] Đang tách giọng nói bằng Demucs...")
    print(f"     Input: {input_audio_path}")
    print(f"     Output Dir: {output_dir}")
    start_time = time.time()
    python_executable = sys.executable
    if os.path.exists(output_dir):
         try: shutil.rmtree(output_dir); print(f"   Xóa thư mục Demucs output cũ: {output_dir}")
         except Exception as e_clean: print(f"[WARNING] Không thể xóa thư mục Demucs cũ: {e_clean}")
    os.makedirs(output_dir, exist_ok=True)
    command = [ python_executable, "-m", "demucs", "--two-stems=vocals", "-o", output_dir, input_audio_path ]
    try:
        print(f"   Đang chạy Demucs: {' '.join(command)}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore')
        while True:
            output = process.stdout.readline();
            if output == '' and process.poll() is not None: break
            if output: print(output.strip())
        if process.poll() != 0: raise subprocess.CalledProcessError(process.poll(), command)
        expected_output_file = None
        print(f"   Tìm kiếm '{BACKGROUND_AUDIO_DEFAULT_NAME}' trong '{output_dir}'...")
        for root, _, files in os.walk(output_dir):
            if BACKGROUND_AUDIO_DEFAULT_NAME in files:
                expected_output_file = os.path.join(root, BACKGROUND_AUDIO_DEFAULT_NAME); print(f"   Đã tìm thấy: {expected_output_file}"); break
        if expected_output_file and os.path.exists(expected_output_file) and os.path.getsize(expected_output_file) > 100:
            end_time = time.time(); print(f"[OK] Demucs tách xong: {expected_output_file} ({end_time - start_time:.2f} giây)"); return expected_output_file
        else: print(f"[ERROR] Demucs chạy xong nhưng không tìm thấy '{BACKGROUND_AUDIO_DEFAULT_NAME}'."); return None
    except subprocess.CalledProcessError as e: print(f"[ERROR] Lỗi chạy Demucs (code: {e.returncode})."); return None
    except FileNotFoundError: print(f"[ERROR] Không tìm thấy Python: {python_executable}"); return None
    except Exception as e: print(f"[ERROR] Lỗi Demucs không xác định: {e}"); import traceback; traceback.print_exc(); return None

def transcribe_audio(audio_path, model_size="medium", language="zh"):
    print(f"\n[2] Đang nhận dạng giọng nói ({language})..."); start_time = time.time()
    if not os.path.exists(audio_path): print(f"[ERROR] Không tìm thấy file audio: {audio_path}"); return None
    try:
        model = whisper.load_model(model_size); print("   Model Whisper đã tải. Bắt đầu nhận dạng...")
        result = model.transcribe(audio_path, language=language, fp16=False, verbose=None); end_time = time.time()
        print(f"[OK] Nhận dạng giọng nói hoàn tất ({end_time - start_time:.2f} giây)")
        return result.get("segments", [])
    except Exception as e: print(f"[ERROR] Lỗi Whisper: {e}"); return None

def save_zh_segments(segments, file_path):
    if not segments: print("[INFO] Không có segment tiếng Trung."); return
    print(f"\n[INFO] Lưu câu gốc vào {file_path}...")
    try:
        count = 0; os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for seg in segments:
                text = seg.get('text', '').strip();
                if text: f.write(text + "\n"); count += 1
        print(f"[OK] Đã lưu {count} câu tiếng Trung: {file_path}")
    except Exception as e: print(f"[ERROR] Lỗi lưu file {file_path}: {e}")

def translate_with_gemini(zh_lines_to_translate):
    print("\n[3] Đang dịch sang tiếng Việt qua Gemini API...")
    if not zh_lines_to_translate: return []
    if not GEMINI_API_KEY: print("[ERROR] Thiếu GEMINI_API_KEY."); return None
    batch_size = 50; all_translated_lines = []; total_lines = len(zh_lines_to_translate); start_time = time.time()
    num_batches = math.ceil(total_lines / batch_size); print(f"   Tổng {total_lines} dòng, chia {num_batches} batch.")
    api_error_count = 0
    with tqdm(total=total_lines, desc="   Translating", unit="line", ncols=100) as pbar:
        for i in range(0, total_lines, batch_size):
            batch_zh = zh_lines_to_translate[i:min(i + batch_size, total_lines)]
            prompt = "Đóng vai chuyên gia phiên dịch Trung - Việt, dịch các câu sau từ Trung sang Việt. Giữ nguyên số dòng, mỗi dòng là bản dịch tương ứng. Chỉ trả về bản dịch:\n\n" + "\n".join(batch_zh)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
            headers = {"Content-Type": "application/json"}; data = {"contents": [{"parts": [{"text": prompt}]}]}
            try:
                response = requests.post(url, headers=headers, json=data, timeout=180); response.raise_for_status(); response_data = response.json()
                if not response_data.get("candidates") or not response_data["candidates"][0].get("content") or not response_data["candidates"][0]["content"].get("parts"):
                     pbar.write(f"\n[ERROR] Phản hồi Gemini không hợp lệ (Batch {i//batch_size + 1})"); api_error_count += len(batch_zh); all_translated_lines.extend(["[LỖI API Format]"] * len(batch_zh)); continue
                content = response_data["candidates"][0]["content"]["parts"][0].get("text", "")
                translated_lines_batch = [line.strip() for line in content.strip().split("\n")]
                if len(translated_lines_batch) < len(batch_zh): pbar.write(f"\n[WARNING] Gemini thiếu dòng (Batch {i//batch_size + 1})"); translated_lines_batch.extend(["[LỖI API Thiếu]"] * (len(batch_zh) - len(translated_lines_batch)))
                elif len(translated_lines_batch) > len(batch_zh): pbar.write(f"\n[WARNING] Gemini thừa dòng (Batch {i//batch_size + 1})"); translated_lines_batch = translated_lines_batch[:len(batch_zh)]
                all_translated_lines.extend(translated_lines_batch)
            except requests.exceptions.Timeout: pbar.write(f"\n[ERROR] Timeout API (Batch {i//batch_size + 1})"); api_error_count += len(batch_zh); all_translated_lines.extend(["[LỖI API Timeout]"] * len(batch_zh))
            except requests.exceptions.RequestException as e: pbar.write(f"\n[ERROR] Lỗi kết nối API (Batch {i//batch_size + 1}): {e}"); api_error_count += len(batch_zh); all_translated_lines.extend(["[LỖI API Kết nối]"] * len(batch_zh))
            except Exception as e: pbar.write(f"\n[ERROR] Lỗi xử lý API (Batch {i//batch_size + 1}): {e}"); api_error_count += len(batch_zh); all_translated_lines.extend(["[LỖI API Xử lý]"] * len(batch_zh))
            pbar.update(len(batch_zh)); time.sleep(1)
    end_time = time.time()
    if api_error_count > 0: print(f"[WARNING] {api_error_count}/{total_lines} dòng lỗi dịch.")
    print(f"[OK] Đã xử lý dịch {total_lines} dòng ({end_time - start_time:.2f} giây)"); return all_translated_lines

def generate_srt(segments, translated_texts, srt_path):
    print(f"\n[4] Tạo file phụ đề SRT: {srt_path}..."); start_time = time.time(); valid_segments_count = 0
    if not segments: print("[WARNING] Không có segment."); return False
    try:
        os.makedirs(os.path.dirname(srt_path), exist_ok=True)
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(tqdm(segments, desc="   Generating SRT", unit="seg", ncols=100)):
                if 'start' not in seg or 'end' not in seg: continue
                text = translated_texts[i].strip() if i < len(translated_texts) and translated_texts[i] else "[LỖI Dịch]"
                if not text: text = "[Dịch rỗng]"
                f.write(f"{valid_segments_count + 1}\n{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n{text}\n\n"); valid_segments_count += 1
        end_time = time.time()
        print(f"[OK] Đã tạo SRT với {valid_segments_count} mục ({end_time - start_time:.2f} giây)")
        return valid_segments_count > 0
    except Exception as e: print(f"[ERROR] Lỗi ghi file SRT {srt_path}: {e}"); return False

def embed_soft_subtitles(video_path, srt_path, output_path):
    print(f"\n[5] Nhúng phụ đề mềm (softsub): {output_path}..."); start_time = time.time()
    if not os.path.exists(video_path): print(f"[ERROR] Video gốc không tồn tại."); return False
    if not os.path.exists(srt_path) or os.path.getsize(srt_path) == 0: print(f"[ERROR] SRT không tồn tại hoặc rỗng."); return False
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    command = [get_ffmpeg_path(), "-y", "-i", video_path, "-i", srt_path, "-map", "0:v", "-map", "0:a", "-map", "1:s", "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=vie", "-disposition:s:0", "default", output_path]
    try:
        print(f"   Đang chạy: {' '.join(command)}"); result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 100: print(f"[ERROR] Tạo video softsub thất bại.\n{result.stderr}"); return False
        end_time = time.time(); print(f"[OK] Đã tạo video softsub: {output_path} ({end_time - start_time:.2f} giây)"); return True
    except subprocess.CalledProcessError as e: print(f"[ERROR] Lỗi ffmpeg nhúng softsub:\n{e.stderr}"); return False
    except FileNotFoundError: print("[ERROR] Không tìm thấy 'ffmpeg'."); return False
    except Exception as e: print(f"[ERROR] Lỗi nhúng softsub: {e}"); return False

def create_hardsub_video(video_in, srt_in, video_out, crop_pixels=0):
    action = f"cắt {crop_pixels}px & " if crop_pixels > 0 else ""
    print(f"\n[6] Đang {action}ghi cứng phụ đề: {video_out}..."); print("    (Có thể mất nhiều thời gian)"); start_time = time.time()
    temp_srt_path_in_cwd = None
    if not os.path.exists(video_in): print(f"[ERROR] Video gốc không tồn tại."); return False
    if not os.path.exists(srt_in) or os.path.getsize(srt_in) == 0: print(f"[ERROR] SRT không tồn tại hoặc rỗng."); return False
    os.makedirs(os.path.dirname(video_out), exist_ok=True)
    try:
        srt_filename = os.path.basename(srt_in); cwd = os.getcwd()
        safe_srt_filename = re.sub(r'[<>:"/\\|?*]', '_', srt_filename)
        temp_srt_filename = f"temp_hardsub_{int(time.time())}_{safe_srt_filename}"
        temp_srt_path_in_cwd = os.path.join(cwd, temp_srt_filename)
        escaped_srt_path_for_filter = temp_srt_path_in_cwd.replace('\\', '/').replace(':', '\\:')
        print(f"   Copy SRT vào CWD: {temp_srt_path_in_cwd}"); shutil.copy2(srt_in, temp_srt_path_in_cwd)
        filter_chain = [f"crop=iw:ih-{crop_pixels}:0:0"] if crop_pixels > 0 else []
        filter_chain.append(f"subtitles=filename='{escaped_srt_path_for_filter}'")
        vf_filter_string = ",".join(filter_chain); print(f"   Filter (-vf): {vf_filter_string}")
        command = [get_ffmpeg_path(), "-y", "-i", video_in, "-vf", vf_filter_string, "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-c:a", "aac", "-b:a", "128k", video_out]
        print(f"   Đang chạy: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if not os.path.exists(video_out) or os.path.getsize(video_out) < 100: print(f"[ERROR] Tạo video hardsub thất bại.\n{result.stderr}"); return False
        end_time = time.time(); print(f"[OK] Đã tạo video hardsub: {video_out} ({end_time - start_time:.2f} giây)"); return True
    except subprocess.CalledProcessError as e: print(f"[ERROR] Lỗi ffmpeg tạo hardsub (code: {e.returncode}).\n        Stderr:\n{e.stderr}"); return False
    except FileNotFoundError: print("[ERROR] Không tìm thấy 'ffmpeg'."); return False
    except Exception as e: print(f"[ERROR] Lỗi tạo hardsub: {e}"); return False
    finally:
        if temp_srt_path_in_cwd and os.path.exists(temp_srt_path_in_cwd):
            try: os.remove(temp_srt_path_in_cwd); print(f"   Đã xóa SRT tạm: {temp_srt_path_in_cwd}")
            except Exception as e_clean: print(f"[WARNING] Không thể xóa SRT tạm {temp_srt_path_in_cwd}: {e_clean}")

def generate_tts_segments(srt_path, output_dir="tts_audio_segments_raw", lang='vi'):
    if not GTTS_AVAILABLE: print("[ERROR] Thiếu thư viện gTTS."); return None
    print(f"\n[7a] Tạo các file TTS gốc: {output_dir}...")
    print("     (Cần internet, có thể chậm)")
    start_time = time.time();
    if not os.path.exists(srt_path): print(f"[ERROR] Không tìm thấy SRT: {srt_path}"); return None
    if os.path.exists(output_dir): shutil.rmtree(output_dir); print(f"   Xóa thư mục TTS gốc cũ: {output_dir}")
    os.makedirs(output_dir, exist_ok=True); tts_segments_info = []; failed_tts_count = 0
    try:
        with open(srt_path, 'r', encoding='utf-8') as f: content = f.read().strip()
        blocks = re.split(r'\n\s*\n+', content); print(f"   Phân tích {len(blocks)} khối phụ đề...")
        for block in tqdm(blocks, desc="   Generating TTS", unit="seg", ncols=100):
            lines = block.strip().split('\n')
            if len(lines) < 3: continue
            try:
                time_line = lines[1]; time_match = re.match(r'(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})', time_line)
                if not time_match: continue
                h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, time_match.groups())
                start_sec = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0; end_sec = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
                duration_sec = end_sec - start_sec
                if duration_sec <= 0: tqdm.write(f"\n[WARNING] Bỏ qua segment thời lượng <= 0: {time_line}"); continue
                current_text = " ".join(lines[2:]).strip(); current_text = re.sub(r'<[^>]+>', '', current_text)
                if not current_text or current_text.startswith('[') or len(current_text.strip()) < 2: continue
                segment_index = len(tts_segments_info) + failed_tts_count + 1
                segment_filename = f"segment_{segment_index:04d}.mp3"; segment_path = os.path.join(output_dir, segment_filename)
                try:
                    tts = gTTS(text=current_text, lang=lang, slow=False); tts.save(segment_path)
                    if os.path.exists(segment_path) and os.path.getsize(segment_path) > 50:
                        tts_segments_info.append({'index': segment_index, 'text': current_text, 'start_sec': start_sec, 'end_sec': end_sec, 'duration_sec': duration_sec, 'audio_path': segment_path})
                    else: tqdm.write(f"\n[WARNING] Tạo TTS thất bại/rỗng: {segment_index}"); failed_tts_count += 1
                except Exception as tts_err: tqdm.write(f"\n[ERROR] Lỗi gTTS seg {segment_index} ({current_text[:20]}...): {tts_err}"); failed_tts_count += 1
                time.sleep(0.2)
            except Exception as parse_err: tqdm.write(f"\n[WARNING] Lỗi xử lý khối SRT: {parse_err}\nBlock:\n{block}")
        end_time = time.time()
        if failed_tts_count > 0: print(f"\n[WARNING] {failed_tts_count} lỗi khi tạo TTS.")
        print(f"\n[OK] Đã tạo {len(tts_segments_info)} file TTS gốc ({end_time - start_time:.2f} giây)")
        return tts_segments_info if tts_segments_info else None
    except Exception as e: print(f"[ERROR] Lỗi nghiêm trọng khi tạo TTS: {e}"); return None

def adjust_tts_segment_tempo(tts_segment_infos, output_dir="tts_audio_segments_adjusted"):
    print(f"\n[7b] Điều chỉnh tốc độ TTS: {output_dir}...")
    print("     (Có thể mất thời gian)")
    start_time = time.time();
    if not tts_segment_infos: print("[ERROR] Không có TTS để xử lý."); return None
    if os.path.exists(output_dir): shutil.rmtree(output_dir); print(f"   Xóa thư mục TTS điều chỉnh cũ: {output_dir}")
    os.makedirs(output_dir, exist_ok=True); adjusted_segments_info = []; ffmpeg_errors = 0; valid_adjusted_count = 0
    for info in tqdm(tts_segment_infos, desc="   Adjusting Tempo", unit="seg", ncols=100):
        raw_audio_path = info['audio_path']; srt_duration = info['duration_sec']
        adjusted_filename = f"segment_{info['index']:04d}_adjusted.mp3"; adjusted_audio_path = os.path.join(output_dir, adjusted_filename)
        info['adjusted_audio_path'] = None
        if not os.path.exists(raw_audio_path) or srt_duration <= 0.01: tqdm.write(f"\n[WARNING] Bỏ qua tempo seg {info['index']} (file/duration lỗi)"); adjusted_segments_info.append(info); continue
        tts_duration = get_audio_duration(raw_audio_path)
        if tts_duration is None or tts_duration <= 0.01: tqdm.write(f"\n[WARNING] Bỏ qua tempo seg {info['index']} (không lấy được duration)"); adjusted_segments_info.append(info); continue
        tempo_factor = (tts_duration / srt_duration) if srt_duration > 0.01 else 1.0
        atempo_filters = []; current_factor = tempo_factor
        while current_factor > ATEMPO_MAX: atempo_filters.append(f"atempo={ATEMPO_MAX}"); current_factor /= ATEMPO_MAX
        while current_factor < ATEMPO_MIN: atempo_filters.append(f"atempo={ATEMPO_MIN}"); current_factor /= ATEMPO_MIN
        if ATEMPO_MIN <= current_factor <= ATEMPO_MAX and abs(current_factor - 1.0) > 0.01: atempo_filters.append(f"atempo={current_factor:.4f}")
        elif not atempo_filters: atempo_filters.append("atempo=1.0")
        filter_string = ",".join(atempo_filters)
        command = [get_ffmpeg_path(), "-y", "-i", raw_audio_path, "-filter:a", filter_string, adjusted_audio_path]
        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
            if os.path.exists(adjusted_audio_path) and os.path.getsize(adjusted_audio_path) > 50:
                info['adjusted_audio_path'] = adjusted_audio_path; valid_adjusted_count += 1
            else: tqdm.write(f"\n[WARNING] Tạo file tempo thất bại/rỗng: {info['index']}"); ffmpeg_errors += 1
        except subprocess.CalledProcessError as e: tqdm.write(f"\n[ERROR] Lỗi ffmpeg tempo seg {info['index']}:\n{e.stderr}"); ffmpeg_errors += 1
        except FileNotFoundError: print("[ERROR] Không tìm thấy 'ffmpeg'."); return None
        except Exception as e: tqdm.write(f"\n[ERROR] Lỗi tempo seg {info['index']}: {e}"); ffmpeg_errors += 1
        adjusted_segments_info.append(info)
    end_time = time.time()
    if ffmpeg_errors > 0: print(f"\n[WARNING] {ffmpeg_errors} lỗi khi điều chỉnh tốc độ.")
    print(f"[OK] Đã xử lý {len(tts_segment_infos)} TTS (Thành công: {valid_adjusted_count}) ({end_time - start_time:.2f} giây)")
    return adjusted_segments_info

# <<< HÀM TẠO VIDEO VỚI THUYẾT MINH ĐỒNG BỘ (ITERATIVE MIXING - SỬA LẠI ADELAY SYNTAX) >>>
def create_synced_narration_video(video_in_path, background_audio_path, adjusted_segment_infos, video_out_path):
    """
    Tạo video với thuyết minh đồng bộ bằng cách trộn lặp lại từng đoạn TTS,
    đồng bộ hóa tham số audio và sử dụng cú pháp adelay cũ với all_pts=1.
    """
    print(f"\n[8] Đồng bộ hóa và TRỘN thuyết minh (iterative mixing v3): {video_out_path}...")
    print("     (Bước này có thể mất nhiều thời gian và tạo nhiều file tạm)")
    start_time = time.time()

    # --- Đường dẫn file tạm ---
    combined_tts_path = os.path.join(SCRIPT_DIR, "combined_tts_final_temp.wav") # Sử dụng .wav
    temp_files_to_delete = []

    # --- Kiểm tra file input ---
    if not os.path.exists(video_in_path): print(f"[ERROR] Không tìm thấy video đầu vào."); return False
    if not background_audio_path or not os.path.exists(background_audio_path): print(f"[ERROR] Không tìm thấy file âm thanh nền."); return False
    if not adjusted_segment_infos: print("[ERROR] Không có TTS đã điều chỉnh."); return False

    os.makedirs(os.path.dirname(video_out_path), exist_ok=True)

    # --- Lấy thời lượng video ---
    video_duration = None
    try:
        ffprobe_executable = os.path.join(os.path.dirname(get_ffmpeg_path()), "ffprobe")
        if not os.path.exists(ffprobe_executable): ffprobe_executable = "ffprobe"
        ffprobe_cmd = [ffprobe_executable, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_in_path]
        duration_str = subprocess.check_output(ffprobe_cmd).strip().decode('utf-8')
        video_duration = float(duration_str)
        print(f"   Thời lượng video gốc: {video_duration:.3f} giây")
    except Exception as e: print(f"[ERROR] Không lấy được thời lượng video gốc: {e}."); cleanup_temp_files(temp_files_to_delete + [combined_tts_path]); return False

    # --- Lọc TTS hợp lệ ---
    valid_tts_segments = [info for info in adjusted_segment_infos if info.get('adjusted_audio_path') and os.path.exists(info['adjusted_audio_path']) and os.path.getsize(info['adjusted_audio_path']) > 50]
    if not valid_tts_segments:
        print("[INFO] Không có TTS hợp lệ. Tạo video chỉ với nền.")
        command_mix_only_bg = [get_ffmpeg_path(), "-y", "-i", video_in_path, "-i", background_audio_path, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", video_out_path]
        try:
             print(f"   Đang chạy ffmpeg (chỉ trộn nền)..."); result = subprocess.run(command_mix_only_bg, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
             if not os.path.exists(video_out_path) or os.path.getsize(video_out_path) < 100: print(f"[ERROR] Tạo video nền thất bại.\n{result.stderr}"); return False
             print(f"[OK] Đã tạo video chỉ với nền: {video_out_path}"); return True
        except Exception as e: print(f"[ERROR] Lỗi tạo video nền: {e}"); return False

    # --- Bước 8a (Iterative): Tạo file âm thanh im lặng ban đầu (.wav) ---
    print("   [8a] Tạo file âm thanh im lặng ban đầu (.wav)...")
    silent_audio_path = os.path.join(SCRIPT_DIR, "iterative_mix_0.wav")
    temp_files_to_delete.append(silent_audio_path)
    audio_channels = 2; audio_channels_str = "2"; audio_samplerate_str = "44100"; target_ch_layout = "stereo"
    try:
        ffprobe_executable = os.path.join(os.path.dirname(get_ffmpeg_path()), "ffprobe")
        if not os.path.exists(ffprobe_executable): ffprobe_executable = "ffprobe"
        ffprobe_cmd_audio = [ffprobe_executable, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels,sample_rate", "-of", "default=noprint_wrappers=1", background_audio_path]
        audio_info_output = subprocess.check_output(ffprobe_cmd_audio).strip().decode('utf-8')
        audio_info = {}
        for line in audio_info_output.splitlines():
             if '=' in line: key, value = line.split('=', 1); audio_info[key.strip()] = value.strip()
        if 'channels' in audio_info:
            try: val = int(audio_info['channels']); audio_channels = val; audio_channels_str = str(val) if 1 <= val <= 8 else "2"
            except ValueError: pass
        if 'sample_rate' in audio_info:
            try: val = int(audio_info['sample_rate']); audio_samplerate_str = str(val) if 8000 <= val <= 192000 else "44100"
            except ValueError: pass
        target_ch_layout = "mono" if audio_channels_str == "1" else "stereo"
        print(f"        Thông tin audio mục tiêu: Ch={audio_channels_str}, Rate={audio_samplerate_str}, Layout={target_ch_layout}")
    except Exception as e: print(f"[WARNING] Lỗi lấy thông tin audio nền: {e}")

    command_silent = [get_ffmpeg_path(), "-y", "-f", "lavfi",
                      "-i", f"anullsrc=channel_layout={target_ch_layout}:sample_rate={audio_samplerate_str}",
                      "-t", str(video_duration), "-c:a", "pcm_s16le", silent_audio_path]
    try:
        print(f"        Đang chạy lệnh tạo silent .wav: {' '.join(command_silent[:7])} ...")
        result_silent = subprocess.run(command_silent, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if not os.path.exists(silent_audio_path) or os.path.getsize(silent_audio_path) == 0:
            raise RuntimeError(f"Tạo file im lặng .wav thất bại.\nStderr:\n{result_silent.stderr if result_silent else 'N/A'}")
        print(f"        [OK] Đã tạo file im lặng .wav: {silent_audio_path}")
    except Exception as e:
        print(f"[ERROR] Lỗi tạo file im lặng .wav: {e}")
        cleanup_temp_files(temp_files_to_delete)
        return False

    # --- Bước 8b (Iterative): Trộn lần lượt từng file TTS (SỬA LẠI ADELAY SYNTAX) ---
    print("   [8b] Trộn lặp lại các file TTS (đồng bộ hóa, adelay all=1)...")
    current_mix_file = silent_audio_path

    with tqdm(total=len(valid_tts_segments), desc="   Mixing TTS", unit="seg", ncols=100) as pbar:
        for i, info in enumerate(valid_tts_segments):
            tts_path = info['adjusted_audio_path']
            iteration = i + 1
            next_mix_file = os.path.join(SCRIPT_DIR, f"iterative_mix_{iteration}.wav")
            temp_files_to_delete.append(next_mix_file)

            start_ms = int(info['start_sec'] * 1000)

            # --- Filter complex: aformat, adelay (đúng cú pháp), amix ---
            # Tạo chuỗi delay dạng "ms|ms|..." cho tất cả các kênh
            delay_str_per_channel = "|".join([str(start_ms)] * audio_channels)

            filter_complex_string_iter = (
                f"[1:a]aformat=sample_fmts=fltp:sample_rates={audio_samplerate_str}:channel_layouts={target_ch_layout}," # Đồng bộ TTS
                f"adelay={delay_str_per_channel}:all=1[delayed_tts];" # Delay TTS (sử dụng cú pháp đúng với all=1)
                f"[0:a][delayed_tts]amix=inputs=2:duration=first:dropout_transition=0[mixed]" # Trộn
            )

            command_iterative_mix = [get_ffmpeg_path(), "-y",
                                     "-i", current_mix_file,
                                     "-i", tts_path,
                                     "-filter_complex", filter_complex_string_iter,
                                     "-map", "[mixed]",
                                     "-c:a", "pcm_s16le", # Sử dụng codec lossless
                                     next_mix_file]

            try:
                result_iter = subprocess.run(command_iterative_mix, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                if not os.path.exists(next_mix_file) or os.path.getsize(next_mix_file) < 50:
                     tqdm.write(f"\n[ERROR] Tạo file mix .wav trung gian thất bại: {next_mix_file}")
                     tqdm.write(f"        Stderr:\n{result_iter.stderr if result_iter else 'N/A'}")
                     cleanup_temp_files(temp_files_to_delete)
                     return False

                if i > 0 and current_mix_file in temp_files_to_delete:
                     try: os.remove(current_mix_file); temp_files_to_delete.remove(current_mix_file)
                     except Exception as e_del: tqdm.write(f"\n[WARNING] Không xóa được file mix tạm .wav {current_mix_file}: {e_del}")

                current_mix_file = next_mix_file
                pbar.update(1)

            except subprocess.CalledProcessError as e:
                tqdm.write(f"\n[ERROR] Lỗi trộn TTS segment {info['index']} (code: {e.returncode}).")
                tqdm.write(f"        Filter: {filter_complex_string_iter}")
                tqdm.write(f"        Stderr:\n{e.stderr}")
                cleanup_temp_files(temp_files_to_delete)
                return False
            except Exception as e:
                tqdm.write(f"\n[ERROR] Lỗi không xác định trộn TTS seg {info['index']}: {e}")
                cleanup_temp_files(temp_files_to_delete)
                return False

    # Đổi tên file mix cuối cùng (.wav)
    try:
        print(f"   Đổi tên file mix .wav cuối cùng thành: {combined_tts_path}")
        if os.path.exists(combined_tts_path): os.remove(combined_tts_path)
        shutil.move(current_mix_file, combined_tts_path)
        if current_mix_file in temp_files_to_delete: temp_files_to_delete.remove(current_mix_file)
        temp_files_to_delete.append(combined_tts_path)
    except Exception as e:
        print(f"[ERROR] Lỗi đổi tên file mix .wav cuối cùng: {e}")
        cleanup_temp_files(temp_files_to_delete)
        return False

    print("   [OK] Hoàn thành trộn lặp lại TTS (.wav).")

    # --- Bước 8c: Trộn video, âm thanh nền và TTS kết hợp (.wav) ---
    print("   [8c] Trộn video, nền và thuyết minh kết hợp (.wav)...")
    command_final_mix = [get_ffmpeg_path(), "-y",
                         "-i", video_in_path,
                         "-i", background_audio_path,
                         "-i", combined_tts_path,
                         "-filter_complex", f"[1:a]aformat=sample_fmts=fltp:sample_rates={audio_samplerate_str}:channel_layouts={target_ch_layout}[bg_sync];"
                                            f"[2:a]aformat=sample_fmts=fltp:sample_rates={audio_samplerate_str}:channel_layouts={target_ch_layout}[tts_sync];"
                                            f"[bg_sync][tts_sync]amix=inputs=2:duration=first:dropout_transition=0[a_final_mix]",
                         "-map", "0:v",
                         "-map", "[a_final_mix]",
                         "-c:v", "copy",
                         "-c:a", "aac", "-b:a", "192k", # Mã hóa sang AAC
                         "-shortest",
                         video_out_path]
    try:
        print(f"        Đang chạy lệnh ffmpeg (final mix): {' '.join(command_final_mix[:7])} ...")
        result_final = subprocess.run(command_final_mix, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if not os.path.exists(video_out_path) or os.path.getsize(video_out_path) < 100:
            print(f"[ERROR] Không tạo được file video cuối cùng.\n        Stderr:\n{result_final.stderr if result_final else 'N/A'}")
            cleanup_temp_files(temp_files_to_delete)
            return False
        end_time = time.time()
        print(f"[OK] Đã tạo video thuyết minh: {video_out_path} (Thời gian bước 8: {end_time - start_time:.2f} giây)")
        cleanup_temp_files([combined_tts_path, silent_audio_path]) # Xóa file .wav cuối và silent ban đầu
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Lỗi trộn video cuối cùng (code: {e.returncode}).")
        print(f"        Stderr:\n{e.stderr}")
        cleanup_temp_files(temp_files_to_delete)
        return False
    except Exception as e:
        print(f"[ERROR] Lỗi không xác định trộn video cuối cùng: {e}")
        cleanup_temp_files(temp_files_to_delete)
        return False

# --- Phần còn lại của code (cleanup_temp_files, main) ---
def cleanup_temp_files(paths_to_clean):
    cleaned_paths = set()
    paths_copy = list(paths_to_clean)
    for path in paths_copy:
        if not path or path in cleaned_paths: continue
        try:
            if os.path.isfile(path): os.remove(path); cleaned_paths.add(path)
            elif os.path.isdir(path): shutil.rmtree(path); cleaned_paths.add(path)
        except Exception as e: print(f"[WARNING] Không thể xóa {path}: {e}")

# --- Hàm Main ---
def main():
    print("--- Bắt đầu quy trình ---"); total_start_time = time.time()
    final_message = []; background_audio_file = None; full_audio_file_path = None
    files_to_cleanup_main = []

    ffmpeg_exe = get_ffmpeg_path()
    try: subprocess.run([ffmpeg_exe, "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE); print(f"[INFO] FFmpeg OK: {ffmpeg_exe}")
    except (FileNotFoundError, subprocess.CalledProcessError): print("[ERROR] FFmpeg lỗi."); return

    files_to_cleanup_main.append(AUDIO_FILE)
    if not extract_audio(VIDEO_FILE, AUDIO_FILE): print("[DỪNG]"); cleanup_temp_files(files_to_cleanup_main); return
    full_audio_file_path = extract_full_audio(VIDEO_FILE, FULL_AUDIO_FILE)
    if full_audio_file_path: files_to_cleanup_main.append(full_audio_file_path)
    if not full_audio_file_path: print("[DỪNG]"); cleanup_temp_files(files_to_cleanup_main); return
    files_to_cleanup_main.append(DEMUCS_OUTPUT_DIR)
    background_audio_file = separate_vocals_demucs(full_audio_file_path, DEMUCS_OUTPUT_DIR)
    if not background_audio_file: print("[DỪNG]"); cleanup_temp_files(files_to_cleanup_main); return
    else: final_message.append(f"Nền: {background_audio_file}")
    segments = transcribe_audio(AUDIO_FILE, MODEL_SIZE, LANGUAGE_CODE)
    if not segments: print("[DỪNG]"); cleanup_temp_files(files_to_cleanup_main); return
    save_zh_segments(segments, SEGMENTS_ZH_FILE)
    zh_lines_original = [seg.get('text', '').strip() for seg in segments]
    zh_lines_to_translate = [line for line in zh_lines_original if line]
    final_vi_lines = [""] * len(segments)
    if zh_lines_to_translate:
        vi_lines_translated = translate_with_gemini(zh_lines_to_translate)
        if vi_lines_translated is None: print("[DỪNG]"); cleanup_temp_files(files_to_cleanup_main); return
        current_translation_index = 0; missing_translations = 0
        for i, zh_line in enumerate(zh_lines_original):
            if zh_line:
                if current_translation_index < len(vi_lines_translated): final_vi_lines[i] = vi_lines_translated[current_translation_index]; current_translation_index += 1
                else: final_vi_lines[i] = f"[LỖI API]"; missing_translations += 1
        if current_translation_index != len(vi_lines_translated) or missing_translations > 0: print(f"[WARNING] Khớp dịch lỗi")
        if len(final_vi_lines) != len(segments): print("[ERROR] Khớp dịch nghiêm trọng."); cleanup_temp_files(files_to_cleanup_main); return
    if not generate_srt(segments, final_vi_lines, SRT_FILE): print("[DỪNG]"); cleanup_temp_files(files_to_cleanup_main); return
    else: final_message.append(f"SRT: {SRT_FILE}")

    softsub_success = embed_soft_subtitles(VIDEO_FILE, SRT_FILE, OUTPUT_VIDEO_FILE_SOFTSUB)
    final_message.append(f"Softsub: {OUTPUT_VIDEO_FILE_SOFTSUB}" if softsub_success else "[LỖI] Softsub")
    hardsub_success = create_hardsub_video(VIDEO_FILE, SRT_FILE, OUTPUT_VIDEO_FILE_HARDSUB, crop_pixels=CROP_BOTTOM_PIXELS)
    final_message.append(f"Hardsub: {OUTPUT_VIDEO_FILE_HARDSUB}" if hardsub_success else "[LỖI] Hardsub")

    if GTTS_AVAILABLE:
        files_to_cleanup_main.extend([TTS_SEGMENTS_DIR, TTS_ADJUSTED_DIR])
        tts_segments_info_raw = generate_tts_segments(SRT_FILE, output_dir=TTS_SEGMENTS_DIR)
        if tts_segments_info_raw:
            final_message.append(f"TTS gốc: {len(tts_segments_info_raw)} files")
            tts_segments_info_adjusted = adjust_tts_segment_tempo(tts_segments_info_raw, output_dir=TTS_ADJUSTED_DIR)
            if tts_segments_info_adjusted:
                 valid_adjusted_count = sum(1 for info in tts_segments_info_adjusted if info.get('adjusted_audio_path'))
                 final_message.append(f"TTS điều chỉnh: {valid_adjusted_count} files")
                 if valid_adjusted_count > 0 and background_audio_file:
                     narrated_synced_success = create_synced_narration_video(VIDEO_FILE, background_audio_file, tts_segments_info_adjusted, OUTPUT_VIDEO_FILE_NARRATED)
                     final_message.append(f"Video thuyết minh: {OUTPUT_VIDEO_FILE_NARRATED}" if narrated_synced_success else "[LỖI] Video thuyết minh")
                 elif not background_audio_file: final_message.append("[LỖI] Thiếu file nền.")
                 else: final_message.append("[INFO] Không có TTS hợp lệ.")
            else: final_message.append("[LỖI] Điều chỉnh tempo")
        else: final_message.append("[LỖI] Tạo TTS gốc")
    else: final_message.append("[INFO] Bỏ qua thuyết minh")

    #cleanup_temp_files(files_to_cleanup_main)
    print("\n" + "="*30 + " KẾT QUẢ " + "="*30)
    print(f"Tổng thời gian: {time.time() - total_start_time:.2f} giây")
    print(f"Video gốc: {VIDEO_FILE}")
    if CROP_BOTTOM_PIXELS > 0: print(f"Crop: {CROP_BOTTOM_PIXELS} px")
    for msg in final_message: print(msg)
    print("="*68)

if __name__ == "__main__":
    main()
