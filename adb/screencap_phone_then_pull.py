import time
import signal
import subprocess
from pathlib import Path
from datetime import datetime
import struct
import shutil
import cv2
import numpy as np

# ====== SETTINGS ======
DISPLAY_ID = ""   # INTERNAL display id
INTERVAL_SEC = 1.0                   # target 1Hz
LOCAL_BASE  = "sessions_screencap_pc_roi"
PROGRESS_EVERY_SEC = 5

# ROI settings (same as your process_image)
x1: int = 1565; y1: int = 150; x2: int = 1907; y2: int = 240
dx: int = 350; dy: int = 200; y2_delta: int = -70
shrink: int = 150
# ======================

HERE = Path(__file__).resolve().parent
ADB  = "adb"


# -------------------- helpers --------------------
def run_cmd(cmd, capture=False, check=False):
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=False
    )
    if check and p.returncode != 0:
        out = (p.stdout or b"").decode("utf-8", errors="replace")
        err = (p.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(
            f"FAILED rc={p.returncode}\nCMD: {' '.join(map(str, cmd))}\n"
            f"STDOUT:\n{out[:1200]}\nSTDERR:\n{err[:1200]}"
        )
    return p

def adb(args, capture=False, check=False):
    return run_cmd([str(ADB)] + list(args), capture=capture, check=check)

def out_str(b: bytes) -> str:
    return (b or b"").decode("utf-8", errors="replace").strip()

def restart_adb():
    run_cmd([str(ADB), "kill-server"], capture=True, check=False)
    run_cmd([str(ADB), "start-server"], capture=True, check=False)

def wait_for_device_ready():
    while True:
        p = adb(["devices"], capture=True)
        s = out_str(p.stdout)
        lines = [ln.strip() for ln in s.splitlines() if "\t" in ln]

        if any(ln.endswith("\tdevice") for ln in lines):
            serial = [ln.split("\t")[0] for ln in lines if ln.endswith("\tdevice")][0]
            print("[+] Device ready:", serial)
            return serial

        if any(ln.endswith("\tunauthorized") for ln in lines):
            print("[+] Device unauthorized. Approve USB debugging on phone.")
        elif any(ln.endswith("\toffline") for ln in lines):
            print("[+] Device offline. Restarting adb...")
            restart_adb()
        else:
            print("[+] Waiting for device...")

        time.sleep(2)

# -------------------- Unicode-safe PNG save --------------------
def save_png_unicode_safe(path: Path, bgr_img: np.ndarray):
    ok, buf = cv2.imencode(".png", bgr_img)
    if not ok:
        raise RuntimeError("imencode failed (cannot encode PNG)")
    path.write_bytes(buf.tobytes())


# -------------------- Stable RAW capture --------------------
def read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)

def adb_execout_screencap_raw(serial: str, display_id: str) -> bytes:
    """
    Reads EXACTLY header(12) + payload(w*h*4) from:
      adb -s SERIAL exec-out screencap -d DISPLAY_ID
    """
    cmd = [str(ADB), "-s", serial, "exec-out", "screencap"]
    if str(display_id).strip() != "":
        cmd += ["-d", str(display_id)]

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    try:
        hdr = read_exact(p.stdout, 12) if p.stdout else b""
        if len(hdr) < 12:
            err = (p.stderr.read() if p.stderr else b"").decode("utf-8", errors="replace")
            p.kill()
            raise RuntimeError(f"short header ({len(hdr)}/12). err={err[:300]}")

        w, h, fmt = struct.unpack("<III", hdr)
        expected = w * h * 4

        payload = read_exact(p.stdout, expected) if p.stdout else b""
        if len(payload) < expected:
            err = (p.stderr.read() if p.stderr else b"").decode("utf-8", errors="replace")
            p.kill()
            raise RuntimeError(
                f"short raw payload: got={len(payload)} expected={expected} (w={w}, h={h}, fmt={fmt}). err={err[:300]}"
            )

        # drain / wait
        p.wait(timeout=3)
        return hdr + payload

    finally:
        try:
            if p.poll() is None:
                p.kill()
        except:
            pass

def capture_screencap_raw_via_execout(serial: str, display_id: str, retries: int = 3) -> np.ndarray:
    """
    Retries with adb restart on failures like:
    - short payload
    - device offline / unauthorized / no devices
    """
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            data = adb_execout_screencap_raw(serial, display_id)
            w, h, fmt = struct.unpack("<III", data[:12])
            payload = data[12:12 + w*h*4]

            rgba = np.frombuffer(payload, dtype=np.uint8).reshape((h, w, 4))
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            return bgr

        except Exception as e:
            last_err = e
            # Soft recovery
            restart_adb()
            time.sleep(0.3)

            # Re-check device state (fast check)
            p = adb(["devices"], capture=True)
            s = out_str(p.stdout)
            if serial not in s or "\tdevice" not in s:
                # wait a bit more for reconnect / re-auth
                time.sleep(1.0)

    raise RuntimeError(f"capture failed after {retries} retries: {last_err}")


# -------------------- main --------------------
def main():
    if shutil.which(str(ADB)) is None:
        raise SystemExit(f"adb not found in PATH: {ADB}. Try: which adb")


    serial = wait_for_device_ready()

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    local_root = HERE / LOCAL_BASE
    local_dir  = local_root / run_id
    local_dir.mkdir(parents=True, exist_ok=True)

    log_path = local_dir / "log.csv"
    log_path.write_text("i,ts,filename,status\n", encoding="utf-8")

    print("[+] Local :", local_dir)
    print("[+] Mode  : adb exec-out screencap (RAW) -> PC -> crop ROI -> save PNG")
    print("Press Ctrl+C to stop.\n")

    stopping = False
    def on_sigint(sig, frame):
        nonlocal stopping
        stopping = True
        print("\n[!] Ctrl+C -> stopping... (finishing current iteration)")

    signal.signal(signal.SIGINT, on_sigint)

    i = 0
    next_t = time.time()
    last_print = 0.0
    last_fname = ""

    while True:
        if stopping:
            break

        now = time.time()
        if now < next_t:
            time.sleep(min(0.05, next_t - now))
            continue

        i += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{i:06d}_{ts}.png"
        out_path = local_dir / fname
        last_fname = fname

        status = "ok"
        try:
            frame_bgr = capture_screencap_raw_via_execout(serial, DISPLAY_ID, retries=3)
            save_png_unicode_safe(out_path, frame_bgr)


        except Exception as e:
            status = f"fail:{type(e).__name__}"
            print(f"[!] capture failed i={i}: {e}")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{i},{ts},{fname},{status}\n")

        next_t += INTERVAL_SEC

        if time.time() - last_print >= PROGRESS_EVERY_SEC:
            cnt = len(list(local_dir.glob("*.png")))
            print(f"[progress] saved={cnt} last={last_fname}")
            last_print = time.time()

    print("[+] Done.")
    print("[+] Output folder:", local_dir)
    print("[+] Log:", log_path)


if __name__ == "__main__":
    main()
