import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import subprocess
import time

def kill_tree(pid: int, force: bool = False):
    """
    Kill a process tree on Windows (PID + all children).
    If force=False: graceful attempt. If force=True: hard kill.
    """
    if not pid:
        return
    args = ["taskkill", "/PID", str(pid), "/T"]
    if force:
        args.append("/F")
    try:
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ======= FIXED PATHS (no args needed) =======
BASE_OUT = Path(r"C:\PUBG\PUBG-UX-Research\captures")

PY_SCRIPT = Path(r"C:\PUBG\PUBG-UX-Research\adb\screencap_phone_then_pull.py")
PS1_SCRIPT = Path(r"C:\PUBG\PUBG-UX-Research\record_pcap.ps1")

# where adb.exe is located (adjust if different)
ADB_DIR = Path(r"C:\PUBG\PUBG-UX-Research\adb")

PY_EXE = sys.executable
# ===========================================


def pump_output(pipe, prefix: str, stop_event: threading.Event):
    """
    Read child process output line-by-line and print it to our console.
    Works for both stdout and stderr.
    """
    try:
        for line in iter(pipe.readline, ""):
            if stop_event.is_set():
                break
            if line:
                print(f"{prefix}{line}", end="")
    except Exception as e:
        # Don't crash the main program if reading output fails
        print(
        f"{prefix}[output-pump-error] {e}")
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def spawn_python(py_exe: str, script_path: Path, out_dir: Path) -> subprocess.Popen:
    # add adb dir into PATH only for this child process
    env = os.environ.copy()
    env["PATH"] = str(ADB_DIR) + os.pathsep + env.get("PATH", "")

    # -u = unbuffered output so prints appear immediately
    # NOTE: Your python script MUST support --out_dir, otherwise it will exit with error.
    cmd = [py_exe, "-u", str(script_path), "--out_dir", str(out_dir)]

    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # line-buffered
    )


def spawn_powershell_ps1(ps1_path: Path, out_dir: Path) -> subprocess.Popen:
    # Force UTF-8 output so Hebrew won't crash decoding
    ps_command = (
        "$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8; "
        f"& '{str(ps1_path)}' -OutDir '{str(out_dir)}'"
    )

    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-Command", ps_command,
    ]

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1
    )


def main():
    if not PY_SCRIPT.exists():
        raise SystemExit(f"Python script not found: {PY_SCRIPT}")
    if not PS1_SCRIPT.exists():
        raise SystemExit(f"PowerShell script not found: {PS1_SCRIPT}")
    if not ADB_DIR.exists():
        raise SystemExit(f"ADB_DIR not found: {ADB_DIR} (expected adb.exe inside)")
    if not (ADB_DIR / "adb.exe").exists():
        print(f"[warn] adb.exe not found in {ADB_DIR}. If adb is elsewhere, update ADB_DIR.")

    # Create readable timestamp folder
    run_stamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    run_dir = BASE_OUT / run_stamp

    out_phone = run_dir
    out_pcap = run_dir
    out_phone.mkdir(parents=True, exist_ok=True)
    out_pcap.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()

    procs = {"py": None, "ps": None}
    pumps = []  # threads that pump output

    def runner_py():
        try:
            p = spawn_python(PY_EXE, PY_SCRIPT, out_phone)
            procs["py"] = p

            # start stdout/stderr pumps
            t_out = threading.Thread(target=pump_output, args=(p.stdout, "[PHONE] ", stop_event), daemon=True)
            t_err = threading.Thread(target=pump_output, args=(p.stderr, "[PHONE][ERR] ", stop_event), daemon=True)
            t_out.start()
            t_err.start()
            pumps.extend([t_out, t_err])

            rc = p.wait()
            print(f"[PHONE] process exited with code {rc}")
            stop_event.set()

        except Exception as e:
            print(f"[PHONE][spawn-error] {e}")
            stop_event.set()

    def runner_ps():
        try:
            p = spawn_powershell_ps1(PS1_SCRIPT, out_pcap)
            procs["ps"] = p

            # start stdout/stderr pumps
            t_out = threading.Thread(target=pump_output, args=(p.stdout, "[PCAP] ", stop_event), daemon=True)
            t_err = threading.Thread(target=pump_output, args=(p.stderr, "[PCAP][ERR] ", stop_event), daemon=True)
            t_out.start()
            t_err.start()
            pumps.extend([t_out, t_err])

            rc = p.wait()
            print(f"[PCAP] process exited with code {rc}")
            stop_event.set()

        except Exception as e:
            print(f"[PCAP][spawn-error] {e}")
            stop_event.set()

    def terminate_all():
        stop_event.set()

        # 1) ניסיון "רך" לעצור את שניהם + כל הילדים שלהם
        for k in ("py", "ps"):
            p = procs.get(k)
            if p and p.poll() is None:
                kill_tree(p.pid, force=False)

        time.sleep(1.0)

        # 2) אם עדיין נשאר משהו חי — להרוג בכוח (כולל ילדים)
        for k in ("py", "ps"):
            p = procs.get(k)
            if p and p.poll() is None:
                kill_tree(p.pid, force=True)


    def on_sigint(sig, frame):
        print("\n[main] Ctrl+C -> stopping both...")
        terminate_all()

    signal.signal(signal.SIGINT, on_sigint)

    print("[main] run folder:", run_dir)
    print("[main] phone output:", out_phone)
    print("[main] pcap output :", out_pcap)
    print("[main] starting both in parallel...")

    t_py = threading.Thread(target=runner_py, daemon=True)
    t_ps = threading.Thread(target=runner_ps, daemon=True)
    t_py.start()
    t_ps.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        terminate_all()
        t_py.join(timeout=2.0)
        t_ps.join(timeout=2.0)
        for t in pumps:
            t.join(timeout=1.0)

    print("[main] done.")


if __name__ == "__main__":
    main()
