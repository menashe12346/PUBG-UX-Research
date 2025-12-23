import subprocess
import time
import re
import ipaddress
from datetime import datetime
from pathlib import Path

# ====================== SETTINGS ======================
# If you can't/won't start hotspot automatically in code,
# turn on Windows "Mobile hotspot" manually in Settings
# and the script will still detect the correct interface.
TRY_START_HOSTEDNETWORK = False

SSID = "MENASHE_LEPTOP"
PASSWORD = "z818Y03a"

OUT_DIR = Path("pcaps_hotspot")
POLL_SEC = 2.0
ARP_STABLE_COUNT = 2

CAPTURE_ONLY_PHONE = True  # True -> capture filter to only the phone (IP/MAC)
# ======================================================

# If tshark is not in PATH, set this explicitly:
# Example:
# TSHARK = r"C:\Program Files\Wireshark\tshark.exe"
TSHARK = "tshark"
NETSH = "netsh"


# ------------------ helpers ------------------
def run(cmd, capture=True, check=False):
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        shell=False,
    )
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed rc={p.returncode}\nCMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    return p


def find_tshark_exe():
    """
    Return a working tshark executable path.
    - If TSHARK already works, keep it.
    - Otherwise, try common install paths.
    """
    global TSHARK

    # 1) Try as-is (PATH)
    try:
        run([TSHARK, "-v"], capture=True, check=True)
        return TSHARK
    except Exception:
        pass

    # 2) Try common Wireshark install locations
    candidates = [
        r"C:\Program Files\Wireshark\tshark.exe",
        r"C:\Program Files (x86)\Wireshark\tshark.exe",
    ]
    for c in candidates:
        try:
            run([c, "-v"], capture=True, check=True)
            TSHARK = c
            return TSHARK
        except Exception:
            continue

    return None


def ensure_tools():
    tshark_path = find_tshark_exe()
    if not tshark_path:
        raise SystemExit(
            "tshark was not found.\n\n"
            "Fix options:\n"
            "1) Install Wireshark and include TShark, then reopen PowerShell and run: tshark -v\n"
            "2) Or set TSHARK to full path in the script, e.g.\n"
            "   TSHARK = r\"C:\\Program Files\\Wireshark\\tshark.exe\"\n"
        )


def powershell(cmd: str) -> str:
    p = run(["powershell", "-NoProfile", "-Command", cmd], capture=True, check=True)
    return (p.stdout or "").strip()


# ------------------ tshark / interface mapping ------------------
def tshark_list_interfaces():
    """
    tshark -D -> [(idx, dev, desc)]
    dev is usually like: \\Device\\NPF_{GUID}
    """
    p = run([TSHARK, "-D"], capture=True, check=True)
    lines = [ln.strip() for ln in (p.stdout or "").splitlines() if ln.strip()]
    out = []
    for ln in lines:
        m = re.match(r"^(\d+)\.\s+(\S+)\s+\((.*)\)\s*$", ln)
        if m:
            idx = int(m.group(1))
            dev = m.group(2)
            desc = m.group(3)
            out.append((idx, dev, desc))
    return out


def get_interface_guid_by_ip(pc_iface_ip: str):
    """
    Get the interface GUID for the adapter that owns pc_iface_ip.
    Uses PowerShell:
      - Get-NetIPAddress -> InterfaceAlias
      - Get-NetAdapter   -> InterfaceGuid
    Returns (alias, guid)
    """
    alias = powershell(
        f"(Get-NetIPAddress -AddressFamily IPv4 | "
        f"Where-Object {{$_.IPAddress -eq '{pc_iface_ip}'}} | "
        f"Select-Object -First 1 -ExpandProperty InterfaceAlias)"
    )
    if not alias:
        return None, None

    guid = powershell(
        f"(Get-NetAdapter -Name \"{alias}\" | "
        f"Select-Object -First 1 -ExpandProperty InterfaceGuid)"
    )
    if not guid:
        return alias, None

    return alias, guid.strip()


def map_guid_to_tshark_index(guid: str):
    """
    Map InterfaceGuid (XXXXXXXX-XXXX-...) to tshark index by matching
    the GUID inside \\Device\\NPF_{GUID} shown by tshark -D.
    """
    guid_norm = guid.strip().lower()
    ifaces = tshark_list_interfaces()

    candidates = []
    for idx, dev, desc in ifaces:
        d = dev.lower()
        if guid_norm in d or ("{" + guid_norm + "}") in d:
            candidates.append((idx, dev, desc))

    if len(candidates) == 1:
        return candidates[0], ifaces

    # fallback: sometimes GUID appears in description
    candidates = []
    for idx, dev, desc in ifaces:
        if guid_norm in desc.lower():
            candidates.append((idx, dev, desc))

    if len(candidates) == 1:
        return candidates[0], ifaces

    return None, ifaces


# ------------------ capture ------------------
def start_capture(tshark_if_idx: int, phone_ip: str | None, phone_mac: str | None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = phone_ip.replace(".", "_") if phone_ip else "unknown"
    out_file = OUT_DIR / f"capture_{ts}_{tag}.pcapng"

    cap_filter = None
    if CAPTURE_ONLY_PHONE:
        if phone_mac:
            cap_filter = f"ether host {phone_mac}"
        elif phone_ip:
            cap_filter = f"host {phone_ip}"

    cmd = [TSHARK, "-i", str(tshark_if_idx), "-w", str(out_file)]
    if cap_filter:
        cmd += ["-f", cap_filter]

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p, out_file, cap_filter


# ------------------ hotspot (optional) ------------------
def hostednetwork_supported():
    p = run([NETSH, "wlan", "show", "drivers"], capture=True, check=False)
    return bool(re.search(r"Hosted\s+network\s+supported\s*:\s*Yes", (p.stdout or ""), re.IGNORECASE))


def configure_hostednetwork():
    run([NETSH, "wlan", "set", "hostednetwork", "mode=allow", f"ssid={SSID}", f"key={PASSWORD}"],
        capture=True, check=True)


def start_hostednetwork():
    p = run([NETSH, "wlan", "start", "hostednetwork"], capture=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"Failed starting hostednetwork.\n{p.stdout}\n{p.stderr}")


def stop_hostednetwork():
    run([NETSH, "wlan", "stop", "hostednetwork"], capture=True, check=False)


# ------------------ ARP detection ------------------
def parse_arp_table():
    """
    arp -a output groups entries by:
      Interface: <PC_IFACE_IP> --- ...
    Each entry line provides:
      <IP>   <MAC>   <TYPE>
    Returns list of (pc_iface_ip, ip, mac, type)
    """
    p = run(["arp", "-a"], capture=True, check=True)
    lines = (p.stdout or "").splitlines()

    entries = []
    current_iface_ip = None

    iface_re = re.compile(r"^Interface:\s*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)\s+---")
    entry_re = re.compile(r"^\s*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)\s+([0-9a-fA-F\-]{17})\s+(\w+)\s*$")

    for ln in lines:
        m = iface_re.match(ln)
        if m:
            current_iface_ip = m.group(1)
            continue

        m2 = entry_re.match(ln)
        if m2 and current_iface_ip:
            ip = m2.group(1)
            mac = m2.group(2).replace("-", ":").lower()
            typ = m2.group(3).lower()

            if mac.startswith("ff:ff:ff:ff:ff:ff"):
                continue

            entries.append((current_iface_ip, ip, mac, typ))

    return entries


def wait_for_phone_in_arp():
    """
    Wait until a client (phone) appears in ARP stably.
    Returns (pc_iface_ip, phone_ip, phone_mac)
    pc_iface_ip is the PC interface IP (gateway) of the correct interface,
    matching what you see when hovering the correct interface in Wireshark.
    """
    stable = {}
    last_seen = {}

    print("[+] Connect your phone to the hotspot and generate traffic (open a website) so ARP fills.\n")

    while True:
        entries = parse_arp_table()

        candidates = []
        for pc_iface_ip, ip, mac, typ in entries:
            if typ not in ("dynamic", "static"):
                continue
            if not ipaddress.IPv4Address(ip).is_private:
                continue
            if ip == pc_iface_ip:
                continue
            candidates.append((pc_iface_ip, ip, mac))

        if not candidates:
            time.sleep(POLL_SEC)
            continue

        now = time.time()
        for pc_iface_ip, ip, mac in candidates:
            key = (pc_iface_ip, ip, mac)
            last_seen[key] = now
            stable[key] = stable.get(key, 0) + 1

        for key in list(stable.keys()):
            if now - last_seen.get(key, 0) > (POLL_SEC * 2.5):
                stable.pop(key, None)
                last_seen.pop(key, None)

        for (pc_iface_ip, ip, mac), cnt in stable.items():
            if cnt >= ARP_STABLE_COUNT:
                print(f"[+] Detected client: phone_ip={ip} phone_mac={mac}")
                print(f"[+] Correct PC interface IP (from ARP group): pc_iface_ip={pc_iface_ip}")
                return pc_iface_ip, ip, mac

        time.sleep(POLL_SEC)


# ------------------ main ------------------
def main():
    ensure_tools()

    hosted_started = False
    cap_proc = None

    try:
        if TRY_START_HOSTEDNETWORK:
            if not hostednetwork_supported():
                print("[!] Hosted Network is NOT supported on this system/driver.")
                print("[!] Turn on Windows Mobile Hotspot manually, then continue.\n")
            else:
                print(f"[+] Configuring Hosted Network SSID={SSID}")
                configure_hostednetwork()
                print("[+] Starting Hosted Network...")
                start_hostednetwork()
                hosted_started = True
                print("[+] Hosted Network started.\n")
        else:
            print("[i] Turn on Windows Mobile Hotspot manually:")
            print("    Settings -> Network & Internet -> Mobile hotspot")
            print("[i] After the phone connects, open a website on the phone so it appears in ARP.\n")

        # 1) Get correct PC interface IP + phone IP/MAC from ARP
        pc_iface_ip, phone_ip, phone_mac = wait_for_phone_in_arp()

        # 2) Get Interface GUID for that PC interface IP
        alias, guid = get_interface_guid_by_ip(pc_iface_ip)
        if not guid:
            raise SystemExit(
                "Failed to read InterfaceGuid via PowerShell.\n"
                "Run PowerShell as Administrator and verify Get-NetAdapter / Get-NetIPAddress work."
            )

        print(f"[+] InterfaceAlias = {alias}")
        print(f"[+] InterfaceGuid  = {guid}")

        # 3) Map GUID -> tshark interface index
        mapped, all_ifaces = map_guid_to_tshark_index(guid)
        if not mapped:
            print("[!] Failed to map GUID to a tshark interface automatically.")
            print("    tshark -D list:")
            for idx, dev, desc in all_ifaces:
                print(f"    {idx}: {desc} [{dev}]")
            raise SystemExit("GUID->tshark mapping failed.")

        tshark_idx, tshark_dev, tshark_desc = mapped
        print(f"[+] Using tshark interface: {tshark_idx} ({tshark_desc})")

        # 4) Start capture
        cap_proc, out_file, cap_filter = start_capture(
            tshark_if_idx=tshark_idx,
            phone_ip=phone_ip,
            phone_mac=phone_mac
        )
        print(f"[+] Capture started -> {out_file}")
        if cap_filter:
            print(f"[+] Capture filter: {cap_filter}")
        print("[+] Press Ctrl+C to stop.\n")

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[!] Stopping...")

    finally:
        try:
            if cap_proc and cap_proc.poll() is None:
                cap_proc.terminate()
                try:
                    cap_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    cap_proc.kill()
        except Exception:
            pass

        try:
            if hosted_started:
                stop_hostednetwork()
        except Exception:
            pass

        print("[+] Done.")


if __name__ == "__main__":
    main()
