# Recording PUBG Ping Screenshots and Network Traffic (PCAP)

This section explains how to record PUBG gameplay data at two levels:

- 📶 Ping screenshots (from the phone screen)
- 🌐 Network traffic (PCAP capture from the computer)

You can record each separately or both simultaneously.

---

## 1. Recording Ping Screenshots (ADB Method)

To capture in-game ping screenshots from your phone:

### Step 1 — Connect the Phone

- Connect your phone to the computer using a **USB cable**
- Enable **Developer Options** on the phone
- Enable **USB Debugging**
- Approve the ADB authorization prompt on the phone

Verify that the device is connected:

```bash
adb devices
```

You should see your device listed.

---

### Step 2 — Run the Screenshot Script

To record only ping screenshots:

```bash
python adb/screencap_phone_then_pull.py
```

This script:
- Captures screenshots directly from the phone
- Pulls them to the computer
- Saves them locally for later ping extraction

Use this mode when you only need visual ping data.

---

## 2. Recording Network Traffic (PCAP Capture)

To capture PUBG network traffic:

### Step 1 — Create a Hotspot from the Computer

- Open a **mobile hotspot from your computer**
- Connect your phone to this hotspot
- Make sure PUBG traffic flows through the computer

This ensures the computer can capture the phone’s network packets.

---

### Step 2 — Manual Capture via Wireshark

If you want to record traffic manually:

1. Open **Wireshark**
2. Select the correct network interface (the hotspot interface)
3. Start capture
4. Play PUBG
5. Stop capture
6. Save the file as `.pcap` or `.pcapng`

This method is useful for manual or quick captures.

---

## 3. Recording Both Ping Screenshots and PCAP (Recommended)

For synchronized recording of both screenshots and network traffic:

Run:

```bash
python record_all.py
```

Before running, update the relevant arguments inside:

```powershell
record_pcap.ps1
```

## Creating Metadata (Ping Image Processing)

To generate metadata from the recorded ping screenshots, run:

```bash
python image_proses/build_metadata.py
```

This script:
- Uses the digit templates located under `digits_font`
- Extracts ping values from the screenshots
- Produces a full debug output folder named:

```text
debug_all_pings/
```

The `debug_all_pings` directory is useful for verifying detection quality and troubleshooting cases where digits were misread.

## Drawing Conclusions from Feature Extraction (Audit & Validation)

After generating the final feature CSV file, an `audit/` directory is automatically created.  
Inside this directory, three types of audit files are generated to validate stream classification quality.

---

### 1️⃣ Per-PCAP Detailed Audit

For each PCAP file:

```text
audit_samples__{folder}__{pcap_file}.csv
```

This file explains for **every detected stream**:

- Whether it was identified as PUBG or not
- If identified as PUBG:
  - Was it detected based on **IP (Tencent ownership)**?
  - Or based on **Port filtering**?

This file provides full transparency at the stream level.

---

### 2️⃣ Per-PCAP Summary

For each PCAP file:

```text
audit_summary__{folder}__{pcap_file}.csv
```

This file summarizes:
- Total number of streams
- Number identified as PUBG
- Number identified via IP
- Number identified via Port
- Number of missed classifications

It serves as a high-level validation report per capture.

---

### 3️⃣ Global Aggregate Audit

A global summary file is also generated:

```text
audit_aggregate_all_runs_{int(time())}.csv
```

This file aggregates results across **all PCAP runs**.

It provides:
- Overall detection rates
- Combined statistics
- Cross-run performance consistency

---

## Missing Stream Analysis

To further analyze classification gaps, run:

```bash
python audit_missing_overall.py
```

This script:

1. Iterates over all `audit_samples` files inside the `audit/` directory
2. For each file:
   - Counts how many rows are marked as `"missed"` in the `category` column
3. Splits each file into three logical segments:
   - First quarter
   - Middle half
   - Final quarter

For each segment, it computes:

- Internal missing density (percentage of missing inside the segment)
- Share of total missing that originated from that segment

---

## Final Output Statistics

At the end, the script reports:

- Per-file missing distribution
- Overall averages across all runs
- Simple mean between files
- Weighted averages:
  - Weighted by total number of rows
  - Weighted by number of missing samples

This analysis helps determine:

- Whether missing classifications cluster at the beginning, middle, or end of sessions
- Whether detection quality is stable across runs
- Whether stream identification logic requires refinement

## Data Exploration

At this stage we explore the extracted dataset and run the modeling notebook:

```text
new_____data__exploration.ipynb
```

The notebook also uses the following helper files:

```text
rank.py
modelse.py
consts.py
```
