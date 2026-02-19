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
---

## 2. Recording Network Traffic (PCAP Capture)

To capture PUBG network traffic:

### Step 1 — Create a Hotspot from the Computer

- Open a **mobile hotspot from your computer**
- Connect your phone to this hotspot
- Make sure PUBG traffic flows through the computer

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

# Creating Metadata (Ping Image Processing)

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

## Feature Extraction

Unlike most mobile FPS games analyzed in this research, **PUBG Mobile runs multiple parallel network streams simultaneously**.  
Therefore, identifying the correct streams before extracting features was a critical step.

The goal was to **precisely identify all PUBG-related streams** and extract features only from them.

---

### Step 1 — Identify PUBG Server IPs

For each PCAP file:

1. Extract all participating IPs  
   ```bash
   python get_ip_participation_summary.py
   ```

2. Generate organized sub-IP prefixes  
   ```bash
   python make_subips.py
   ```

3. For each sub-IP:
   - Run `whois`
   - Verify whether the prefix belongs to **Tencent**
   ```bash
   split_by_tencent.py
   ```

Since PUBG servers are operated by Tencent, any IP range owned by Tencent was considered a PUBG server candidate.

The final list of Tencent-related IPs is stored under:

```text
out/tencent/
```

---

### Step 2 — Discover High-Frequency Ports

In parallel, identify which ports are most actively used across all PCAP files:

```bash
python Port_Discovery.py
```

Additionally:
- Cross-reference with official PUBG documentation and online sources
- Filter streams based on the discovered high-frequency ports

This produced a **second filtered stream list**.

---

### Step 3 — Merge and Clean Stream Lists

- Combine the Tencent-based stream list
- Combine the port-based stream list
- Remove duplicates
- Generate a final authoritative PUBG stream list

At this point, we have the **final list of PUBG streams**.

---

## Extracting Features from Final Stream List

Once the final stream list is ready, run:

```bash
python new___analyze.py
```

This script:
- Filters traffic based on the identified PUBG streams
- Performs aggregation and feature extraction
- Outputs a final `.csv` file inside:

```text
band/
```

This CSV file contains the extracted features used for further modeling and analysis.

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

It helped determine for the first 100 recordingsx that missing classifications cluster most frequently at the beginning, which is expected, since it typically takes around thirty seconds on average for a game to start, and during that period there is no ping activity.

---

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
