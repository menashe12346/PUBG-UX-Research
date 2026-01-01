# record_pcap_final.ps1
# Works with Windows Mobile Hotspot (ICS) and your confirmed dumpcap interface index (5).
# Auto-detects the connected phone (single real device) and captures ONLY its traffic.
# Output: <OutDir>\phone-<ip>-<timestamp>.pcap
#
# Stop: CTRL+C  (script will ALWAYS stop dumpcap so it won't keep running in background)

param(
    [Parameter(Mandatory=$false)]
    [string]$OutDir
)

$WiresharkDir = "C:\Program Files\Wireshark"
$Dumpcap   = Join-Path $WiresharkDir "dumpcap.exe"
$Capinfos  = Join-Path $WiresharkDir "capinfos.exe"

if (!(Test-Path $Capinfos)) { throw "capinfos not found at: $Capinfos" }
if (!(Test-Path $Dumpcap))  { throw "dumpcap not found at: $Dumpcap"  }

# Your confirmed correct capture interface index from: dumpcap -i 5 -a duration:10 ...
$CaptureInterfaceIndex = 5

# Default gateway for Windows Mobile Hotspot / ICS
$HotspotGatewayIp = "192.168.137.1"

function Is-RealIPv4([string]$ip, [string]$gatewayIp) {
    if (-not $ip) { return $false }
    if ($ip -eq "0.0.0.0") { return $false }
    if ($ip -eq $gatewayIp) { return $false }
    if ($ip -eq "255.255.255.255") { return $false }
    # multicast 224-239
    if ($ip -match '^22[4-9]\.' -or $ip -match '^23[0-9]\.') { return $false }
    # subnet broadcast x.x.x.255
    if ($ip -match '\.255$') { return $false }
    return $true
}

Write-Host "Finding hotspot interface by gateway IP ($HotspotGatewayIp)..."

$hotspotIpObj = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -eq $HotspotGatewayIp } |
    Select-Object -First 1

if (-not $hotspotIpObj) {
    throw "Hotspot not found. Make sure Mobile Hotspot is enabled and gateway is $HotspotGatewayIp."
}

$hotspotAlias = $hotspotIpObj.InterfaceAlias
Write-Host "Hotspot InterfaceAlias: $hotspotAlias"

Start-Sleep -Seconds 2

Write-Host "Detecting connected phone (single real device) from Neighbor table..."

$neighbors = Get-NetNeighbor -InterfaceAlias $hotspotAlias -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Select-Object IPAddress, LinkLayerAddress, State

if (-not $neighbors) {
    throw "No neighbors returned. Ensure phone is connected + generating traffic and retry."
}

$real = $neighbors | Where-Object { Is-RealIPv4 $_.IPAddress $HotspotGatewayIp }
$real = @($real)

Write-Host "`nReal candidates:"
$real | Select-Object IPAddress, LinkLayerAddress, State | Format-Table -AutoSize

if ($real.Count -lt 1) {
    throw "No real device found. Open a website on the phone and rerun."
}

# Assuming only one real device (your phone)
$phoneIp  = $real[0].IPAddress
$phoneMac = $real[0].LinkLayerAddress

Write-Host "Auto-selected phone: IP=$phoneIp  MAC=$phoneMac"

# If OutDir not passed, fallback to: <current folder>\captures
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $PWD "captures"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Output path (PCAP only)
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$pcap = Join-Path $OutDir "phone-$phoneIp-$timestamp.pcap"

# Capture filter: use IP filter (very reliable on virtual hotspot interfaces)
$filter = "host $phoneIp"

Write-Host ""
Write-Host "Starting capture on dumpcap interface index: $CaptureInterfaceIndex"
Write-Host "Capture filter: $filter"
Write-Host "Saving to: $pcap"
Write-Host "Press CTRL+C to stop."

# IMPORTANT: Clean up any leftover dumpcap processes from previous runs
$old = Get-Process dumpcap -ErrorAction SilentlyContinue
if ($old) {
    Write-Host "Found existing dumpcap processes from previous runs. Stopping them..."
    $old | Stop-Process -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 300
}

# Build Process
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $Dumpcap
# -F pcap => write PCAP directly (not pcapng)
$psi.Arguments = "-i $CaptureInterfaceIndex -f `"$filter`" -F pcap -w `"$pcap`" --update-interval 5000"
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.RedirectStandardError = $true
$psi.RedirectStandardOutput = $true

$p = New-Object System.Diagnostics.Process
$p.StartInfo = $psi

$capOk = $false

# Catch CTRL+C so we can stop dumpcap for sure
$script:cancelled = $false
$cancelSub = Register-EngineEvent -SourceIdentifier ConsoleCancelEvent -Action {
    $script:cancelled = $true
} | Out-Null

try {
    if (-not $p.Start()) {
        throw "Failed to start dumpcap process."
    }

    $lastPrint = Get-Date
    $lastPackets = -1

    while (-not $p.HasExited) {

        # If user pressed CTRL+C, stop dumpcap explicitly
        if ($script:cancelled) {
            Write-Host "`nCTRL+C detected. Stopping dumpcap (PID=$($p.Id))..."
            try { Stop-Process -Id $p.Id -ErrorAction SilentlyContinue } catch {}
            break
        }

        if (-not $p.StandardError.EndOfStream) {
            $line = $p.StandardError.ReadLine()

            # dumpcap sometimes prints: "Packets: 28229"
            $m = [regex]::Match($line, '(?i)\bPackets:\s*(\d+)')
            if ($m.Success) {
                $pkt = [int]$m.Groups[1].Value
                $now = Get-Date

                if (($now - $lastPrint).TotalSeconds -ge 5 -and $pkt -ne $lastPackets) {
                    Write-Host "packets=$pkt"
                    $lastPrint = $now
                    $lastPackets = $pkt
                }
            }

            if ($line -match '(?i)\berr(or)?\b|unrecognized|Usage:') {
                Write-Host "[ERR] $line"
            }
        }

        Start-Sleep -Milliseconds 50
    }

    # If file exists and has size > 24 bytes (pcap global header is 24 bytes)
    if (Test-Path $pcap) {
        $len = (Get-Item $pcap).Length
        if ($len -ge 24) { $capOk = $true }
    }
}
finally {
    # Unregister CTRL+C event handler
    try { Unregister-Event -SourceIdentifier ConsoleCancelEvent -ErrorAction SilentlyContinue } catch {}
    try { Remove-Event -SourceIdentifier ConsoleCancelEvent -ErrorAction SilentlyContinue } catch {}

    # Always stop dumpcap if still running (prevents background capture)
    if ($p -and -not $p.HasExited) {
        Write-Host "`nStopping dumpcap process (PID=$($p.Id))..."
        try { Stop-Process -Id $p.Id -ErrorAction SilentlyContinue } catch {}
        Start-Sleep -Milliseconds 300
    }

    if ($capOk) {
        Write-Host "`nCapture completed."
        Write-Host "Saved PCAP: $pcap"
    } else {
        Write-Host "`nCapture failed or produced no usable file."
        if (Test-Path $pcap) {
            Write-Host "File exists but may be empty/partial: $pcap"
        }
    }
}
