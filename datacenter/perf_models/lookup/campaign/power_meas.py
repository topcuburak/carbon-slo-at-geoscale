#!/usr/bin/env python3
import csv
import os
import time
import subprocess
from datetime import datetime
import glob
import requests

def run(cmd: list[str]) -> str:
  return subprocess.check_output(cmd, text=True).strip()

def read_int(path: str) -> int:
  with open(path, "r") as f:
    return int(f.read().strip())

def get_cpu_energy_uj() -> int | None:
  cpu_total = 0
  psys_total = 0
  #for name_path in glob.glob("/sys/class/powercap/intel-rapl:*/name"):
  #  name = open(name_path).read().strip().lower()
  #  epath = name_path.replace("/name", "/energy_uj")
  #  if name.startswith("package-"):
  #    cpu_total += int(open(epath).read().strip())
  #  elif name == "psys" or "psys" in name:
  #    psys_total += int(open(epath).read().strip())
  for name_path in glob.glob("/sys/class/powercap/intel-rapl:*/name"):
    try:
      with open(name_path) as f:
        name = f.read().strip().lower()
    except Exception:
      continue  # cannot read the name → skip
    epath = name_path.replace("/name", "/energy_uj")
    try:
      with open(epath) as f:
        energy = int(f.read().strip())
    except Exception:
      continue  # permission denied or missing → skip

    if name.startswith("package-"):
        cpu_total += energy
    elif name == "psys" or "psys" in name:
        psys_total += energy  

  #if (cpu_total == 0 and psys_total == 0):
  #  print("No CPU or PSYS package domains found in RAPL")
  return cpu_total, psys_total

def get_gpu_power_w() -> list[float]:
  try:
    out = run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"])
    power_draws = [float(line.strip()) for line in out.splitlines()]
    return power_draws
  except Exception:
    return []

def get_carbon_intensity_pjm(zone: str) -> float:
  # export ELECTRICITY_MAPS_TOKEN=<your_token>  (free for research, https://api-portal.electricitymaps.com/)
  token = os.getenv("ELECTRICITY_MAPS_TOKEN")
  if not token:
    raise RuntimeError("Please set ELECTRICITY_MAPS_TOKEN environment variable")
  
  url = "https://api.electricitymap.org/v3/carbon-intensity/latest"
  params = {"zone": zone}
  headers = {"auth-token": token}

  resp = requests.get(url, params=params, headers=headers, timeout=10)
  resp.raise_for_status()  # Raise an error if bad response
  data = resp.json()
  # Extract intensity value (gCO2eq/kWh)
  intensity = data.get("carbonIntensity") or data.get("value")
  return intensity

def main(out_csv="power_log.csv", interval_s=1.0, duration_s=None, args=None):
  ci = None
  if args.CO2_en:
    try:
      ci = get_carbon_intensity_pjm(args.CI_region)
      print(f"Current carbon intensity (gCO2eq/kWh): {ci}")
    except Exception as e:
      print("Failed to fetch carbon intensity:", e)

  # Prime energies for delta-based power (J/s = W)
  prev_t = time.time()
  prev_cpu_uj, prev_psys_uj = get_cpu_energy_uj()
  nof_gpus = len(get_gpu_power_w())
 
  with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
      "timestamp",
      *[f"gpu_power_w_{i}" for i in range(nof_gpus)],
      "cpu_pkg_power_w",
      "psys_power_w",
      "cpu_energy_j",
      "psys_energy_j",
      f"CI_gCO2eq_per_kWh = {ci}" if ci is not None else None
    ])

    start = time.time()
    while True:
      now = time.time()
      ts = int(round(now - start, 1))
      gpu_w = get_gpu_power_w()

      cpu_uj, psys_uj = get_cpu_energy_uj()

      dt = now - prev_t if now > prev_t else None
      cpu_pkg_w = None
      psys_pkg_w = None
      cpu_j = (cpu_uj - prev_cpu_uj) / 1e6 if cpu_uj is not None else None
      psys_j = (psys_uj - prev_psys_uj) / 1e6 if psys_uj is not None else None 

      if dt and cpu_uj is not None and prev_cpu_uj is not None:
        cpu_pkg_w = ((cpu_uj - prev_cpu_uj) / 1e6) / dt

      if dt and psys_uj is not None and prev_psys_uj is not None:
        psys_pkg_w = ((psys_uj - prev_psys_uj) / 1e6) / dt

      w.writerow([ts, *[round(g, 2) for g in gpu_w], 
          round(cpu_pkg_w, 2) if cpu_pkg_w is not None else None, 
          round(psys_pkg_w, 2) if psys_pkg_w is not None else None, 
          round(cpu_j, 2) if cpu_j is not None else None, 
          round(psys_j, 2) if psys_j is not None else None])
      f.flush()
      prev_t = now
      prev_cpu_uj = cpu_uj
      prev_psys_uj = psys_uj

      time.sleep(interval_s)
      if duration_s and now - start >= duration_s:
        break
      try:
        if not run(["pgrep", "-f", "model_calls.py"]):
          print("model_calls.py process not found, stopping...")
          break
      except Exception:
        pass

import argparse
if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--interval_sec", type=float, default=1.0, 
                  help="Interval in seconds between power measurements")
  ap.add_argument("--CPU", type=bool, default=True,
                  help="Collect CPU power measurements")
  ap.add_argument("--GPU", type=bool, default=True,
                  help="Collect GPU power measurements")
  ap.add_argument("--duration_sec", type=float, default=None,
                  help="Duration in seconds to collect power measurements")
  ap.add_argument("--output_file", type=str, default='power_log.csv',
                  help="Output file to save power measurements")
  ap.add_argument("--CO2_en", type=bool, default=True,
                  help="Carbon emission collection enabled? (default: True)")
  ap.add_argument("--CI_region", type=str, default='US-MIDA-PJM',
                  help="What's the electricity grid region for CO2 conversion? E.g., US-PJM")
  args = ap.parse_args()

  main(out_csv=args.output_file,
       interval_s=args.interval_sec,
       duration_s=args.duration_sec, 
       args=args)
  
