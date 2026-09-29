"""
Keep the FlyLab supervisor running without anyone touching it (Windows).

    python autostart.py install   # Startup-folder entry (at sign-in) + hourly watchdog task
    python autostart.py ensure    # start supervise.py if it is not already running
    python autostart.py status    # is it running? what is installed?
    python autostart.py remove    # undo install (does not stop a running supervisor)

`ensure` is idempotent: if a supervise.py process exists it does nothing, so the
sign-in entry and the hourly task can never start a second copy. The supervisor
is launched through WMI (Win32_Process.Create), which keeps it independent of
whatever started it, so it survives the launching console or task ending.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PY = REPO / ".venv-sim" / "Scripts" / "python.exe"
PYW = REPO / ".venv-sim" / "Scripts" / "pythonw.exe"
TASK = "FlyLab supervisor watchdog"
STARTUP = Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs/Startup/FlyLab supervisor.cmd"
LOG = HERE / "training" / "logs" / "autostart.log"


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now().isoformat(timespec='seconds')}  {msg}\n")
    print(msg)


def ps(script):
    return subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                          capture_output=True, text=True)


def supervisor_pids():
    r = ps("Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='pythonw.exe'\" | "
           "Where-Object { $_.CommandLine -like '*supervise.py*' } | ForEach-Object { $_.ProcessId }")
    return [int(x) for x in r.stdout.split() if x.strip().isdigit()]


def ensure():
    pids = supervisor_pids()
    if pids:
        print(f"supervisor already running (pids {pids})")
        return
    cmd = f'"{PY}" -u supervise.py --keep-awake'
    # 0x208 = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS: no console, immune to Ctrl+C elsewhere
    r = ps(f"$s = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{{CreateFlags=[uint32]0x208}}; "
           f"$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
           f"@{{CommandLine='{cmd}'; CurrentDirectory='{HERE}'; ProcessStartupInformation=$s}}; "
           f"\"$($r.ReturnValue) $($r.ProcessId)\"")
    code, _, pid = r.stdout.strip().partition(" ")
    log(f"supervisor was not running; started it (WMI return {code}, pid {pid})" if code == "0"
        else f"FAILED to start supervisor: {r.stdout.strip()} {r.stderr.strip()}")


def install():
    STARTUP.parent.mkdir(parents=True, exist_ok=True)
    STARTUP.write_text(f'@echo off\r\nstart "" "{PYW}" "{Path(__file__)}" ensure\r\n', encoding="ascii")
    r = subprocess.run(["schtasks", "/Create", "/F", "/TN", TASK, "/SC", "HOURLY", "/MO", "1",
                        "/TR", f'"{PYW}" "{Path(__file__)}" ensure'], capture_output=True, text=True)
    log(f"installed sign-in entry: {STARTUP}")
    log(f"hourly watchdog task '{TASK}': {'ok' if r.returncode == 0 else r.stderr.strip() or r.stdout.strip()}")


def remove():
    if STARTUP.exists():
        STARTUP.unlink()
    r = subprocess.run(["schtasks", "/Delete", "/F", "/TN", TASK], capture_output=True, text=True)
    log(f"removed sign-in entry and watchdog task ({'ok' if r.returncode == 0 else r.stderr.strip()})")


def status():
    pids = supervisor_pids()
    print(f"supervisor: {'running, pids ' + str(pids) if pids else 'NOT running'}")
    print(f"sign-in entry: {'installed' if STARTUP.exists() else 'missing'} ({STARTUP})")
    r = subprocess.run(["schtasks", "/Query", "/TN", TASK, "/FO", "LIST"], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.split(":")[0].strip() in ("Next Run Time", "Status", "Last Run Time")]
    print(f"watchdog task: {'installed; ' + '; '.join(l.strip() for l in lines) if r.returncode == 0 else 'missing'}")


if __name__ == "__main__":
    {"install": install, "ensure": ensure, "status": status, "remove": remove}.get(
        (sys.argv[1:] or ["status"])[0], status)()
