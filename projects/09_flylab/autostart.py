"""
Keep the FlyLab supervisor running without anyone touching it (Windows).

    python autostart.py install   # Startup-folder entry (at sign-in) + hourly watchdog task
    python autostart.py ensure    # start the guardian if it is not already running
    python autostart.py guard     # (run by ensure) keep supervise.py alive, log every exit code
    python autostart.py status    # is it running? what is installed?
    python autostart.py remove    # undo install (does not stop a running supervisor)

`ensure` starts the guardian (through WMI, so it is independent of whatever
ran ensure) unless one is already running. The guardian runs supervise.py as a
child, restarts it within seconds whenever it exits, and logs the exit code to
training/logs/autostart.log, so an unexplained death leaves evidence behind.
Nothing here ever starts a second guardian or a second supervisor.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
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


def pids_matching(pattern):
    for attempt in (1, 2):   # a failed query must not read as "nothing running": retry once
        r = ps("Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='pythonw.exe'\" | "
               f"Where-Object {{ $_.CommandLine -like '*{pattern}*' }} | ForEach-Object {{ $_.ProcessId }}")
        if not (r.returncode or r.stderr.strip()):
            break
        log(f"process query for {pattern!r} failed (code {r.returncode}, attempt {attempt}): {r.stderr.strip()[:200]}")
    return [int(x) for x in r.stdout.split() if x.strip().isdigit()]


def supervisor_pids():
    return pids_matching("supervise.py")


def guardian_pids():
    return pids_matching("autostart.py*guard")   # the command line quotes the path: autostart.py" guard


def guard():
    """Run supervise.py as a child and restart it the moment it exits, logging
    the exit code (1 = killed by TerminateProcess/taskkill, 0xC000013A = console
    closed / Ctrl+Break, other 0xC... = native crash). Never starts a second
    supervisor: if one is already running elsewhere it waits for it to end."""
    others = [p for p in guardian_pids() if p != os.getpid()]
    if len(others) > 2:            # a guardian is a venv launcher + interpreter pair
        log(f"guardian {os.getpid()} found another guardian running ({others}); exiting")
        return
    log(f"guardian started (pid {os.getpid()})")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    quick_deaths = 0
    while True:
        if supervisor_pids():
            time.sleep(60)         # one is already running (started elsewhere); don't duplicate
            continue
        t0 = time.time()
        proc = subprocess.Popen([str(PY), "-u", "supervise.py", "--keep-awake"], cwd=HERE, creationflags=flags)
        log(f"guardian started supervisor (pid {proc.pid})")
        code = proc.wait()
        mins = (time.time() - t0) / 60
        code_u = code & 0xFFFFFFFF
        why = {1: "killed (TerminateProcess / taskkill /F)", 0xC000013A: "console closed or Ctrl+Break",
               0xC0000005: "access violation (native crash)", 0xC0000409: "stack buffer overrun (native crash)"}
        log(f"supervisor exited after {mins:.0f} min with code {code} (0x{code_u:08X}): "
            f"{why.get(code_u, 'see code')}")
        quick_deaths = quick_deaths + 1 if mins < 5 else 0
        time.sleep(min(15 * 2 ** quick_deaths, 600))   # back off if it keeps dying at once


def ensure():
    """Make sure the guardian is running (it keeps the supervisor alive)."""
    pids = guardian_pids()
    if pids:
        print(f"guardian already running (pids {pids}); supervisor pids {supervisor_pids()}")
        return
    cmd = f'"{PYW}" "{Path(__file__)}" guard'
    # 0x208 = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS: no console, immune to Ctrl+C elsewhere
    r = ps(f"$s = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{{CreateFlags=[uint32]0x208}}; "
           f"$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
           f"@{{CommandLine='{cmd}'; CurrentDirectory='{HERE}'; ProcessStartupInformation=$s}}; "
           f"\"$($r.ReturnValue) $($r.ProcessId)\"")
    code, _, pid = r.stdout.strip().partition(" ")
    log(f"guardian was not running; started it (WMI return {code}, pid {pid})" if code == "0"
        else f"FAILED to start guardian: {r.stdout.strip()} {r.stderr.strip()}")


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
    pids, g = supervisor_pids(), guardian_pids()
    print(f"guardian: {'running, pids ' + str(g) if g else 'NOT running'}")
    print(f"supervisor: {'running, pids ' + str(pids) if pids else 'NOT running'}")
    print(f"sign-in entry: {'installed' if STARTUP.exists() else 'missing'} ({STARTUP})")
    r = subprocess.run(["schtasks", "/Query", "/TN", TASK, "/FO", "LIST"], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.split(":")[0].strip() in ("Next Run Time", "Status", "Last Run Time")]
    print(f"watchdog task: {'installed; ' + '; '.join(l.strip() for l in lines) if r.returncode == 0 else 'missing'}")


if __name__ == "__main__":
    {"install": install, "ensure": ensure, "guard": guard, "status": status, "remove": remove}.get(
        (sys.argv[1:] or ["status"])[0], status)()
