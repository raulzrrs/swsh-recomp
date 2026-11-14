"""scenario.py: runs one Sword 1.3.2 test scenario and records the evidence (docs/engineering/README.md).

  python tools/swsh/scenario.py <scenario> [--profile 1440p30|1440p60|1080p30|...] [--cache warm|cold]
                               [--runs N] [--validation [sync|gpuav]] [--label L] [--exe build/v132/lasrecomp.exe]

A scenario is tests/scenarios/<name>.json:
  {"save": "build/scenario-saves/stairs", "input": "600:A:8,...", "exit_at": 3000, "measure": [2800, 3000],
   "shots": [2735], "env": {"VAR": "x"}, "description": "..."}
Frame numbers are game presents at 30 fps; a 60 fps profile doubles them (the script and the shots stay at the same
game time). The run never touches the player's save (save-sword-132), the player's shader cache (cache/) or any
process it did not start:
  - the save is copied to a fresh folder per run;
  - "warm" uses build/testcache (seeded once from cache/), "cold" an empty folder (LARECOMP_CACHE_DIR);
  - a run waits while another Sword 1.3.2 instance is running (a second instance starting while one exits dies in
    ~2 s), and a run that overruns its timeout is ended by its own process id.
Output: reports/runs/<scenario>/<YYYYmmdd-HHMMSS>-<label>/ (log.txt, frametimes.csv, frames/, meta.json, summary.json).
reports/runs/ is ignored by git (game images); the summaries are copied into the cycle reports.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import frame_stats, parse_log  # noqa: E402

PROFILES = {  # render height, fps option
    "1440p30": (1440, 30), "1440p60": (1440, 60), "1080p30": (1080, 30), "1080p60": (1080, 60), "720p30": (720, 30), "720p60": (720, 60),
}
MSYS = r"C:\msys64\ucrt64\bin"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def sword_instances():
    """Process ids of running Sword 1.3.2 runtimes (command line with extracted-132), ours or the player's. One that has
    exited but is still being torn down is left out (a validation-layer run left 2 million handles for Windows to close:
    it stayed listed for hours, its last thread in the driver)."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='lasrecomp.exe'\" | "
          "Where-Object { $_.CommandLine -like '*extracted-132*' } | "
          "Where-Object { $p = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue; $p -and -not $p.HasExited } | "
          "ForEach-Object { $_.ProcessId }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def other_runtimes():
    """Running lasrecomp.exe that are not a Sword 1.3.2 run of ours (another project's tests share the GPU and the CPU).
    Only active ones count: a working set over 500 MB (a stopped leftover sits at ~40 MB, a running game uses GBs)."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='lasrecomp.exe'\" | "
          "Where-Object { $_.CommandLine -notlike '*extracted-132*' -and $_.WorkingSetSize -gt 500MB } | "
          "ForEach-Object { $_.ProcessId }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def wait_idle(limit_s):
    t0 = time.time()
    while sword_instances():
        if time.time() - t0 > limit_s:
            return False
        time.sleep(2)
    return True


def scale_input(spec, k):
    out = []
    for item in filter(None, spec.split(",")):
        parts = item.split(":")
        parts[0] = str(int(parts[0]) * k)
        if len(parts) >= 3 and parts[-1].isdigit() and not parts[1].startswith("shot"):
            parts[-1] = str(int(parts[-1]) * k)
        out.append(":".join(parts))
    return ",".join(out)


def run_once(name, sc, args, run_dir):
    height, fps = PROFILES[args.profile]
    k = 2 if fps == 60 else 1
    os.makedirs(run_dir, exist_ok=True)
    save = os.path.join(run_dir, "save")
    shutil.copytree(os.path.join(ROOT, sc["save"]), save)
    ini = os.path.join(run_dir, "larecomp.ini")
    with open(ini, "w") as f:
        f.write(f"fullscreen = 0\nfps = {fps}\nrender_height = {height}\nsmaa = 1\nupscaler = fsr1\nshow_fps = 0\n"
                f"render_thread = 1\nvertex_read = auto\n")
        for kv in args.ini or []:  # (later lines win: --ini fullscreen=1 runs the profile in full screen)
            ik, _, iv = kv.partition("=")
            f.write(f"{ik.strip()} = {iv.strip()}\n")
    if args.cache == "cold":
        cache = os.path.join(run_dir, "coldcache")
        os.makedirs(cache)
    else:
        cache = os.path.join(ROOT, "build", "testcache")
        if not os.path.isdir(cache):
            shutil.copytree(os.path.join(ROOT, "cache"), cache)
    frames = os.path.join(run_dir, "frames")
    exit_at = sc["exit_at"] * k
    env = dict(os.environ)
    env["PATH"] = MSYS + os.pathsep + env.get("PATH", "")
    env.update({
        "LARECOMP_SETTINGS": ini, "LARECOMP_SAVE_DIR": save, "LARECOMP_CACHE_DIR": cache,
        "LARECOMP_FRAMETIME_CSV": os.path.join(run_dir, "frametimes.csv"), "LARECOMP_EXIT_AT_PRESENT": str(exit_at),
        "LARECOMP_NO_LAUNCHER": "1", "LARECOMP_NO_USER_INPUT": "1",
        # the game lights its scenes by the time of day: every run starts at the same clock (20/10/2026 14:00 -03)
        "LARECOMP_FIXED_TIME": str(sc.get("fixed_time", 1792515600)),
        # no shader preparation (runtime/nvn/vulkan/prep.c): minutes of background work that would compete with what a
        # run measures; the "prep" scenario turns it on
        "LARECOMP_SHADER_PREP": "0",
    })
    if sc.get("input"):
        env["LARECOMP_INPUT"] = scale_input(sc["input"], k)
    shots = [s * k for s in sc.get("shots", [])]
    if shots:
        env["LARECOMP_SHOT_AT"] = ",".join(map(str, sorted(shots)))
    for key, val in sc.get("env", {}).items():
        env[key] = str(val)
    for kv in args.env or []:
        key, _, val = kv.partition("=")
        env[key] = val
    if args.validation:
        env["LARECOMP_VK_VALIDATION"] = "1"
        env["VK_ADD_LAYER_PATH"] = MSYS
        if "sync" in args.validation:
            env["VK_LAYER_VALIDATE_SYNC"] = "1"
        if "gpuav" in args.validation:
            env["VK_LAYER_GPUAV_ENABLE"] = "1"
    timeout = sc.get("timeout_s", 120 + exit_at / 12)  # (an outdoor scene ran at 19 fps: room for a slow build)
    if args.validation:
        timeout *= 4
    if not wait_idle(120):
        return {"status": "BLOCKED", "reason": "another Sword 1.3.2 instance kept running (the player's?)"}
    # benchmarks need the machine to themselves: wait (up to LARECOMP_IDLE_WAIT_S, default 600) for other runtimes
    # to end; a run that still shares the machine is marked contaminated (ab.py / baseline leave it out)
    t_idle = time.time()
    while other_runtimes() and time.time() - t_idle < float(os.environ.get("LARECOMP_IDLE_WAIT_S", "600")):
        time.sleep(5)
    others_at_start = other_runtimes()
    exe = os.path.join(ROOT, args.exe)
    t0 = time.time()
    before = set(sword_instances())
    seen_others = []  # (sampled every 5 s while the run lasts)
    mem_mb = []       # this run's working set, MB, every 5 s (leaks over long sessions)
    stop_watch = threading.Event()

    def watch():
        while not stop_watch.wait(5):
            o = other_runtimes()
            if o:
                seen_others.extend(o)
            ps = ("Get-CimInstance Win32_Process -Filter \"Name='lasrecomp.exe'\" | Where-Object { $_.CommandLine -like "
                  "'*extracted-132*' } | Measure-Object -Property WorkingSetSize -Maximum | ForEach-Object { $_.Maximum }")
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.strip()
            if out.replace(".", "").isdigit():
                mem_mb.append(round(float(out) / 2**20))
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    with open(os.path.join(run_dir, "log.txt"), "wb") as log:
        p = subprocess.Popen([exe, "extracted-132", "--play"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = p.wait(timeout=timeout)
            status = "OK"
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
            code = p.wait()
            status = "TIMEOUT"
        # The runtime re-executes itself when its guest address range is taken (main.c reserve_guest_space): the
        # process we started exits 0 and a child goes on writing the same log. Follow such a child to its end.
        children = [pid for pid in sword_instances() if pid not in before and pid != p.pid]
        while children and status == "OK":
            if time.time() - t0 > timeout:
                for pid in children:
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
                status = "TIMEOUT"
                break
            time.sleep(1)
            children = [pid for pid in sword_instances() if pid in children]
        if children == [] and code == 0 and time.time() - t0 < 30:
            code = 0  # (re-executed child finished: its exit is the run's)
    secs = time.time() - t0
    stop_watch.set()
    contaminated = bool(others_at_start or seen_others or other_runtimes())
    if os.path.isdir(os.path.join(ROOT, "build", "frames")):  # (LARECOMP_SHOT_AT writes build/frames/)
        shutil.move(os.path.join(ROOT, "build", "frames"), frames)
    with open(os.path.join(run_dir, "log.txt"), encoding="utf-8", errors="replace") as f:
        logtext = f.read()
    info = parse_log(logtext, sc["measure"][0] * k, sc["measure"][1] * k)
    if info["crashes"]:
        status = "CRASH"
    elif status == "OK" and code == 1:
        # taskkill /F's exit code, with no FATAL / exception in the log: another tool ended every lasrecomp.exe by
        # image name (letsgo-recomp's old tools/trace_run.sh did) - not this build's failure; main() reruns it
        status = "EXTERNAL-KILL"
    elif status == "OK" and code != 0:
        status = f"EXIT {code}"
    ft = frame_stats(os.path.join(run_dir, "frametimes.csv"), sc["measure"][0] * k, sc["measure"][1] * k, exclude=shots)
    dll = os.path.join(os.path.dirname(exe), "lasgame.dll")
    meta = {
        "scenario": name, "profile": args.profile, "cache": args.cache, "validation": args.validation or "",
        "exe": args.exe, "exe_sha256": sha256(exe), "dll_sha256": sha256(dll) if os.path.exists(dll) else "",
        "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no", "runtime", "tools")),
        "gpu": info.get("gpu", ""), "os": f"{os.name} {sys.getwindowsversion().major}.{sys.getwindowsversion().build}"
        if hasattr(sys, "getwindowsversion") else os.name,
        "env": {k_: v for k_, v in env.items() if k_.startswith(("LARECOMP_", "VK_"))},
        "started": datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
    }
    mem = {"first_mb": mem_mb[0], "max_mb": max(mem_mb), "last_mb": mem_mb[-1], "samples": len(mem_mb),
           "series_mb": mem_mb[::12]} if mem_mb else None
    summary = {"status": status, "exit_code": code, "seconds": round(secs, 1), "contaminated": contaminated, "memory": mem,
               "frametime": ft, **{
        k_: v for k_, v in info.items() if k_ != "gpu"}}
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario")
    ap.add_argument("--profile", default="1440p30", choices=sorted(PROFILES))
    ap.add_argument("--cache", default="warm", choices=["warm", "cold"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--validation", nargs="?", const="basic", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--exe", default=os.path.join("build", "v132", "lasrecomp.exe"))
    ap.add_argument("--env", action="append", help="VAR=value for this run (repeatable)")
    ap.add_argument("--ini", action="append", help="key=value added to the run's larecomp.ini, over the profile's (repeatable)")
    args = ap.parse_args()
    with open(os.path.join(ROOT, "tests", "scenarios", args.scenario + ".json")) as f:
        sc = json.load(f)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    label = args.label or f"{args.profile}-{args.cache}" + (f"-val{args.validation}" if args.validation else "")
    results = []
    for i in range(args.runs):
        run_dir = os.path.join(ROOT, "reports", "runs", args.scenario, f"{stamp}-{label}-{i + 1}")
        s = run_once(args.scenario, sc, args, run_dir)
        for retry in range(2):  # (an external kill says nothing about the build: run it again)
            if s["status"] != "EXTERNAL-KILL":
                break
            print(json.dumps({"run": os.path.relpath(run_dir, ROOT), "status": s["status"], "retry": retry + 1}), flush=True)
            run_dir = f"{run_dir}-retry{retry + 1}"
            s = run_once(args.scenario, sc, args, run_dir)
        results.append(s)
        ft = s.get("frametime") or {}
        print(json.dumps({"run": os.path.relpath(run_dir, ROOT), "status": s["status"], "contaminated": s.get("contaminated"), "fps": ft.get("fps_mean"),
                          "p99_ms": ft.get("p99_ms"), "skipped": s.get("skipped_total"),
                          "vkval_errors": s.get("vkval_errors")}), flush=True)
    sys.exit(0 if all(r["status"] == "OK" for r in results) else 1)


if __name__ == "__main__":
    main()
