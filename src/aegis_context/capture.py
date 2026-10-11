"""Screen + active-window capture. Needs `uv sync --extra capture` for screenshots."""
from __future__ import annotations

import platform
import sys
from dataclasses import dataclass


@dataclass
class WindowInfo:
    bundle_id: str
    app_name: str
    title: str
    platform: str
    pid: int = 0


def _platform() -> str:
    return {"Windows": "windows", "Darwin": "macos", "Linux": "linux"}.get(platform.system(), "unknown")


def active_window() -> WindowInfo:
    plat = _platform()
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            u32, k32 = ctypes.windll.user32, ctypes.windll.kernel32
            hwnd = u32.GetForegroundWindow()
            buf = ctypes.create_unicode_buffer(512)
            u32.GetWindowTextW(hwnd, buf, 512)
            pid = wintypes.DWORD()
            u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            h = k32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
            exe = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            k32.QueryFullProcessImageNameW(h, 0, exe, ctypes.byref(size))
            k32.CloseHandle(h)
            name = exe.value.rsplit("\\", 1)[-1] or "unknown.exe"
            return WindowInfo(name.lower(), name.removesuffix(".exe"), buf.value, plat, int(pid.value))
        if sys.platform == "darwin":
            import subprocess

            out = subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to get name of first process whose frontmost is true'],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            return WindowInfo(out.lower(), out, "", plat)
        import subprocess

        title = subprocess.run(["xdotool", "getactivewindow", "getwindowname"],
                               capture_output=True, text=True, timeout=2).stdout.strip()
        return WindowInfo("x11", "X11", title, plat)
    except Exception:
        return WindowInfo("unknown", "Unknown", "", plat)


@dataclass
class Frame:
    png: bytes
    dhash: int | None = None


def dhash(img, size: int = 16) -> int:
    """Perceptual difference hash (size*size bits): stable under cursor blinks and tiny redraws."""
    g = img.convert("L").resize((size + 1, size))
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        row = r * (size + 1)
        for c in range(size):
            bits = (bits << 1) | (px[row + c] > px[row + c + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def grab_frame(max_side: int = 1280) -> Frame:
    import mss
    from PIL import Image

    from .backends import png_bytes

    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[1])
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    img.thumbnail((max_side, max_side))
    return Frame(png_bytes(img), dhash(img))


def grab_screen_png(max_side: int = 1280) -> bytes:
    return grab_frame(max_side).png


def idle_seconds() -> float:
    """Seconds since the last keyboard/mouse input (Windows); 0 elsewhere."""
    if sys.platform != "win32":
        return 0.0
    import ctypes

    class _LII(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    lii = _LII()
    lii.cbSize = ctypes.sizeof(lii)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
        return 0.0
    return ((ctypes.windll.kernel32.GetTickCount() - lii.dwTime) & 0xFFFFFFFF) / 1000.0
