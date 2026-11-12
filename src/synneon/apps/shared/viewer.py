"""
Persistent capture viewer.

Reads newline-delimited JSON from stdin. Each message has the shape:
    {"image_path": "...", "result": {...}}

The single Tkinter window updates in-place on each message.
Run as: python -m synneon.apps.shared.viewer
"""

from __future__ import annotations

import io
import json
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk


def _load_photo(root: tk.Tk, image_path: Path) -> tk.PhotoImage | None:
    try:
        from PIL import Image

        img = Image.open(image_path)
        max_w, max_h = 1600, 800
        scale = min(max_w / img.width, max_h / img.height, 1.0)
        if scale < 1.0:
            resample = getattr(Image, "Resampling", Image).LANCZOS
            img = img.resize((int(img.width * scale), int(img.height * scale)), resample)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return tk.PhotoImage(data=buf.getvalue())
    except Exception as e:
        print(f"[viewer] image load error: {e}", file=sys.stderr)
        return None


class CaptureViewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Synneon Image Context Analysis")
        root.geometry("1800x700")

        paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.frame_left = ttk.Frame(paned)
        paned.add(self.frame_left, weight=4)

        self.frame_right = ttk.Frame(paned)
        paned.add(self.frame_right, weight=1)

        self.label_img = ttk.Label(self.frame_left)
        self.label_img.pack(expand=True, padx=5, pady=5)

        self.text = tk.Text(self.frame_right, wrap="word", padx=10, pady=10, font=("Helvetica", 12))
        self.text.pack(fill=tk.BOTH, expand=True)
        self.text.tag_configure("h1", font=("Helvetica", 18, "bold"), foreground="white", spacing3=20, spacing1=10)
        self.text.tag_configure("h2", font=("Helvetica", 11, "bold"), foreground="#7F8C8D", spacing3=2)
        self.text.tag_configure("content", font=("Helvetica", 13), spacing1=2, spacing3=12, lmargin1=10)

        # Queue of (image_path, result) updates
        self._pending: list[tuple[Path, dict]] = []
        self._lock = threading.Lock()

        # Poll for updates every 100 ms
        root.after(100, self._poll)

    def update(self, image_path: Path, result: dict):
        with self._lock:
            self._pending.append((image_path, result))

    def _poll(self):
        with self._lock:
            pending = self._pending[:]
            self._pending.clear()

        for image_path, result in pending:
            self._apply(image_path, result)

        self.root.after(100, self._poll)

    def _apply(self, image_path: Path, result: dict):
        # Update image
        photo = _load_photo(self.root, image_path)
        if photo:
            self.label_img.configure(image=photo)
            self.label_img.image = photo  # keep reference
        else:
            self.label_img.configure(image="", text=f"Error loading\n{image_path.name}")

        # Update text
        visual = result.get("parsed_visual", {})
        pragmatic = result.get("pragmatic", {})

        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)

        self.text.insert(tk.END, "Visual Analysis\n", "h1")
        self.text.insert(tk.END, "PRIMARY FOCUS\n", "h2")
        self.text.insert(tk.END, f"{(visual.get('focused_object') or 'None').capitalize()}\n", "content")
        self.text.insert(tk.END, "PERIPHERAL OBJECTS\n", "h2")
        det_objs = visual.get("detected_objects", [])
        self.text.insert(tk.END, f"{', '.join(det_objs).capitalize() if det_objs else 'None'}\n", "content")

        self.text.insert(tk.END, "\nPragmatic Inference\n", "h1")
        self.text.insert(tk.END, "COMMUNICATIVE INTENT\n", "h2")
        self.text.insert(tk.END, f"{(pragmatic.get('intent') or 'Unknown').capitalize()}\n", "content")
        self.text.insert(tk.END, "SOCIAL AUDIENCE\n", "h2")
        self.text.insert(tk.END, f"{(pragmatic.get('audience') or 'Unknown').capitalize()}\n", "content")
        self.text.insert(tk.END, "PHYSICAL SETTING\n", "h2")
        self.text.insert(tk.END, f"{(pragmatic.get('physical_setting') or 'Not specified').capitalize()}\n", "content")

        self.text.configure(state="disabled")

        self.root.title(f"Synneon — {image_path.name}")
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after_idle(self.root.attributes, "-topmost", False)


def _stdin_reader(viewer: CaptureViewer):
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            viewer.update(Path(msg["image_path"]), msg["result"])
        except Exception as e:
            print(f"[viewer] parse error: {e}", file=sys.stderr)


def main():
    root = tk.Tk()
    viewer = CaptureViewer(root)
    t = threading.Thread(target=_stdin_reader, args=(viewer,), daemon=True)
    t.start()
    root.mainloop()


if __name__ == "__main__":
    main()
