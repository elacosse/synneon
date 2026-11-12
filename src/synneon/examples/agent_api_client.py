import base64
from pathlib import Path

import httpx
import typer
import yaml
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from synneon.schemas.context import SteeringContext
from synneon.schemas.requests import (
    CognitiveSteeringInferenceRequest,
    Image2ContextRequest,
)

app = typer.Typer(help="Synneon Cognitive Steering REST Client")
console = Console()


def load_example_data(yaml_path: Path) -> tuple[SteeringContext, str, str | None]:
    """Load context, tokens, and ground truth from a YAML example file."""
    if not yaml_path.exists():
        # Fallback to absolute check if relative fails
        alt_path = Path(__file__).parent.parent.parent.parent / yaml_path
        if not alt_path.exists():
            raise FileNotFoundError(f"Example file not found at {yaml_path}")
        path = alt_path
    else:
        path = yaml_path

    with open(path) as f:
        data = yaml.safe_load(f)

    # Extract tokens and ground truth separately
    tokens = data.pop("tokens", "")
    ground_truth = data.pop("ground_truth", None)
    context = SteeringContext(**data)

    return context, tokens, ground_truth


def display_results_window(image_path: Path, result: dict):
    """Display image and analysis results in a Tkinter window."""
    try:
        import tkinter as tk
        from tkinter import ttk

        from PIL import Image
    except ImportError:
        console.print("[bold red]Error:[/bold red] PIL (Pillow) and tkinter are required for GUI display.")
        console.print("Install with: [bold]pip install pillow[/bold] (and ensure tkinter is installed)")
        return

    root = tk.Tk()
    root.title("Synneon Image Context Analysis")
    root.geometry("1200x800")

    # Use a paned window for adjustable split
    paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Left frame for Image
    frame_left = ttk.Frame(paned)
    paned.add(frame_left, weight=2)

    # Right frame for Text
    frame_right = ttk.Frame(paned)
    paned.add(frame_right, weight=1)

    # Load and display image
    try:
        img = Image.open(image_path)

        # Resize logic
        max_width = 800
        max_height = 800
        width_ratio = max_width / img.width
        height_ratio = max_height / img.height
        scale = min(width_ratio, height_ratio, 1.0)

        if scale < 1.0:
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)

            # Safe resample constant
            if hasattr(Image, "Resampling"):
                resample = Image.Resampling.LANCZOS
            else:
                resample = getattr(Image, "LANCZOS", 1)

            img = img.resize((new_w, new_h), resample)

        # FIX: Bypass ImageTk.PhotoImage and use native tkinter.PhotoImage via PNG bytes.
        # This avoids "bad argument type" errors caused by broken PIL/Tkinter C-extension links.
        import io

        byte_io = io.BytesIO()
        img.save(byte_io, format="PNG")
        byte_data = byte_io.getvalue()

        photo = tk.PhotoImage(data=byte_data)

        label_img = ttk.Label(frame_left, image=photo)
        label_img.image = photo  # Keep reference!
        label_img.pack(expand=True, padx=5, pady=5)

    except Exception as e:
        console.print(f"[bold red]Error displaying image:[/bold red] {e}")
        # Show error text in the GUI as well
        ttk.Label(frame_left, text=f"Error loading image:\n{e}").pack(padx=10, pady=10)

    # Formatted Text Output
    text_widget = tk.Text(frame_right, wrap="word", padx=10, pady=10, font=("Helvetica", 12))
    text_widget.pack(fill=tk.BOTH, expand=True)

    # Define styles
    text_widget.tag_configure("h1", font=("Helvetica", 18, "bold"), foreground="#2C3E50", spacing3=20, spacing1=10)
    text_widget.tag_configure("h2", font=("Helvetica", 11, "bold"), foreground="#7F8C8D", spacing3=2)
    text_widget.tag_configure("content", font=("Helvetica", 13), spacing1=2, spacing3=12, lmargin1=10)

    # Insert content
    visual = result.get("parsed_visual", {})
    pragmatic = result.get("pragmatic", {})

    text_widget.insert(tk.END, "Visual Analysis\n", "h1")

    text_widget.insert(tk.END, "PRIMARY FOCUS\n", "h2")
    text_widget.insert(tk.END, f"{(visual.get('focused_object') or 'None').capitalize()}\n", "content")

    text_widget.insert(tk.END, "PERIPHERAL OBJECTS\n", "h2")
    det_objs = visual.get("detected_objects", [])
    text_widget.insert(tk.END, f"{', '.join(det_objs).capitalize() if det_objs else 'None'}\n", "content")

    text_widget.insert(tk.END, "\nPragmatic Inference\n", "h1")

    text_widget.insert(tk.END, "COMMUNICATIVE INTENT\n", "h2")
    text_widget.insert(tk.END, f"{(pragmatic.get('intent') or 'Unknown').capitalize()}\n", "content")

    text_widget.insert(tk.END, "SOCIAL AUDIENCE\n", "h2")
    text_widget.insert(tk.END, f"{(pragmatic.get('audience') or 'Unknown').capitalize()}\n", "content")

    text_widget.insert(tk.END, "PHYSICAL SETTING\n", "h2")
    text_widget.insert(tk.END, f"{(pragmatic.get('physical_setting') or 'Not specified').capitalize()}\n", "content")

    text_widget.configure(state="disabled")

    # Bring window to front
    root.lift()
    root.attributes("-topmost", True)
    root.after_idle(root.attributes, "-topmost", False)

    root.mainloop()


@app.command()
def expand(
    url: str = typer.Option(
        "http://localhost:8000/inference/expand", "--url", "-u", help="URL of the expansion service"
    ),
    example_yaml: Path = typer.Option(
        "data/examples/example_1.yaml", "--example-yaml", "-e", help="Path to the example data YAML"
    ),
    profile: str = typer.Option("default", help="Agent profile to use from TOML configuration"),
):
    """
    Perform abbreviation expansion by calling the Synneon REST API.
    """
    # 1. Load Data
    try:
        context, tokens, ground_truth = load_example_data(example_yaml)
    except Exception as e:
        console.print(f"[bold red]Error loading data:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    # 2. Prepare Request
    request_data = CognitiveSteeringInferenceRequest(context=context, tokens=tokens, profile=profile)

    req_table = Table(show_header=False, box=None, padding=(0, 1))
    req_table.add_row("[bold blue]Endpoint:[/bold blue]", url)
    req_table.add_row("[bold blue]Tokens:[/bold blue]", f'[italic green]"{tokens}"[/italic green]')
    req_table.add_row("[bold blue]Profile:[/bold blue]", profile)

    ctx_table = Table(show_header=False, box=None, padding=(0, 1))
    ctx_table.add_row("[bold cyan]Emotion:[/bold cyan]", str(context.emotion))
    ctx_table.add_row("[bold cyan]Visual:[/bold cyan]", str(context.parsed_visual))
    ctx_table.add_row("[bold cyan]Pragmatic:[/bold cyan]", str(context.pragmatic))

    console.print(
        Panel.fit(
            Group(req_table, Panel(ctx_table, title="[dim]Steering Context[/dim]", border_style="dim", padding=(0, 1))),
            title="REST Expansion Request",
            border_style="blue",
        )
    )

    # 3. Execute Request
    try:
        with httpx.Client(timeout=60.0) as client:
            console.print(f"\n[yellow]Sending POST request to {url}...[/yellow]")
            response = client.post(
                url, content=request_data.model_dump_json(), headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPStatusError as e:
        console.print(f"[bold red]Server error ({e.response.status_code}):[/bold red] {e.response.text}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[bold red]Connection error:[/bold red] {e}")
        console.print("[dim]Make sure the server is running with 'uv run python -m synneon.server'[/dim]")
        raise typer.Exit(code=1) from e

    # 4. Display Results
    console.print("\n[bold green]Expansion Candidates:[/bold green]")
    candidates = result.get("candidates", [])

    if not candidates:
        console.print("[yellow]No candidates returned from service.[/yellow]")
    else:
        res_table = Table(show_header=True, header_style="bold green", box=None, padding=(0, 2))
        res_table.add_column("#", justify="right", style="dim")
        res_table.add_column("Expansion Text", style="bold white")

        for c in candidates:
            res_table.add_row(str(c["rank"]), c["text"])

        console.print(Panel.fit(res_table, border_style="green", title="Expansion Results"))

        if ground_truth:
            match = any(ground_truth.lower() in c["text"].lower() for c in candidates)
            if match:
                console.print(
                    f"[bold green]✓ Match found![/bold green] Ground truth '{ground_truth}' is in the result set."
                )
            else:
                console.print(
                    f"[bold red]✗ No match.[/bold red] Ground truth '{ground_truth}' was not found in the results."
                )


@app.command()
def describe(
    image_path: Path = typer.Argument(..., help="Path to the image file to describe"),
    url: str = typer.Option(
        None,
        "--url",
        "-u",
        help="URL of the image2context service (overrides default based on --base64)",
    ),
    profile: str = typer.Option("image2context", help="Agent profile to use from TOML configuration"),
    use_base64: bool = typer.Option(False, "--base64", help="Use base64 encoding instead of multipart/form-data"),
    display: bool = typer.Option(False, "--display", "-d", help="Display the results in a GUI window"),
):
    """
    Perform image-to-context inference by calling the Synneon REST API.
    """
    if not image_path.exists():
        console.print(f"[bold red]Error:[/bold red] Image file {image_path} does not exist.")
        raise typer.Exit(code=1)

    # Determine default endpoint if not provided
    if url is None:
        if use_base64:
            url = "http://localhost:8000/inference/image2context/base64"
        else:
            url = "http://localhost:8000/inference/image2context"

    req_table = Table(show_header=False, box=None, padding=(0, 1))
    req_table.add_row("[bold blue]Endpoint:[/bold blue]", url)
    req_table.add_row("[bold blue]Image:[/bold blue]", str(image_path))
    req_table.add_row("[bold blue]Profile:[/bold blue]", profile)
    req_table.add_row("[bold blue]Mode:[/bold blue]", "Base64" if use_base64 else "Multipart")

    console.print(
        Panel.fit(
            req_table,
            title="REST Image2Context Request",
            border_style="blue",
        )
    )

    # 1. Execute Request
    try:
        # Determine media type based on extension
        ext = image_path.suffix.lower()
        media_type = "image/jpeg"
        if ext == ".png":
            media_type = "image/png"
        elif ext == ".webp":
            media_type = "image/webp"

        with httpx.Client(timeout=60.0) as client:
            if use_base64:
                console.print(f"\n[yellow]Sending base64 encoded image to {url}...[/yellow]")
                image_bytes = image_path.read_bytes()
                encoded = base64.b64encode(image_bytes).decode("utf-8")

                request_data = Image2ContextRequest(image_base64=encoded, media_type=media_type, profile=profile)
                response = client.post(
                    url,
                    content=request_data.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )
            else:
                console.print(f"\n[yellow]Uploading image to {url}...[/yellow]")
                with open(image_path, "rb") as f:
                    files = {"file": (image_path.name, f, media_type)}
                    params = {"profile": profile}
                    response = client.post(url, files=files, params=params)

            response.raise_for_status()
            result = response.json()
    except httpx.HTTPStatusError as e:
        console.print(f"[bold red]Server error ({e.response.status_code}):[/bold red] {e.response.text}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[bold red]Connection error:[/bold red] {e}")
        console.print("[dim]Make sure the server is running with 'uv run python -m synneon.server'[/dim]")
        raise typer.Exit(code=1) from e

    # 2. Display Results
    console.print("\n[bold green]Received Context Analysis:[/bold green]")

    visual = result.get("parsed_visual", {})
    pragmatic = result.get("pragmatic", {})

    vis_table = Table(show_header=False, box=None, padding=(0, 1))
    vis_table.add_row("[bold cyan]Focused Object:[/bold cyan]", (visual.get("focused_object") or "N/A").capitalize())
    det_objs = visual.get("detected_objects", [])
    vis_table.add_row(
        "[bold cyan]Detected Objects:[/bold cyan]", ", ".join(det_objs).capitalize() if det_objs else "None"
    )

    prag_table = Table(show_header=False, box=None, padding=(0, 1))
    prag_table.add_row("[bold magenta]Intent:[/bold magenta]", (pragmatic.get("intent") or "N/A").capitalize())
    prag_table.add_row("[bold magenta]Audience:[/bold magenta]", (pragmatic.get("audience") or "N/A").capitalize())
    prag_table.add_row(
        "[bold magenta]Setting:[/bold magenta]", (pragmatic.get("physical_setting") or "N/A").capitalize()
    )

    console.print(
        Panel.fit(
            Group(
                Panel(vis_table, border_style="cyan", title="[bold white]Visual[/bold white]", padding=(0, 1)),
                Panel(prag_table, border_style="magenta", title="[bold white]Pragmatic[/bold white]", padding=(0, 1)),
            ),
            title="Inferred Context",
            border_style="green",
        )
    )

    if display:
        console.print("\n[bold green]Opening display window...[/bold green]")
        display_results_window(image_path, result)


if __name__ == "__main__":
    app()
