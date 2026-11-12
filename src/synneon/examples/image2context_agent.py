import asyncio
import tomllib
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from synneon.agents.base import VLMImage2ContextAgent
from synneon.schemas.agents import AgentSettings

app = typer.Typer(help="VLM Image to Context Agent Example")
console = Console()


def load_agent_settings(toml_path: Path, profile: str = "image2context") -> AgentSettings:
    """Load agent settings from a TOML file for a specific profile."""
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    # The agent.toml structure is [agent.profile_name]
    agent_data = data.get("agent", {}).get(profile)
    if not agent_data:
        raise ValueError(f"Profile '{profile}' not found in {toml_path}")

    return AgentSettings(**agent_data)


@app.command()
def describe(
    image_path: Path = typer.Argument(..., help="Path to the image file to describe"),
    agent_toml: Path = typer.Option(
        "configs/interpreter/agent.toml", "--agent-toml", "-a", help="Path to the agent configuration TOML"
    ),
    profile: str = typer.Option("image2context", help="Agent profile to use from TOML"),
):
    """
    Use VLMImage2ContextAgent to get a description of an image.
    """
    # 1. Load Configurations
    try:
        settings = load_agent_settings(agent_toml, profile)
    except Exception as e:
        console.print(f"[bold red]Error loading settings:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    if not image_path.exists():
        console.print(f"[bold red]Error:[/bold red] Image file {image_path} does not exist.")
        raise typer.Exit(code=1)

    setup_info = [
        f"[bold blue]Agent Profile:[/bold blue] {profile}",
        f"[bold blue]Model:[/bold blue] {settings.model}",
        f"[bold blue]Image:[/bold blue] {image_path}",
    ]
    console.print(Panel.fit("\n".join(setup_info), title="VLM Setup"))

    # 2. Initialize the Agent
    # For description tasks, we typically use str as output type
    agent = VLMImage2ContextAgent(settings=settings, output_type=str)

    # 3. Execute Inference
    async def execute():
        console.print("\n[yellow]Reading image and running VLM inference...[/yellow]")

        # Read image bytes
        image_bytes = image_path.read_bytes()

        # Determine media type based on extension
        ext = image_path.suffix.lower()
        media_type = "image/jpeg"
        if ext == ".png":
            media_type = "image/png"
        elif ext == ".webp":
            media_type = "image/webp"

        try:
            # We run the agent, the prompt_template from settings will be used
            result = await agent.run(image_bytes, media_type=media_type)

            console.print("\n[bold green]Model Description:[/bold green]")
            console.print(Panel(str(result.output), border_style="green", title="VLM Response"))
        except Exception as e:
            console.print(f"[bold red]Inference failed:[/bold red] {e}")

    asyncio.run(execute())


if __name__ == "__main__":
    load_dotenv()
    app()
