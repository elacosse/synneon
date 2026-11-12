import asyncio
import tomllib
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.panel import Panel

from synneon.agents.base import CognitiveSteeringAgent
from synneon.schemas.agents import AgentSettings
from synneon.schemas.context import SteeringContext

app = typer.Typer(help="Cognitive Steering Inference Example")
console = Console()


def load_agent_settings(toml_path: Path, profile: str = "default") -> AgentSettings:
    """Load agent settings from a TOML file for a specific profile."""
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    # The agent.toml structure is [agent.profile_name]
    agent_data = data.get("agent", {}).get(profile)
    if not agent_data:
        raise ValueError(f"Profile '{profile}' not found in {toml_path}")

    return AgentSettings(**agent_data)


def load_example_data(yaml_path: Path) -> tuple[SteeringContext, str, str | None]:
    """Load context and tokens from a YAML example file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Extract tokens and ground truth separately
    tokens = data.pop("tokens", "")
    ground_truth = data.pop("ground_truth", None)
    context = SteeringContext(**data)

    return context, tokens, ground_truth


@app.command()
def run(
    agent_toml: Path = typer.Option(
        "configs/interpreter/agent.toml", "--agent-toml", "-a", help="Path to the agent configuration TOML"
    ),
    example_yaml: Path = typer.Option(
        "data/examples/example_1.yaml", "--example-yaml", "-e", help="Path to the example data YAML"
    ),
    profile: str = typer.Option("default", help="Agent profile to use from TOML"),
):
    """
    Perform inference using a CognitiveSteeringAgent with loaded context and settings.
    """
    # 1. Load Configurations
    try:
        settings = load_agent_settings(agent_toml, profile)
        context, tokens, ground_truth = load_example_data(example_yaml)
    except Exception as e:
        console.print(f"[bold red]Error loading data:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    setup_info = [
        f"[bold blue]Agent:[/bold blue] {settings.provider}:{settings.model}",
        f"[bold blue]Tokens:[/bold blue] {tokens}",
        f"[bold blue]Context:[/bold blue] {context}",
    ]
    if ground_truth:
        setup_info.append(f"[bold blue]Ground Truth:[/bold blue] {ground_truth}")

    console.print(Panel.fit("\n".join(setup_info), title="Inference Setup"))

    # 2. Initialize the Agent
    # We use 'str' as output type here for the abbreviation expansion baseline
    agent = CognitiveSteeringAgent(settings=settings, output_type=str)

    # 3. Execute Inference
    async def execute():
        # Build and display the prompts for transparency
        system_prompt, user_prompt = agent.builder.build_prompts(context, tokens=tokens)

        console.print(Panel(system_prompt, title="[bold magenta]Generated System Prompt[/bold magenta]"))
        console.print(Panel(user_prompt, title="[bold magenta]Generated User Prompt[/bold magenta]"))

        console.print("\n[yellow]Running inference...[/yellow]")
        # We pass tokens as a kwarg so the builder can inject it into {tokens} in the template
        result = await agent.run(context, tokens=tokens)

        console.print("\n[bold green]Expansion Candidates:[/bold green]")
        output_text = result.output
        console.print(Panel(str(output_text), border_style="green", title="Model Response"))

        if ground_truth:
            # Simple lexical matching (case-insensitive)
            if ground_truth.lower() in str(output_text).lower():
                console.print(
                    f"[bold green]✓ Match found![/bold green] Ground truth '{ground_truth}' is in the output."
                )
            else:
                console.print(
                    f"[bold red]✗ No match.[/bold red] Ground truth '{ground_truth}' was not found in the output."
                )

    asyncio.run(execute())


if __name__ == "__main__":
    app()
