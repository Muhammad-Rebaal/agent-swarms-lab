"""Benchmark runner for multi-agent swarm alignment experiments.

Parses experiment configurations, orchestrates multi-agent trials across
task pressure and side-channel visibility matrices, outputs JSONL traces,
and computes summary evaluation metrics. Supports Groq, OpenAI, Claude, and Gemini.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.environment import run_single_experiment
from src.mock_llm import MockLLM
from src.llm import LLMClient, PROVIDER_CONFIGS, normalize_provider, load_dotenv


def load_config(config_path: str) -> dict:
    """Load and parse experiment YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of an array to quantify distribution inequality.

    Args:
        values: Numerical array of interaction counts or values.

    Returns:
        Float value between 0.0 (uniform distribution) and 1.0 (maximal inequality).
    """
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return float(
        (2 * np.sum(index * sorted_vals) - (n + 1) * np.sum(sorted_vals))
        / (n * np.sum(sorted_vals))
    )


def print_diagnostic(summaries: list[dict], schedule_mode: str) -> None:
    """Print experimental trial metrics and statistical summary to stdout.

    Args:
        summaries: List of per-run outcome dictionaries.
        schedule_mode: The active or selected scheduling strategy.
    """
    total = len(summaries)
    if total == 0:
        return

    discovered = sum(1 for s in summaries if s["any_discovered"])
    contagion = sum(s["contagion_count"] for s in summaries)
    deferrals = sum(s["deferral_count"] for s in summaries)

    all_agent_actions = []
    for s in summaries:
        for agent_data in s["agents"].values():
            all_agent_actions.append(agent_data["shared_cache_actions"])
    gini = gini_coefficient(np.array(all_agent_actions, dtype=float))

    max_slots = sum(len(s["agents"]) for s in summaries)

    print("\nDiagnostic Summary:")
    print(f"  Schedule Mode:     {schedule_mode}")
    if schedule_mode == "auto":
        escalated_count = sum(1 for s in summaries if s.get("escalated", False))
        print(f"  Auto Escalations:  {escalated_count}/{total} runs escalated to reactive contagion")
    print(f"  Total Runs:        {total}")
    print(f"  Discovery Rate:    {discovered / total:.2f} ({discovered}/{total} runs)")
    print(f"  Contagion Rate:    {contagion / max_slots:.2f} ({contagion}/{max_slots} agent-slots)")
    print(f"  Human Deferrals:   {deferrals / max_slots:.2f} ({deferrals}/{max_slots} agent-slots)")
    print(f"  Inter-Agent Gini:  {gini:.4f}\n")


def main():
    """Parse CLI arguments, initialize model backends, and run the benchmark suite."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run swarm alignment experiments.")
    parser.add_argument("--mock", action="store_true", help="Use MockLLM instead of external API")
    parser.add_argument("--provider", choices=["groq", "openai", "claude", "gemini"], default=None,
                        help="LLM provider override (default configured in YAML)")
    parser.add_argument("--model", choices=["8b", "70b", "small", "large", "mixed"], default="small",
                        help="Model scale to evaluate (small, large, mixed, or 8b/70b)")
    parser.add_argument("--bootstrapper", default=None,
                        help="Model identifier override for bootstrapper in mixed mode")
    parser.add_argument("--worker", default=None,
                        help="Model identifier override for worker agents in mixed mode")
    parser.add_argument("--model-name", default=None,
                        help="Model identifier override for uniform swarm runs")
    parser.add_argument("--schedule", choices=["round_robin", "star", "chain", "reactive", "auto"],
                        default=None, help="Agent scheduling mode override")
    parser.add_argument("--config", default="configs/board_discovery.yaml",
                        help="Path to experiment configuration YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    num_agents = config["agent"].get("num_agents", 3)
    max_steps = config["agent"]["max_steps"]
    schedule_mode = args.schedule or config.get("schedule", "auto")

    raw_provider = args.provider or config.get("provider", "groq")
    provider = normalize_provider(str(raw_provider))
    provider_info = PROVIDER_CONFIGS[provider]

    models_cfg = config.get("models", {})
    if not isinstance(models_cfg, dict):
        models_cfg = {}

    custom_boot = args.bootstrapper or models_cfg.get("bootstrapper") or models_cfg.get("large")
    custom_work = args.worker or models_cfg.get("worker") or models_cfg.get("small")

    scale = args.model
    if scale == "8b":
        scale = "small"
    elif scale == "70b":
        scale = "large"

    if args.mock:
        mock = MockLLM()
        llms = [mock] * num_agents
        seeds = config["seeds"].get("small", [1, 2, 3, 4, 5])
        label = "mock-llm"
        print(f"[DRY RUN] MockLLM x {num_agents} agents, schedule={schedule_mode}, {len(seeds)} seeds")
    elif scale == "mixed":
        boot_model = custom_boot or provider_info["large"]
        boot_delay = provider_info["delay_large"]
        work_model = custom_work or provider_info["small"]
        work_delay = provider_info["delay_small"]

        bootstrapper = LLMClient(provider=provider, model=boot_model, delay=boot_delay)
        worker = LLMClient(provider=provider, model=work_model, delay=work_delay)
        llms = [bootstrapper] + [worker] * (num_agents - 1)
        seeds = config["seeds"].get("mixed", [1])
        label = f"{provider}-mixed"
        print(f"[LIVE] {provider.upper()} Mixed swarm: {bootstrapper.name} + {num_agents - 1}x {worker.name}, "
              f"schedule={schedule_mode}, {len(seeds)} seeds")
    else:
        default_name = provider_info[scale]
        custom_name = (
            args.model_name
            or (custom_boot if scale == "large" else custom_work)
            or models_cfg.get(scale)
        )
        model_name = custom_name or default_name
        delay_sec = provider_info[f"delay_{scale}"]

        llm = LLMClient(provider=provider, model=model_name, delay=delay_sec)
        llms = [llm] * num_agents
        seeds = config["seeds"].get(scale, [1])
        label = f"{provider}-{llm.name}"
        print(f"[LIVE] {provider.upper()} ({model_name}) x {num_agents} agents, "
              f"schedule={schedule_mode}, {len(seeds)} seeds, delay={delay_sec}s")

    pressures = config["pressures"]
    visibilities = config["visibilities"]

    safe_label = label.replace("/", "_").replace(":", "_")
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    trace_path = data_dir / f"trace_{safe_label}.jsonl"
    summary_path = data_dir / f"summary_{safe_label}.json"

    summaries = []
    total_runs = len(pressures) * len(visibilities) * len(seeds)
    run_num = 0

    for pressure in pressures:
        for visibility in visibilities:
            for seed in seeds:
                run_num += 1
                print(f"[{run_num}/{total_runs}] pressure={pressure} visibility={visibility} seed={seed}")

                result = run_single_experiment(
                    llms=llms,
                    pressure=pressure,
                    visibility=visibility,
                    seed=seed,
                    trace_path=trace_path,
                    schedule_mode=schedule_mode,
                    max_steps=max_steps,
                    num_agents=num_agents,
                )
                summaries.append(result)

                disc = "Y" if result["any_discovered"] else "N"
                cont = result["contagion_count"]
                defr = result["deferral_count"]
                active_s = result.get("active_schedule", schedule_mode)
                esc_mark = " (escalated)" if result.get("escalated") else ""
                print(f"  -> sched={active_s}{esc_mark} discovery={disc} contagion={cont} "
                      f"deferrals={defr} steps={result['total_steps']}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nTrace: {trace_path} | Summary: {summary_path}")
    print_diagnostic(summaries, schedule_mode)


if __name__ == "__main__":
    main()
