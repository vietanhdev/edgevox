"""Exercise every robot-agent tool/skill against the real sim.

Sanity check for the tool-calling benchmark: before we ship scenarios
to an LLM, verify that every tool the LLM can pick actually returns a
sensible value on a freshly-booted environment, and that realistic
chains of actions (navigate → stop → navigate; grasp → move → release;
turn → walk → stand) complete without deadlock.

Usage::

    python scripts/smoke_test_robot_tools.py              # run every agent
    python scripts/smoke_test_robot_tools.py --only panda scout

Prints a markdown-style table plus a verdict per chain. Exits non-zero
if any tool/skill raised or any chain timed out, so it can be wired
into CI later.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgevox.agents.base import AgentContext
from edgevox.agents.skills import GoalStatus, Skill
from edgevox.llm.tools import Tool as _ToolDescriptor


def _normalize_tools(items: Any) -> list[_ToolDescriptor]:
    """Accept a list of ``@tool`` functions OR a :class:`ToolRegistry` and
    return the list of real tool descriptors — skipping fallbacks that a
    ``LLMAgent`` auto-registers for its skills."""
    out: list[_ToolDescriptor] = []
    if hasattr(items, "tools") and isinstance(items.tools, dict):
        # ToolRegistry. Filter out skill fallbacks (they share the name
        # with a skill and live in the same registry for schema purposes).
        for desc in items.tools.values():
            if not desc.func.__name__.endswith("_fallback"):
                out.append(desc)
        return out
    # Plain list — elements are raw @tool functions carrying the descriptor.
    for f in items:
        desc = getattr(f, "__edgevox_tool__", None)
        if desc is None:
            raise TypeError(f"{f!r} is not a @tool-decorated function")
        out.append(desc)
    return out


def _normalize_skills(items: Any) -> list[Skill]:
    if isinstance(items, dict):
        return list(items.values())
    return list(items)


@dataclass
class CallResult:
    name: str
    kind: str  # "tool" | "skill"
    ok: bool
    elapsed_s: float
    value: Any = None
    error: str | None = None

    def short_value(self) -> str:
        if self.error:
            return f"ERR: {self.error}"
        s = repr(self.value)
        return s if len(s) < 80 else s[:77] + "..."


@dataclass
class ChainStep:
    label: str
    ok: bool
    elapsed_s: float
    detail: str = ""


@dataclass
class AgentReport:
    name: str
    calls: list[CallResult] = field(default_factory=list)
    chains: list[tuple[str, list[ChainStep]]] = field(default_factory=list)
    skipped_reason: str | None = None


def _call_tool(tool: Any, ctx: AgentContext, kwargs: dict[str, Any]) -> CallResult:
    """Invoke a ``@tool`` by calling the wrapped function directly.

    ``Tool`` descriptors carry the real callable on ``.func``.
    """
    t0 = time.perf_counter()
    try:
        value = tool.func(ctx=ctx, **kwargs)
        return CallResult(tool.name, "tool", True, time.perf_counter() - t0, value=value)
    except Exception as e:
        return CallResult(tool.name, "tool", False, time.perf_counter() - t0, error=f"{type(e).__name__}: {e}")


def _call_skill(skill: Skill, ctx: AgentContext, kwargs: dict[str, Any], poll_timeout: float = 30.0) -> CallResult:
    t0 = time.perf_counter()
    handle = skill.start(ctx, **kwargs)
    status = handle.poll(timeout=poll_timeout)
    elapsed = time.perf_counter() - t0
    if status is GoalStatus.SUCCEEDED:
        return CallResult(skill.name, "skill", True, elapsed, value=handle.result)
    return CallResult(skill.name, "skill", False, elapsed, error=f"{status.name}: {handle.error}")


def _run_scout(report: AgentReport) -> None:
    """Scout uses ToyWorld — pure stdlib, no downloads."""
    from edgevox.agents import ToyWorld
    from edgevox.examples.agents.robot_scout import APP

    env = ToyWorld()
    ctx = AgentContext(deps=env)

    tools_list = _normalize_tools(APP.agent.tools)
    skills_list = _normalize_skills(APP.agent.skills)

    # --- tools ---
    for tool in tools_list:
        kwargs: dict[str, Any] = {}
        if tool.name == "light_state":
            kwargs["room"] = "kitchen"
        report.calls.append(_call_tool(tool, ctx, kwargs))

    # --- skills ---
    for skill in skills_list:
        kwargs = {}
        if skill.name == "navigate_to_room" or skill.name in ("turn_on_light", "turn_off_light"):
            kwargs["room"] = "kitchen"
        report.calls.append(_call_skill(skill, ctx, kwargs, poll_timeout=15.0))

    # --- chains ---
    skills = {s.name: s for s in skills_list}
    tools = {t.name: t for t in tools_list}

    # Chain 1: turn on kitchen light, navigate to kitchen, check light_state.
    steps: list[ChainStep] = []
    for label, fn in [
        ("turn_on_light(kitchen)", lambda: _call_skill(skills["turn_on_light"], ctx, {"room": "kitchen"}, 5.0)),
        ("navigate_to_room(kitchen)", lambda: _call_skill(skills["navigate_to_room"], ctx, {"room": "kitchen"}, 15.0)),
        ("current_room", lambda: _call_tool(tools["current_room"], ctx, {})),
        ("light_state(kitchen)", lambda: _call_tool(tools["light_state"], ctx, {"room": "kitchen"})),
        ("turn_off_light(kitchen)", lambda: _call_skill(skills["turn_off_light"], ctx, {"room": "kitchen"}, 5.0)),
    ]:
        r = fn()
        steps.append(ChainStep(label, r.ok, r.elapsed_s, r.short_value()))
    report.chains.append(("light-on → nav → check → light-off", steps))

    # Chain 2: navigate to bedroom, return_home.
    steps = []
    for label, fn in [
        ("navigate_to_room(bedroom)", lambda: _call_skill(skills["navigate_to_room"], ctx, {"room": "bedroom"}, 15.0)),
        ("return_home", lambda: _call_skill(skills["return_home"], ctx, {}, 15.0)),
        ("current_room", lambda: _call_tool(tools["current_room"], ctx, {})),
    ]:
        r = fn()
        steps.append(ChainStep(label, r.ok, r.elapsed_s, r.short_value()))
    report.chains.append(("nav(bedroom) → return_home", steps))


def _run_irsim(report: AgentReport, render: bool = False) -> None:
    from edgevox.examples.agents.robot_irsim import APP

    try:
        from edgevox.integrations.sim.irsim import IrSimEnvironment
    except Exception as e:  # pragma: no cover
        report.skipped_reason = f"irsim adapter unavailable: {e}"
        return

    try:
        env = IrSimEnvironment(render=render)
    except Exception as e:
        report.skipped_reason = f"IrSimEnvironment construction failed: {e}"
        return

    try:
        ctx = AgentContext(deps=env)

        tools_list = _normalize_tools(APP.tools)
        skills_list = _normalize_skills(APP.skills)

        for tool in tools_list:
            kwargs: dict[str, Any] = {}
            if tool.name == "describe_room":
                kwargs["room"] = "kitchen"
            report.calls.append(_call_tool(tool, ctx, kwargs))

        for skill in skills_list:
            kwargs = {}
            if skill.name == "navigate_to_room":
                kwargs["room"] = "kitchen"
            elif skill.name == "navigate_to_point":
                kwargs["x"] = 3.0
                kwargs["y"] = 3.0
            timeout = 5.0 if skill.name == "stop" else 20.0
            report.calls.append(_call_skill(skill, ctx, kwargs, poll_timeout=timeout))

        skills = {s.name: s for s in skills_list}
        tools = {t.name: t for t in tools_list}

        # Chain: navigate → stop (barge-in mid-flight) → navigate to a new room.
        steps: list[ChainStep] = []
        t0 = time.perf_counter()
        handle = skills["navigate_to_room"].start(ctx, room="bedroom")
        time.sleep(0.5)
        r_stop = _call_skill(skills["stop"], ctx, {}, 3.0)
        handle.cancel()
        nav_status = handle.poll(timeout=3.0)
        nav_elapsed = time.perf_counter() - t0
        steps.append(
            ChainStep(
                "navigate_to_room(bedroom) → stop",
                r_stop.ok and nav_status is not GoalStatus.SUCCEEDED,
                nav_elapsed,
                f"nav={nav_status.name}",
            )
        )

        r = _call_skill(skills["navigate_to_room"], ctx, {"room": "kitchen"}, 30.0)
        steps.append(ChainStep("navigate_to_room(kitchen)", r.ok, r.elapsed_s, r.short_value()))

        r = _call_tool(tools["current_room"], ctx, {})
        steps.append(ChainStep("current_room", r.ok, r.elapsed_s, r.short_value()))

        r = _call_skill(skills["return_home"], ctx, {}, 30.0)
        steps.append(ChainStep("return_home", r.ok, r.elapsed_s, r.short_value()))

        report.chains.append(("navigate → stop → navigate → return_home", steps))
    finally:
        with contextlib.suppress(Exception):
            env.close()


def _run_panda(report: AgentReport, render: bool = False) -> None:
    from edgevox.examples.agents.robot_panda import APP

    try:
        from edgevox.integrations.sim.mujoco_arm import MujocoArmEnvironment
    except Exception as e:
        report.skipped_reason = f"mujoco_arm adapter unavailable: {e}"
        return

    try:
        env = MujocoArmEnvironment(model_source="franka", render=render)
    except Exception as e:
        report.skipped_reason = f"MujocoArmEnvironment construction failed: {e}"
        return

    try:
        ctx = AgentContext(deps=env)

        tools_list = _normalize_tools(APP.tools)
        skills_list = _normalize_skills(APP.skills)

        for tool in tools_list:
            kwargs: dict[str, Any] = {}
            if tool.name == "locate_object":
                kwargs["name"] = "red_cube"
            report.calls.append(_call_tool(tool, ctx, kwargs))

        for skill in skills_list:
            kwargs = {}
            if skill.name == "move_to_point":
                kwargs.update(x=0.35, y=0.0, z=0.3)
            elif skill.name == "move_above_object" or skill.name == "grasp":
                kwargs["object"] = "red_cube"
            report.calls.append(_call_skill(skill, ctx, kwargs, poll_timeout=30.0))

        skills = {s.name: s for s in skills_list}
        tools = {t.name: t for t in tools_list}

        # Chain: locate_object → move_above_object → grasp → get_gripper_state → release → goto_home
        steps: list[ChainStep] = []
        for label, fn in [
            ("locate_object(green_cube)", lambda: _call_tool(tools["locate_object"], ctx, {"name": "green_cube"})),
            (
                "move_above_object(green_cube)",
                lambda: _call_skill(skills["move_above_object"], ctx, {"object": "green_cube"}, 30.0),
            ),
            ("grasp(green_cube)", lambda: _call_skill(skills["grasp"], ctx, {"object": "green_cube"}, 30.0)),
            ("get_gripper_state", lambda: _call_tool(tools["get_gripper_state"], ctx, {})),
            (
                "move_to_point(0.4,0.1,0.4)",
                lambda: _call_skill(skills["move_to_point"], ctx, {"x": 0.4, "y": 0.1, "z": 0.4}, 30.0),
            ),
            ("release", lambda: _call_skill(skills["release"], ctx, {}, 20.0)),
            ("goto_home", lambda: _call_skill(skills["goto_home"], ctx, {}, 30.0)),
        ]:
            r = fn()
            steps.append(ChainStep(label, r.ok, r.elapsed_s, r.short_value()))
        report.chains.append(("locate → hover → grasp → carry → release → home", steps))
    finally:
        with contextlib.suppress(Exception):
            env.close()


def _run_humanoid(report: AgentReport, render: bool = False) -> None:
    from edgevox.examples.agents.robot_humanoid import APP

    try:
        from edgevox.integrations.sim.mujoco_humanoid import MujocoHumanoidEnvironment
    except Exception as e:
        report.skipped_reason = f"mujoco_humanoid adapter unavailable: {e}"
        return

    try:
        env = MujocoHumanoidEnvironment(model_source="unitree_g1", render=render)
    except Exception as e:
        report.skipped_reason = f"MujocoHumanoidEnvironment construction failed: {e}"
        return

    try:
        ctx = AgentContext(deps=env)

        tools_list = _normalize_tools(APP.tools)
        skills_list = _normalize_skills(APP.skills)

        for tool in tools_list:
            report.calls.append(_call_tool(tool, ctx, {}))

        for skill in skills_list:
            kwargs: dict[str, Any] = {}
            if skill.name in ("walk_forward", "walk_backward"):
                kwargs["distance"] = 0.3
            elif skill.name in ("turn_left", "turn_right"):
                kwargs["degrees"] = 30.0
            report.calls.append(_call_skill(skill, ctx, kwargs, poll_timeout=25.0))

        skills = {s.name: s for s in skills_list}
        tools = {t.name: t for t in tools_list}

        steps: list[ChainStep] = []
        for label, fn in [
            ("walk_forward(0.5)", lambda: _call_skill(skills["walk_forward"], ctx, {"distance": 0.5}, 25.0)),
            ("turn_left(45)", lambda: _call_skill(skills["turn_left"], ctx, {"degrees": 45.0}, 15.0)),
            ("walk_forward(0.3)", lambda: _call_skill(skills["walk_forward"], ctx, {"distance": 0.3}, 25.0)),
            ("stand", lambda: _call_skill(skills["stand"], ctx, {}, 10.0)),
            ("is_standing", lambda: _call_tool(tools["is_standing"], ctx, {})),
        ]:
            r = fn()
            steps.append(ChainStep(label, r.ok, r.elapsed_s, r.short_value()))
        report.chains.append(("walk → turn → walk → stand → check", steps))
    finally:
        with contextlib.suppress(Exception):
            env.close()


def _print_report(report: AgentReport) -> bool:
    print(f"\n## {report.name}")
    print()
    if report.skipped_reason:
        print(f"**SKIPPED:** {report.skipped_reason}")
        return True

    print("### Individual tool / skill calls")
    print()
    print("| kind | name | ok | elapsed | value / error |")
    print("|---|---|---|---|---|")
    all_ok = True
    for c in report.calls:
        mark = "✅" if c.ok else "❌"
        if not c.ok:
            all_ok = False
        print(f"| {c.kind} | `{c.name}` | {mark} | {c.elapsed_s:.2f}s | {c.short_value()} |")

    for chain_name, steps in report.chains:
        print()
        print(f"### Chain: {chain_name}")
        print()
        print("| step | ok | elapsed | detail |")
        print("|---|---|---|---|")
        for s in steps:
            mark = "✅" if s.ok else "❌"
            if not s.ok:
                all_ok = False
            print(f"| {s.label} | {mark} | {s.elapsed_s:.2f}s | {s.detail} |")

    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None, choices=["scout", "irsim", "panda", "humanoid"])
    parser.add_argument("--render", action="store_true", help="Open sim windows (default: headless).")
    args = parser.parse_args()

    targets = args.only or ["scout", "irsim", "panda", "humanoid"]
    all_ok = True

    print("# Robot tool-surface smoke test")
    for name in targets:
        report = AgentReport(name=name)
        try:
            if name == "scout":
                _run_scout(report)
            elif name == "irsim":
                _run_irsim(report, render=args.render)
            elif name == "panda":
                _run_panda(report, render=args.render)
            elif name == "humanoid":
                _run_humanoid(report, render=args.render)
        except Exception as e:
            report.skipped_reason = f"unexpected crash: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        ok = _print_report(report)
        all_ok = all_ok and ok

    print()
    print("## Summary")
    print()
    print("All checks passed." if all_ok else "Some checks failed — see tables above.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
