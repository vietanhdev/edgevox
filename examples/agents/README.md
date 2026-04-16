# EdgeVox agent examples

Three runnable examples that show how to build voice agents with
`@tool`-decorated Python functions. All are self-contained: they load
the default Gemma GGUF the first time you run them and talk to the LLM
in the terminal.

## Usage

```bash
# pick any example
uv run python examples/agents/home_assistant.py
uv run python examples/agents/robot_commander.py
uv run python examples/agents/dev_toolbox.py
```

Each script drops you into a text REPL that goes through the same
`LLM.chat()` path the voice pipeline uses. Typing `quit` exits.

## Wiring an example into the full voice loop

Any of these tool sets can be handed to the normal voice pipeline by
constructing the LLM with `tools=[...]`:

```python
from edgevox.llm import LLM
from examples.agents.home_assistant import HOME_TOOLS

llm = LLM(tools=HOME_TOOLS, on_tool_call=lambda r: print("→", r))
```

## Third-party tool packages

If you want to publish a tool set as its own `pip install`-able package,
declare an entry point in your `pyproject.toml`:

```toml
[project.entry-points."edgevox.tools"]
home_assistant = "my_pkg.tools:HOME_TOOLS"
```

Then load with:

```python
from edgevox.llm import LLM, load_entry_point_tools

llm = LLM(tools=load_entry_point_tools())
```

The entry-point target can be a list of `@tool` functions, a single
function, or a `ToolRegistry`.
