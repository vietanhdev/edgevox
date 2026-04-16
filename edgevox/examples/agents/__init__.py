"""Built-in agent examples.

Each submodule defines a list of ``@tool``-decorated functions plus a
``main()`` that drops the user into a text REPL. The ``cli`` module
dispatches between them for the ``edgevox-agent`` entry point.
"""

from edgevox.examples.agents.dev_toolbox import DEV_TOOLS
from edgevox.examples.agents.home_assistant import HOME_TOOLS
from edgevox.examples.agents.robot_commander import ROBOT_TOOLS

__all__ = ["DEV_TOOLS", "HOME_TOOLS", "ROBOT_TOOLS"]
