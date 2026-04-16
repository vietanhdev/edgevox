"""Run the home-assistant agent example.

This file is a thin forwarder — the real implementation lives in
``edgevox.examples.agents.home_assistant`` so it can be shipped inside
the installed ``edgevox`` package and launched via ``edgevox-agent``.

Equivalent invocations::

    edgevox-agent home
    python -m edgevox.examples.agents.home_assistant
    python examples/agents/home_assistant.py
"""

from edgevox.examples.agents.home_assistant import main

if __name__ == "__main__":
    main()
