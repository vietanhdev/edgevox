"""Tests for the ROS-shaped pub/sub foundation:
:class:`~edgevox.nodes.bus.Bus`, :class:`~edgevox.nodes.node.Node`
lifecycle, and the :class:`WorldStatePublisherNode` /
:class:`VisualizerNode` pair.
"""

from __future__ import annotations

import threading
import time

import pytest

from edgevox.nodes import Bus, Node, NodeState
from edgevox.nodes.frames import (
    UserUtterance,
    WorldStateFrame,
)
from edgevox.nodes.visualizer import VisualizerNode
from edgevox.nodes.world_state import WorldStatePublisherNode

# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------


class TestBus:
    def test_publish_subscribe_basic(self):
        bus = Bus()
        seen: list[str] = []
        bus.subscribe("test/topic", lambda f: seen.append(f.text))
        bus.publish("test/topic", UserUtterance(text="hello"))
        bus.publish("test/topic", UserUtterance(text="world"))
        assert seen == ["hello", "world"]

    def test_multiple_subscribers_each_get_messages(self):
        bus = Bus()
        a, b = [], []
        bus.subscribe("test/topic", a.append)
        bus.subscribe("test/topic", b.append)
        bus.publish("test/topic", "msg")
        assert a == ["msg"]
        assert b == ["msg"]

    def test_unsubscribe_removes_callback(self):
        bus = Bus()
        seen = []
        sub = bus.subscribe("test/topic", seen.append)
        bus.publish("test/topic", "first")
        sub.unsubscribe()
        bus.publish("test/topic", "second")
        assert seen == ["first"]

    def test_subscriber_exception_does_not_break_other_subscribers(self):
        """One broken subscriber must not halt the others. Without
        this guarantee, a flaky logger would silently mute everything
        downstream -- a particularly nasty class of production bug."""
        bus = Bus()

        def boom(f):
            raise RuntimeError("intentional")

        seen = []
        bus.subscribe("test/topic", boom)
        bus.subscribe("test/topic", seen.append)
        bus.publish("test/topic", "msg")
        assert seen == ["msg"]

    def test_latest_returns_most_recent(self):
        bus = Bus()
        bus.publish("test/topic", "first")
        bus.publish("test/topic", "second")
        assert bus.latest("test/topic") == "second"
        assert bus.latest("never_published") is None

    def test_deliver_latest_fires_on_subscribe(self):
        """ROS2 ``transient_local`` durability: a subscriber arriving
        late should immediately get the most-recent value so its
        viewer (or dashboard, or whatever) doesn't sit blank."""
        bus = Bus()
        bus.publish("test/topic", "old")
        seen = []
        bus.subscribe("test/topic", seen.append, deliver_latest=True)
        assert seen == ["old"]

    def test_publish_thread_safety(self):
        """Concurrent publishers and subscribers shouldn't corrupt
        the subscriber list or drop messages."""
        bus = Bus()
        seen: list[int] = []
        seen_lock = threading.Lock()

        def collect(f):
            with seen_lock:
                seen.append(f)

        bus.subscribe("test/topic", collect)

        def publisher(start, count):
            for i in range(start, start + count):
                bus.publish("test/topic", i)

        threads = [threading.Thread(target=publisher, args=(i * 100, 100)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(seen) == 400
        assert sorted(seen) == list(range(400))

    def test_stats_records_publish_counts(self):
        bus = Bus()
        bus.publish("a", 1)
        bus.publish("a", 2)
        bus.publish("b", 1)
        s = bus.stats()
        assert s["a"] == 2
        assert s["b"] == 1


# ---------------------------------------------------------------------------
# Node lifecycle
# ---------------------------------------------------------------------------


class _CountingNode(Node):
    """Test node that counts each loop iteration."""

    def __init__(self, name, bus, *, period_s: float = 0.01):
        super().__init__(name, bus)
        self._period = period_s
        self.iters = 0
        self.setup_called = False
        self.teardown_called = False

    def setup(self):
        self.setup_called = True
        self.subscribe("test/in", lambda f: None)

    def loop(self):
        while not self._stop.is_set():
            self.iters += 1
            self._stop.wait(self._period)

    def teardown(self):
        self.teardown_called = True


class TestNodeLifecycle:
    def test_states_progress_correctly(self):
        bus = Bus()
        n = _CountingNode("counter", bus)
        assert n.state is NodeState.UNCONFIGURED
        n.configure()
        assert n.state is NodeState.INACTIVE
        assert n.setup_called
        n.activate()
        assert n.state is NodeState.ACTIVE
        time.sleep(0.05)
        n.deactivate()
        assert n.state is NodeState.INACTIVE
        n.cleanup()
        assert n.state is NodeState.FINALISED
        assert n.teardown_called

    def test_loop_runs_until_stopped(self):
        bus = Bus()
        n = _CountingNode("counter", bus, period_s=0.005)
        n.configure()
        n.activate()
        time.sleep(0.05)
        n.deactivate()
        assert n.iters >= 1

    def test_subscriptions_dropped_at_cleanup(self):
        bus = Bus()
        n = _CountingNode("counter", bus)
        n.configure()
        n.activate()
        # Counter has subscribed in setup; bus should know.
        assert "test/in" in [t for t, subs in bus._subs.items() if subs]
        n.cleanup()
        # After cleanup, the node's sub list is empty.
        assert n._subs == []

    def test_double_configure_raises(self):
        bus = Bus()
        n = _CountingNode("counter", bus)
        n.configure()
        with pytest.raises(RuntimeError):
            n.configure()


# ---------------------------------------------------------------------------
# WorldStatePublisherNode + VisualizerNode integration
# ---------------------------------------------------------------------------


class _FakeEnv:
    """Minimal env exposing get_world_state() so the publisher can
    poll without bringing in MuJoCo / IR-SIM."""

    def __init__(self):
        self._tick = 0
        self._lock = threading.Lock()

    def get_world_state(self):
        with self._lock:
            self._tick += 1
            return {"tick": self._tick, "robot": {"x": 0.0, "y": 0.0}}


class TestWorldStatePublisher:
    def test_publishes_state_frames_at_rate(self):
        bus = Bus()
        env = _FakeEnv()
        node = WorldStatePublisherNode("phys", bus, env, sim_label="fake", rate_hz=200.0)
        node.configure()
        node.activate()
        time.sleep(0.1)
        node.cleanup()

        # 100 ms at 200 Hz should produce 10-30 frames depending on
        # scheduler -- relax to "more than 1".
        n = bus.stats().get("world/state", 0)
        assert n > 1, f"expected several frames, got {n}"
        latest = bus.latest("world/state")
        assert isinstance(latest, WorldStateFrame)
        assert latest.sim == "fake"
        assert latest.state["tick"] >= 1


class TestVisualizerNode:
    def test_pump_main_renders_latest_frame(self):
        bus = Bus()
        rendered: list[WorldStateFrame] = []
        viz = VisualizerNode("viz", bus, lambda frame: rendered.append(frame), backend="main_thread_pump")
        viz.configure()
        viz.activate()
        # No frame yet -- pump should be a no-op, not raise.
        viz.pump_main()
        assert rendered == []
        # Publish a frame; pump renders it.
        f = WorldStateFrame(sim="fake", state={"tick": 1})
        bus.publish("world/state", f)
        viz.pump_main()
        assert rendered == [f]
        # Pump again with no new publish -- still renders the latest
        # (viewer at higher rate than publisher is allowed).
        viz.pump_main()
        assert rendered == [f, f]
        viz.cleanup()

    def test_late_subscriber_gets_initial_frame(self):
        """Visualizer that opens after physics has been running
        should immediately see current state, not a blank window."""
        bus = Bus()
        bus.publish("world/state", WorldStateFrame(sim="fake", state={"tick": 42}))

        rendered = []
        viz = VisualizerNode("viz", bus, lambda f: rendered.append(f), backend="main_thread_pump")
        viz.configure()
        viz.activate()
        # No publish since subscribe, but deliver_latest fired -- the
        # frame is in the slot already.
        viz.pump_main()
        assert len(rendered) == 1
        assert rendered[0].state["tick"] == 42
        viz.cleanup()

    def test_render_exception_swallowed(self):
        """A broken render_fn must not crash the visualizer node."""
        bus = Bus()
        viz = VisualizerNode(
            "viz",
            bus,
            lambda f: (_ for _ in ()).throw(RuntimeError("intentional")),
            backend="main_thread_pump",
        )
        viz.configure()
        viz.activate()
        bus.publish("world/state", WorldStateFrame(sim="fake", state={"tick": 1}))
        # Should not raise.
        viz.pump_main()
        viz.cleanup()


# ---------------------------------------------------------------------------
# End-to-end: publisher + visualizer sharing a bus
# ---------------------------------------------------------------------------


class TestEndToEndNodeGraph:
    def test_publisher_to_visualizer_pipeline(self):
        bus = Bus()
        env = _FakeEnv()

        publisher = WorldStatePublisherNode("phys", bus, env, sim_label="fake", rate_hz=100.0)
        rendered = []
        viz = VisualizerNode("viz", bus, lambda f: rendered.append(f), backend="main_thread_pump")

        for n in (publisher, viz):
            n.configure()
            n.activate()

        # Pump the visualizer at ~30 Hz from the test thread for
        # 50 ms; inspect what the publisher delivered.
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            viz.pump_main()
            time.sleep(1 / 60)

        for n in (publisher, viz):
            n.cleanup()

        # Visualizer rendered something; publisher published more
        # than that (drops fine).
        assert len(rendered) > 0
        pub_count = bus.stats().get("world/state", 0)
        assert pub_count > 0
