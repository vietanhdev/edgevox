"""ROS2 bridge integration test — runs inside the Docker container.

Usage:
    docker compose -f docker-compose.ros2.yml run --rm edgevox-ros2 \
        bash -c "source /opt/ros/jazzy/setup.bash && python3 tests/test_ros2_docker.py"
"""

import sys


def main():
    import rclpy
    from rclpy.qos import DurabilityPolicy, ReliabilityPolicy

    from edgevox.integrations.ros2_bridge import NullBridge, ROS2Bridge, create_bridge

    # Test 1: rclpy available
    print("[PASS] rclpy imported successfully")

    # Test 2: Create bridge (default namespace /edgevox)
    bridge = create_bridge(enabled=True)
    assert isinstance(bridge, ROS2Bridge), f"Expected ROS2Bridge, got {type(bridge)}"
    print("[PASS] ROS2Bridge created")

    # Test 3: All publishers present under default namespace
    pub_topics = [p.topic_name for p in bridge._node.publishers]
    expected_pubs = [
        "/edgevox/transcription",
        "/edgevox/response",
        "/edgevox/state",
        "/edgevox/audio_level",
        "/edgevox/metrics",
        "/edgevox/bot_token",
        "/edgevox/bot_sentence",
        "/edgevox/wakeword",
        "/edgevox/info",
    ]
    for topic in expected_pubs:
        assert topic in pub_topics, f"Missing publisher: {topic} (have: {pub_topics})"
    print(f"[PASS] All {len(expected_pubs)} publishers created")

    # Test 4: All subscribers present under default namespace
    sub_topics = [s.topic_name for s in bridge._node.subscriptions]
    expected_subs = [
        "/edgevox/tts_request",
        "/edgevox/command",
        "/edgevox/text_input",
        "/edgevox/interrupt",
        "/edgevox/set_language",
        "/edgevox/set_voice",
    ]
    for topic in expected_subs:
        assert topic in sub_topics, f"Missing subscriber: {topic} (have: {sub_topics})"
    print(f"[PASS] All {len(expected_subs)} subscribers created")

    # Test 5: Parameters declared with correct defaults
    params = bridge._node.get_parameters(["language", "voice", "muted"])
    assert params[0].value == "en", f"language default should be 'en', got {params[0].value}"
    assert params[1].value == "", f"voice default should be '', got {params[1].value}"
    assert params[2].value is False, f"muted default should be False, got {params[2].value}"
    print(f"[PASS] Parameters: language={params[0].value}, voice={params[1].value}, muted={params[2].value}")

    # Test 6: Publish methods
    bridge.publish_state("listening")
    bridge.publish_transcription("hello")
    bridge.publish_response("hi there")
    bridge.publish_audio_level(0.5)
    bridge.publish_metrics({"stt": 0.1, "llm": 0.2})
    bridge.publish_bot_token("hi")
    bridge.publish_bot_sentence("Hello there.")
    bridge.publish_wakeword("hey jarvis")
    bridge.publish_info({"query": "test", "ok": True})
    print("[PASS] All publish methods work")

    # Test 7: Callback setters
    bridge.set_tts_callback(lambda t: None)
    bridge.set_command_callback(lambda c: None)
    bridge.set_text_input_callback(lambda t: None)
    bridge.set_interrupt_callback(lambda: None)
    bridge.set_set_language_callback(lambda lang: None)
    bridge.set_set_voice_callback(lambda v: None)
    bridge.set_query_callback(lambda q: {"test": True})
    print("[PASS] All callback setters work")

    # Test 8: NullBridge no-ops
    null = NullBridge()
    null.publish_transcription("test")
    null.publish_response("test")
    null.publish_state("test")
    null.publish_audio_level(0.5)
    null.publish_metrics({})
    null.publish_bot_token("t")
    null.publish_bot_sentence("sentence")
    null.publish_wakeword("hey")
    null.publish_info({})
    null.set_tts_callback(lambda t: None)
    null.set_command_callback(lambda c: None)
    null.set_text_input_callback(lambda t: None)
    null.set_interrupt_callback(lambda: None)
    null.set_set_language_callback(lambda lang: None)
    null.set_set_voice_callback(lambda v: None)
    null.set_query_callback(lambda q: None)
    null.spin()
    null.shutdown()
    print("[PASS] NullBridge no-ops work")

    # Test 9: Custom namespace — topics resolve under the given namespace
    bridge.shutdown()
    rclpy.init()
    bridge2 = ROS2Bridge(namespace="/robot1/voice")
    pub_topics2 = [p.topic_name for p in bridge2._node.publishers]
    assert "/robot1/voice/transcription" in pub_topics2, f"Topics: {pub_topics2}"
    assert "/robot1/voice/bot_token" in pub_topics2
    assert "/robot1/voice/wakeword" in pub_topics2
    assert "/robot1/voice/state" in pub_topics2
    sub_topics2 = [s.topic_name for s in bridge2._node.subscriptions]
    assert "/robot1/voice/tts_request" in sub_topics2, f"Subs: {sub_topics2}"
    assert "/robot1/voice/command" in sub_topics2
    print("[PASS] Custom namespace: /robot1/voice/*")
    bridge2.shutdown()

    # Test 10: QoS profiles — verify actual policies on publishers
    rclpy.init()
    bridge3 = ROS2Bridge()

    # Build lookup: topic_name -> publisher
    pubs_by_topic = {p.topic_name: p for p in bridge3._node.publishers}

    # state topic should be TRANSIENT_LOCAL (late joiners get the last value)
    state_qos = pubs_by_topic["/edgevox/state"].qos_profile
    assert state_qos.durability == DurabilityPolicy.TRANSIENT_LOCAL, (
        f"state durability should be TRANSIENT_LOCAL, got {state_qos.durability}"
    )
    assert state_qos.reliability == ReliabilityPolicy.RELIABLE, (
        f"state reliability should be RELIABLE, got {state_qos.reliability}"
    )

    # audio_level should be BEST_EFFORT (sensor-like, drop stale)
    audio_qos = pubs_by_topic["/edgevox/audio_level"].qos_profile
    assert audio_qos.reliability == ReliabilityPolicy.BEST_EFFORT, (
        f"audio_level reliability should be BEST_EFFORT, got {audio_qos.reliability}"
    )

    # bot_token should be BEST_EFFORT
    token_qos = pubs_by_topic["/edgevox/bot_token"].qos_profile
    assert token_qos.reliability == ReliabilityPolicy.BEST_EFFORT, (
        f"bot_token reliability should be BEST_EFFORT, got {token_qos.reliability}"
    )

    # transcription should be RELIABLE + VOLATILE (standard reliable)
    tx_qos = pubs_by_topic["/edgevox/transcription"].qos_profile
    assert tx_qos.reliability == ReliabilityPolicy.RELIABLE, (
        f"transcription reliability should be RELIABLE, got {tx_qos.reliability}"
    )
    assert tx_qos.durability == DurabilityPolicy.VOLATILE, (
        f"transcription durability should be VOLATILE, got {tx_qos.durability}"
    )

    print("[PASS] QoS profiles verified: state=TRANSIENT_LOCAL, sensor=BEST_EFFORT, events=RELIABLE")
    bridge3.shutdown()

    # Test 11: create_bridge(enabled=False) returns NullBridge
    null2 = create_bridge(enabled=False)
    assert isinstance(null2, NullBridge), f"Expected NullBridge, got {type(null2)}"
    print("[PASS] create_bridge(enabled=False) returns NullBridge")

    # Test 12: create_bridge with custom namespace
    rclpy.init()
    bridge4 = create_bridge(enabled=True, namespace="/test/ns")
    assert isinstance(bridge4, ROS2Bridge)
    ptopics = [p.topic_name for p in bridge4._node.publishers]
    assert "/test/ns/transcription" in ptopics, f"Topics: {ptopics}"
    print("[PASS] create_bridge(namespace='/test/ns') works")
    bridge4.shutdown()

    print()
    print("All 12 tests passed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        sys.exit(1)
