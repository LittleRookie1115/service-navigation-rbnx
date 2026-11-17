import importlib.util
from concurrent.futures import Future
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from nav2_wrapper.rotation_guard_core import GuardLimits, RotationGuard, normalize_uuid_octets


class VelocityGuardDepthTest(unittest.TestCase):
    @staticmethod
    def _twist():
        return SimpleNamespace(
            linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            angular=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        )

    def setUp(self):
        node_api = Mock()
        node_api.get_clock.return_value.now.return_value.nanoseconds = 0
        node_type = type("Node", (), {
            "__init__": lambda _self, _name: None,
            **{name: getattr(node_api, name) for name in (
                "create_publisher", "create_subscription", "create_client",
                "create_timer", "get_clock", "get_logger",
            )},
        })
        self.status = SimpleNamespace(
            STATUS_ACCEPTED=1, STATUS_EXECUTING=2, STATUS_CANCELING=3,
            STATUS_SUCCEEDED=4, STATUS_CANCELED=5, STATUS_ABORTED=6,
        )
        self.status_qos = object()
        fake_ros = {
            "rclpy": SimpleNamespace(),
            "rclpy.node": SimpleNamespace(Node=node_type),
            "rclpy.executors": SimpleNamespace(ExternalShutdownException=RuntimeError),
            "rclpy.qos": SimpleNamespace(
                qos_profile_sensor_data=object(),
                qos_profile_action_status_default=self.status_qos,
            ),
            "action_msgs.msg": SimpleNamespace(GoalStatus=self.status, GoalStatusArray=object),
            "action_msgs.srv": SimpleNamespace(CancelGoal=SimpleNamespace(
                Request=lambda: SimpleNamespace(
                    goal_info=SimpleNamespace(goal_id=SimpleNamespace(uuid=[])),
                ),
            )),
            "geometry_msgs.msg": SimpleNamespace(Twist=self._twist),
            "nav2_msgs.action": SimpleNamespace(NavigateToPose=SimpleNamespace(
                Impl=SimpleNamespace(FeedbackMessage=object),
            )),
            "nav2_msgs.msg": SimpleNamespace(SpeedLimit=object),
            "nav_msgs.msg": SimpleNamespace(Odometry=object, Path=object),
            "sensor_msgs.msg": SimpleNamespace(LaserScan=object, PointCloud2=object),
        }
        spec = importlib.util.spec_from_file_location(
            "velocity_guard_under_test",
            Path(__file__).parent / "nav2_wrapper" / "velocity_guard.py",
        )
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, fake_ros):
            spec.loader.exec_module(self.module)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.environment = {
            "ROBONIX_VELOCITY_OUTPUT_TOPIC": "/test/guarded_velocity",
            "ROBONIX_NAV_MAX_LINEAR_SPEED_MPS": "0.3",
            "ROBONIX_NAV_DEFAULT_SPEED_PERCENTAGE": "100",
            "ROBONIX_NAV_GUARD_SPEED_LIMIT_TOPIC": "/test/speed_limit",
            "ROBONIX_NAV_TRACE_DIR": directory.name,
            "ROBONIX_GUARD_DEPTH_TOPIC": "/test/depth_points",
        }
        environment_patch = patch.dict(os.environ, self.environment, clear=True)
        environment_patch.start()
        self.addCleanup(environment_patch.stop)
        self.now = 10.0
        clock_patch = patch.object(self.module.time, "monotonic", side_effect=lambda: self.now)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.node_api = node_api

    def _node(self, **environment):
        self.node_api.create_publisher.return_value = Mock()
        cancel = Mock()
        cancel.service_is_ready.return_value = True
        cancel.call_async.side_effect = lambda _request: Future()
        self.node_api.create_client.return_value = cancel
        with patch.dict(os.environ, environment):
            node = self.module.VelocityGuardNode()
        self.addCleanup(node._scan_log.close)
        self.addCleanup(lambda: node._trace.close() if node._trace else None)
        return node

    def _goal(self, node, number=1):
        raw_uuid = [number] * 16
        self._status(node, (raw_uuid, self.status.STATUS_EXECUTING))
        return raw_uuid

    @staticmethod
    def _status_message(*goals):
        entries = []
        for raw_uuid, status, *stamp in goals:
            stamp_ns = stamp[0] if stamp else (raw_uuid[0] or 1) * 1_000_000_000
            entries.append(SimpleNamespace(
                goal_info=SimpleNamespace(
                    goal_id=SimpleNamespace(uuid=raw_uuid),
                    stamp=SimpleNamespace(
                        sec=stamp_ns // 1_000_000_000,
                        nanosec=stamp_ns % 1_000_000_000,
                    ),
                ),
                status=status,
            ))
        return SimpleNamespace(status_list=entries)

    def _status(self, node, *goals):
        node._on_status(self._status_message(*goals))

    @staticmethod
    def _feedback(node, raw_uuid):
        node._on_feedback(SimpleNamespace(
            goal_id=SimpleNamespace(uuid=raw_uuid),
            feedback=SimpleNamespace(
                current_pose=SimpleNamespace(pose=SimpleNamespace(
                    position=SimpleNamespace(x=0.1, y=0.2),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                )),
                distance_remaining=1.0,
                number_of_recoveries=0,
            ),
        ))

    @staticmethod
    def _cancel_response(raw_uuid=None, code=0, error=None):
        future = Future()
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(SimpleNamespace(
                return_code=code,
                goals_canceling=[] if raw_uuid is None else [
                    SimpleNamespace(goal_id=SimpleNamespace(uuid=raw_uuid)),
                ],
            ))
        return future

    @staticmethod
    def _depth(node, stamp=1):
        node._on_depth(SimpleNamespace(header=SimpleNamespace(
            stamp=SimpleNamespace(sec=stamp, nanosec=0),
        )))

    def _command(self, node):
        command = self._twist()
        command.linear.x = 0.2
        command.angular.z = 0.1
        node._on_cmd(command)

    def _assert_zero(self, node):
        command = node._pub.publish.call_args.args[0]
        self.assertEqual(
            (command.linear.x, command.linear.y, command.linear.z,
             command.angular.x, command.angular.y, command.angular.z),
            (0.0,) * 6,
        )

    def test_idle_never_writes_or_faults_with_missing_or_stale_depth(self):
        for depth_topic in ("/test/depth_points", ""):
            for received_depth in (False, True):
                with self.subTest(depth_topic=depth_topic, received_depth=received_depth):
                    node = self._node(ROBONIX_GUARD_DEPTH_TOPIC=depth_topic)
                    if received_depth:
                        self._depth(node)
                    for _ in range(20):
                        self.now += 10.0
                        node._publish_latched_zero()
                        self._command(node)
                        self._feedback(node, [1] * 16)
                    self.assertEqual(node._check_depth(), "")
                    self.assertEqual(node.guard.latched_reason, "")
                    self.assertEqual(node._active_uuid, "")
                    self.assertIsNone(node._robot_pose)
                    self.assertIsNone(node._trace)
                    node._pub.publish.assert_not_called()
                    node._cancel.call_async.assert_not_called()

    def test_idle_guard_does_not_overwrite_another_driver(self):
        for after_goal in (False, True):
            with self.subTest(after_goal=after_goal):
                node = self._node()
                if after_goal:
                    self._depth(node)
                    raw_uuid = self._goal(node)
                    self._command(node)
                    self._status(node, (raw_uuid, self.status.STATUS_SUCCEEDED))
                writes = []
                node._pub.publish.reset_mock()
                node._pub.publish.side_effect = writes.append
                driver_command = self._twist()
                driver_command.linear.x = 0.15
                other_driver = Mock(side_effect=writes.append)
                other_driver(driver_command)
                for _ in range(20):
                    self.now += 1.0
                    node._publish_latched_zero()
                    self._command(node)
                self.assertEqual(writes, [driver_command])
                self.assertEqual(node.guard.latched_reason, "")
                node._pub.publish.assert_not_called()
                node._cancel.call_async.assert_not_called()

    def test_missing_first_sample_blocks_and_cancels_exact_goal(self):
        node = self._node()
        raw_uuid = self._goal(node)
        self._command(node)
        self._assert_zero(node)
        self.assertIn("has not received", node.guard.latched_reason)
        request = node._cancel.call_async.call_args.args[0]
        self.assertEqual(request.goal_info.goal_id.uuid, raw_uuid)

    def test_timer_detects_expiry_without_another_command(self):
        node = self._node()
        self._depth(node)
        self._goal(node)
        self._command(node)
        self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)
        self.now += 1.0
        node._publish_latched_zero()
        self._assert_zero(node)
        self.assertIn("timeout", node.guard.latched_reason)
        node._pub.publish.reset_mock()
        self.now += 1.0
        node._publish_latched_zero()
        self._assert_zero(node)
        node._cancel.call_async.assert_called_once()

    def test_command_path_detects_expiry_without_waiting_for_timer(self):
        node = self._node()
        self._depth(node)
        self._goal(node)
        self.now += 1.1
        self._command(node)
        self._assert_zero(node)
        node._cancel.call_async.assert_called_once()

    def test_advancing_samples_keep_the_same_goal_healthy(self):
        node = self._node()
        self._depth(node)
        self._goal(node)
        for stamp in range(2, 8):
            self.now += 0.5
            self._depth(node, stamp)
            node._publish_latched_zero()
            self._command(node)
            self.assertEqual(node.guard.latched_reason, "")
            self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)
        node._cancel.call_async.assert_not_called()

    def test_repeated_older_and_zero_stamps_do_not_refresh_watchdog(self):
        for stamp in (0, 1, 2):
            with self.subTest(stamp=stamp):
                node = self._node()
                self._depth(node, 2)
                self._goal(node)
                received_at = node._depth_received_at
                self.now += 0.75
                self._depth(node, stamp)
                self.assertEqual(node._depth_received_at, received_at)
                self.now += 0.25
                node._publish_latched_zero()
                self._assert_zero(node)
                self.assertIn("timeout", node.guard.latched_reason)

    def test_terminal_status_retires_control_and_ignores_late_messages(self):
        for terminal in (self.status.STATUS_SUCCEEDED, self.status.STATUS_CANCELED,
                         self.status.STATUS_ABORTED, None):
            for faulted in (False, True):
                with self.subTest(terminal=terminal, faulted=faulted):
                    node = self._node()
                    raw_uuid = self._goal(node)
                    if faulted:
                        node._publish_latched_zero()
                    else:
                        self._depth(node)
                        self._command(node)
                    trace = node._trace
                    goals = () if terminal is None else ((raw_uuid, terminal),)
                    self._status(node, *goals)
                    self.assertTrue(trace.closed)
                    self.assertEqual(node._active_uuid, "")
                    self.assertEqual(node.guard.goal_uuid, "")
                    node._pub.publish.reset_mock()
                    node._cancel.call_async.reset_mock()
                    self._depth(node, 2)
                    for active in (self.status.STATUS_ACCEPTED, self.status.STATUS_EXECUTING,
                                   self.status.STATUS_CANCELING):
                        self._feedback(node, raw_uuid)
                        self._status(node, (raw_uuid, active))
                        self._command(node)
                        self.now += 1.0
                        node._publish_latched_zero()
                    self.assertEqual(node._active_uuid, "")
                    self.assertEqual(node.guard.latched_reason, "")
                    self.assertIsNone(node._robot_pose)
                    node._pub.publish.assert_not_called()
                    node._cancel.call_async.assert_not_called()

    def test_feedback_requires_status_ownership(self):
        node = self._node()
        raw_uuid = [1] * 16
        self._feedback(node, raw_uuid)
        self.assertEqual(node._active_uuid, "")
        self.assertIsNone(node._robot_pose)
        self._status(node, (raw_uuid, self.status.STATUS_ACCEPTED))
        self._feedback(node, raw_uuid)
        self.assertEqual(node._robot_pose, (0.1, 0.2, 0.0))
        self._feedback(node, [2] * 16)
        self.assertEqual(node._active_uuid, bytes(raw_uuid).hex())
        self.assertEqual(node._robot_pose, (0.1, 0.2, 0.0))

    def test_late_status_and_feedback_cannot_replace_a_new_goal(self):
        node = self._node()
        self._depth(node)
        old_uuid = self._goal(node)
        new_uuid = self._goal(node, 2)
        self._status(node, (new_uuid, self.status.STATUS_EXECUTING),
                     (old_uuid, self.status.STATUS_EXECUTING))
        self._feedback(node, old_uuid)
        self.assertIsNone(node._robot_pose)
        self._status(node, (old_uuid, self.status.STATUS_ABORTED),
                     (new_uuid, self.status.STATUS_EXECUTING))
        self._command(node)
        self.assertEqual(node._active_uuid, bytes(new_uuid).hex())
        self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)

    def test_pending_executing_goal_rejection_preserves_current_owner(self):
        for terminal in (self.status.STATUS_ABORTED, self.status.STATUS_CANCELED):
            with self.subTest(terminal=terminal):
                node = self._node()
                self._depth(node)
                current = self._goal(node)
                pending = [2] * 16
                trace = node._trace
                self._status(node, (current, self.status.STATUS_EXECUTING),
                             (pending, self.status.STATUS_EXECUTING))
                self.assertEqual(node._active_uuid, bytes(current).hex())
                self.assertIs(node._trace, trace)
                self.assertFalse(trace.closed)
                self._feedback(node, pending)
                self.assertIsNone(node._robot_pose)
                self._status(node, (current, self.status.STATUS_EXECUTING),
                             (pending, terminal))
                self._feedback(node, current)
                self._command(node)
                self.assertEqual(node._active_uuid, bytes(current).hex())
                self.assertEqual(node._robot_pose, (0.1, 0.2, 0.0))
                self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)
                node._cancel.call_async.assert_not_called()

    def test_pending_executing_goal_cannot_reset_fault_or_steal_cancel(self):
        node = self._node()
        current = self._goal(node)
        pending = [2] * 16
        node._publish_latched_zero()
        self._status(node, (current, self.status.STATUS_EXECUTING),
                     (pending, self.status.STATUS_EXECUTING))
        for _ in range(10):
            self._feedback(node, pending)
            self._command(node)
            node._publish_latched_zero()
        self.assertEqual(node._active_uuid, bytes(current).hex())
        self.assertEqual(node._pub.publish.call_count, 3)
        node._cancel.call_async.assert_called_once()
        self.assertEqual(
            node._cancel.call_async.call_args.args[0].goal_info.goal_id.uuid, current,
        )
        self._status(node, (current, self.status.STATUS_EXECUTING),
                     (pending, self.status.STATUS_ABORTED))
        self.assertTrue(node.guard.latched_reason)
        self.assertEqual(node._stop_frames_remaining, 0)

    def test_initial_multiple_live_goals_fail_closed_in_either_order(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                node = self._node()
                self._depth(node)
                current, pending = [1] * 16, [2] * 16
                goals = [(current, self.status.STATUS_EXECUTING),
                         (pending, self.status.STATUS_EXECUTING)]
                if reverse:
                    goals.reverse()
                for _ in range(10):
                    self._status(node, *goals)
                    self._feedback(node, current)
                    self._feedback(node, pending)
                    self._command(node)
                    node._publish_latched_zero()
                self.assertEqual(node._active_uuid, "")
                node._pub.publish.assert_not_called()
                node._cancel.call_async.assert_not_called()
                self._status(node, (current, self.status.STATUS_ABORTED),
                             (pending, self.status.STATUS_EXECUTING))
                self._command(node)
                self.assertEqual(node._active_uuid, bytes(pending).hex())
                self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)

    def test_terminal_current_goal_hands_off_to_pending_goal(self):
        node = self._node()
        self._depth(node)
        current = self._goal(node)
        pending = [2] * 16
        self._status(node, (current, self.status.STATUS_EXECUTING),
                     (pending, self.status.STATUS_EXECUTING))
        self._status(node, (current, self.status.STATUS_ABORTED),
                     (pending, self.status.STATUS_EXECUTING))
        self._feedback(node, current)
        self._command(node)
        self.assertEqual(node._active_uuid, bytes(pending).hex())
        self.assertIsNone(node._robot_pose)
        self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)

    def test_retained_live_and_terminal_status_use_action_status_qos(self):
        for status in (self.status.STATUS_EXECUTING, self.status.STATUS_SUCCEEDED):
            with self.subTest(status=status):
                node = self._node()
                self._depth(node)
                subscription = next(
                    call for call in self.node_api.create_subscription.call_args_list
                    if call.args[1] == "/navigate_to_pose/_action/status"
                    and call.args[2].__self__ is node
                )
                self.assertIs(subscription.args[3], self.status_qos)
                callback = subscription.args[2]
                raw_uuid = [1] * 16
                callback(self._status_message((raw_uuid, status)))
                self._feedback(node, raw_uuid)
                self._command(node)
                if status == self.status.STATUS_EXECUTING:
                    self.assertEqual(node._active_uuid, bytes(raw_uuid).hex())
                    self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)
                else:
                    callback(self._status_message((raw_uuid, self.status.STATUS_EXECUTING)))
                    self._command(node)
                    node._publish_latched_zero()
                    self.assertEqual(node._active_uuid, "")
                    node._pub.publish.assert_not_called()
                node._cancel.call_async.assert_not_called()

    def test_newer_failed_pending_goal_does_not_fence_out_unique_live_goal(self):
        for initially_ambiguous in (False, True):
            with self.subTest(initially_ambiguous=initially_ambiguous):
                node = self._node()
                self._depth(node)
                current, pending = [1] * 16, [2] * 16
                if initially_ambiguous:
                    self._status(node, (current, self.status.STATUS_EXECUTING, 10),
                                 (pending, self.status.STATUS_EXECUTING, 11))
                    self._feedback(node, current)
                    self._command(node)
                    self.assertEqual(node._active_uuid, "")
                    node._pub.publish.assert_not_called()
                self._status(node, (current, self.status.STATUS_EXECUTING, 10),
                             (pending, self.status.STATUS_ABORTED, 11))
                self._feedback(node, current)
                self._command(node)
                self.assertEqual(node._acceptance_fence, 11)
                self.assertEqual(node._active_uuid, bytes(current).hex())
                self.assertEqual(node._active_stamp, 10)
                self.assertEqual(node._robot_pose, (0.1, 0.2, 0.0))
                self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)
                self.now += 1.0
                node._publish_latched_zero()
                self.assertEqual(
                    node._cancel.call_async.call_args.args[0].goal_info.goal_id.uuid,
                    current,
                )

    def test_disappeared_candidate_cannot_return_below_acceptance_fence(self):
        node = self._node()
        current, pending = [1] * 16, [2] * 16
        self._status(node, (current, self.status.STATUS_EXECUTING, 10),
                     (pending, self.status.STATUS_EXECUTING, 11))
        self._status(node, (pending, self.status.STATUS_ABORTED, 11))
        self.assertEqual(node._live_candidates, {})
        self._status(node, (current, self.status.STATUS_EXECUTING, 10))
        self._feedback(node, current)
        self._command(node)
        node._publish_latched_zero()
        self.assertEqual(node._active_uuid, "")
        node._pub.publish.assert_not_called()
        node._cancel.call_async.assert_not_called()

    def test_live_candidate_overflow_stays_bounded_and_fails_closed(self):
        node = self._node()
        self._depth(node)
        goals = [
            (list(number.to_bytes(16, "big")), self.status.STATUS_EXECUTING, 100)
            for number in range(1, self.module._LIVE_CANDIDATE_LIMIT + 2)
        ]
        self._status(node, *goals)
        self.assertTrue(node._fence_closed)
        self.assertLessEqual(len(node._live_candidates), self.module._LIVE_CANDIDATE_LIMIT)
        self._status(node, goals[0])
        self._feedback(node, goals[0][0])
        self._command(node)
        node._publish_latched_zero()
        self.assertEqual(node._active_uuid, "")
        node._pub.publish.assert_not_called()
        newer = [200] * 16
        self._status(node, (newer, self.status.STATUS_EXECUTING, 101))
        self._command(node)
        self.assertEqual(node._active_uuid, bytes(newer).hex())
        self.assertFalse(node._fence_closed)
        self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)

    def test_acceptance_fence_bounds_history_and_rejects_evicted_uuids(self):
        node = self._node()
        for number in range(1, self.module._RETIRED_UUID_LIMIT * 2 + 2):
            raw_uuid = list(number.to_bytes(16, "big"))
            self._status(node, (raw_uuid, self.status.STATUS_EXECUTING, number))
            self.assertEqual(node._active_uuid, bytes(raw_uuid).hex())
            self._status(node, (raw_uuid, self.status.STATUS_SUCCEEDED, number))
            self.assertLessEqual(len(node._retired_uuids), self.module._RETIRED_UUID_LIMIT)
        self.assertNotIn(bytes((1).to_bytes(16, "big")).hex(), node._retired_uuids)
        for number in range(1, self.module._RETIRED_UUID_LIMIT * 2 + 2):
            raw_uuid = list(number.to_bytes(16, "big"))
            self._status(node, (raw_uuid, self.status.STATUS_EXECUTING, number))
            self._feedback(node, raw_uuid)
            self._command(node)
            node._publish_latched_zero()
        self.assertEqual(node._active_uuid, "")
        node._pub.publish.assert_not_called()
        node._cancel.call_async.assert_not_called()

    def test_equal_stamp_goals_can_progress_but_retired_uuids_cannot_return(self):
        node = self._node()
        self._depth(node)
        for number in range(1, 4):
            raw_uuid = [number] * 16
            self._status(node, (raw_uuid, self.status.STATUS_EXECUTING, 100))
            self._command(node)
            self.assertEqual(node._active_uuid, bytes(raw_uuid).hex())
            self._status(node, (raw_uuid, self.status.STATUS_CANCELED, 100))
        node._pub.publish.reset_mock()
        for number in range(1, 4):
            raw_uuid = [number] * 16
            self._status(node, (raw_uuid, self.status.STATUS_EXECUTING, 100))
            self._feedback(node, raw_uuid)
            self._command(node)
        self.assertEqual(node._active_uuid, "")
        node._pub.publish.assert_not_called()

    def test_equal_stamp_overflow_fails_closed_until_newer_acceptance(self):
        node = self._node()
        self._depth(node)
        for number in range(1, self.module._RETIRED_UUID_LIMIT + 2):
            raw_uuid = list(number.to_bytes(16, "big"))
            self._status(node, (raw_uuid, self.status.STATUS_ABORTED, 100))
        self.assertEqual(len(node._retired_uuids), self.module._RETIRED_UUID_LIMIT)
        self.assertTrue(node._fence_closed)
        fresh_uuid = [200] * 16
        for raw_uuid in (list((1).to_bytes(16, "big")), fresh_uuid):
            self._status(node, (raw_uuid, self.status.STATUS_EXECUTING, 100))
            self._feedback(node, raw_uuid)
            self._command(node)
        node._pub.publish.assert_not_called()
        self.assertEqual(node._active_uuid, "")
        self._status(node, (fresh_uuid, self.status.STATUS_EXECUTING, 101))
        self._command(node)
        self.assertFalse(node._fence_closed)
        self.assertEqual(node._active_uuid, bytes(fresh_uuid).hex())
        self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)
        self._status(node, (fresh_uuid, self.status.STATUS_SUCCEEDED, 101))
        node._pub.publish.reset_mock()
        self._status(node, (list((1).to_bytes(16, "big")), self.status.STATUS_EXECUTING, 100))
        self._command(node)
        node._pub.publish.assert_not_called()

    def test_tie_overflow_does_not_displace_reported_live_owner(self):
        node = self._node()
        self._depth(node)
        current = [1] * 16
        self._status(node, (current, self.status.STATUS_EXECUTING, 100))
        goals = [(current, self.status.STATUS_EXECUTING, 100)]
        goals.extend(
            (list(number.to_bytes(16, "big")), self.status.STATUS_ABORTED, 100)
            for number in range(2, self.module._RETIRED_UUID_LIMIT + 3)
        )
        self._status(node, *goals)
        self._command(node)
        self.assertTrue(node._fence_closed)
        self.assertEqual(node._active_uuid, bytes(current).hex())
        self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)
        self._status(node, (current, self.status.STATUS_SUCCEEDED, 100))
        node._pub.publish.reset_mock()
        self._status(node, (current, self.status.STATUS_EXECUTING, 100))
        self._command(node)
        node._pub.publish.assert_not_called()

    def test_zero_acceptance_stamp_is_silent_and_positive_nanoseconds_can_acquire(self):
        node = self._node()
        self._depth(node)
        raw_uuid = [1] * 16
        for stamp in (0, -1):
            self._status(node, (raw_uuid, self.status.STATUS_EXECUTING, stamp))
            self._feedback(node, raw_uuid)
            self._command(node)
            node._publish_latched_zero()
        self.assertEqual(node._active_uuid, "")
        node._pub.publish.assert_not_called()
        node._cancel.call_async.assert_not_called()
        self._status(node, (raw_uuid, self.status.STATUS_EXECUTING, 1))
        self._command(node)
        self.assertEqual(node._active_uuid, bytes(raw_uuid).hex())
        self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)

    def test_acceptance_clock_reset_cannot_replace_owner_and_restart_resets_fence(self):
        node = self._node()
        current = [1] * 16
        self._status(node, (current, self.status.STATUS_EXECUTING, 100))
        node._publish_latched_zero()
        pending = node._cancel_future
        reset_uuid = [2] * 16
        self._status(node, (reset_uuid, self.status.STATUS_EXECUTING, 50))
        self.assertEqual(node._active_uuid, bytes(current).hex())
        self.assertIs(node._cancel_future, pending)
        self._status(node, (current, self.status.STATUS_CANCELED, 100))
        self._status(node, (reset_uuid, self.status.STATUS_EXECUTING, 50))
        self.assertEqual(node._active_uuid, "")
        node._cancel.remove_pending_request.assert_called_once_with(pending)
        node._pub.publish.reset_mock()
        for _ in range(5):
            self._status(node, (reset_uuid, self.status.STATUS_EXECUTING, 50))
            self._feedback(node, reset_uuid)
            self._command(node)
            node._publish_latched_zero()
        node._pub.publish.assert_not_called()
        restarted = self._node()
        self._depth(restarted)
        self._status(restarted, (reset_uuid, self.status.STATUS_EXECUTING, 50))
        self._command(restarted)
        self.assertEqual(restarted._active_uuid, bytes(reset_uuid).hex())
        self.assertEqual(restarted._pub.publish.call_args.args[0].linear.x, 0.2)

    def test_older_terminal_snapshot_cannot_mutate_or_retire_live_owner(self):
        for faulted in (False, True):
            with self.subTest(faulted=faulted):
                node = self._node()
                current = [1] * 16
                self._status(node, (current, self.status.STATUS_EXECUTING, 10))
                if faulted:
                    node._publish_latched_zero()
                else:
                    self._depth(node)
                trace = node._trace
                future = node._cancel_future
                candidates = dict(node._live_candidates)
                retired = set(node._retired_uuids)
                self._status(node, ([2] * 16, self.status.STATUS_ABORTED, 5))
                self.assertEqual(node._active_uuid, bytes(current).hex())
                self.assertEqual(node._active_stamp, 10)
                self.assertEqual(node._acceptance_fence, 10)
                self.assertEqual(node._live_candidates, candidates)
                self.assertEqual(node._retired_uuids, retired)
                self.assertIs(node._cancel_future, future)
                self.assertFalse(trace.closed)
                node._cancel.remove_pending_request.assert_not_called()
                self._feedback(node, current)
                self._command(node)
                if not faulted:
                    self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)
                self._status(node)
                self.assertEqual(node._active_uuid, "")
                self.assertTrue(trace.closed)
                if future is not None:
                    node._cancel.remove_pending_request.assert_called_once_with(future)

    def test_terminal_entry_wins_over_stale_active_entry_in_same_status(self):
        node = self._node()
        raw_uuid = self._goal(node)
        self._status(node, (raw_uuid, self.status.STATUS_SUCCEEDED),
                     (raw_uuid, self.status.STATUS_EXECUTING))
        self._command(node)
        node._publish_latched_zero()
        self.assertEqual(node._active_uuid, "")
        node._pub.publish.assert_not_called()

    def test_depth_fault_has_finite_burst_despite_late_commands_and_feedback(self):
        node = self._node()
        raw_uuid = self._goal(node)
        self._command(node)
        self.assertEqual(node._pub.publish.call_count, 1)
        reason = node.guard.latched_reason
        for _ in range(100):
            self._command(node)
        self.assertEqual(node._pub.publish.call_count, 1)
        for _ in range(100):
            self._command(node)
            self._feedback(node, raw_uuid)
            node._publish_latched_zero()
        self.assertEqual(node._pub.publish.call_count, 3)
        self._depth(node)
        self._status(node, (raw_uuid, self.status.STATUS_EXECUTING))
        for _ in range(20):
            self._command(node)
            node._publish_latched_zero()
        self.assertEqual(node._pub.publish.call_count, 3)
        self.assertEqual(node.guard.latched_reason, reason)
        node._cancel.call_async.assert_called_once()
        for call in node._pub.publish.call_args_list:
            command = call.args[0]
            self.assertEqual(
                (command.linear.x, command.linear.y, command.linear.z,
                 command.angular.x, command.angular.y, command.angular.z),
                (0.0,) * 6,
            )

    def test_new_goal_with_recovered_data_can_move(self):
        node = self._node()
        self._depth(node)
        self._goal(node)
        self.now += 1.1
        node._publish_latched_zero()
        self._depth(node, 2)
        self._goal(node, 2)
        self._command(node)
        self.assertEqual(node.guard.latched_reason, "")
        self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)
        self.assertFalse(node._cancel_sent)

    def test_retired_faulted_goal_allows_only_a_new_uuid_with_fresh_depth(self):
        node = self._node()
        old_uuid = self._goal(node)
        node._publish_latched_zero()
        self._status(node, (old_uuid, self.status.STATUS_CANCELED))
        self._depth(node)
        node._pub.publish.reset_mock()
        self._status(node, (old_uuid, self.status.STATUS_EXECUTING))
        self._command(node)
        node._pub.publish.assert_not_called()
        new_uuid = self._goal(node, 2)
        self._feedback(node, old_uuid)
        self._command(node)
        self.assertEqual(node._active_uuid, bytes(new_uuid).hex())
        self.assertEqual(node.guard.latched_reason, "")
        self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)

    def test_rotation_fault_retires_without_idle_publication(self):
        node = self._node()
        self._depth(node)
        self._goal(node)
        node.guard.latched_reason = "terminal rotation timeout"
        node._on_status(SimpleNamespace(status_list=[]))
        self.assertEqual(node.guard.latched_reason, "")
        self._command(node)
        node._publish_latched_zero()
        node._pub.publish.assert_not_called()
        self._goal(node, 2)
        self._command(node)
        self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)
        node._cancel.call_async.assert_not_called()

    def test_new_goal_cannot_refresh_old_data(self):
        node = self._node()
        self._depth(node, 2)
        self._goal(node)
        self.now += 1.1
        node._publish_latched_zero()
        self._depth(node, 2)
        raw_uuid = self._goal(node, 2)
        self._command(node)
        self._assert_zero(node)
        self.assertIn("timeout", node.guard.latched_reason)
        request = node._cancel.call_async.call_args.args[0]
        self.assertEqual(request.goal_info.goal_id.uuid, raw_uuid)

    def test_cancel_unavailable_or_send_failure_retries_from_timer(self):
        node = self._node()
        self._goal(node)
        node._cancel.service_is_ready.return_value = False
        self._command(node)
        self.assertFalse(node._cancel_sent)
        node._cancel.call_async.assert_not_called()
        node._cancel.service_is_ready.return_value = True
        node._cancel.call_async.side_effect = [RuntimeError("unavailable"), Future()]
        self.now += self.module._CANCEL_RETRY_S
        node._publish_latched_zero()
        self._assert_zero(node)
        self.assertFalse(node._cancel_sent)
        self.now += self.module._CANCEL_RETRY_S
        node._publish_latched_zero()
        self.assertTrue(node._cancel_sent)
        self.assertEqual(node._cancel.call_async.call_count, 2)
        for _ in range(20):
            node._publish_latched_zero()
            self._command(node)
        self.assertEqual(node._pub.publish.call_count, 3)
        self.assertEqual(node._cancel.call_async.call_count, 2)

    def test_missing_or_zero_uuid_never_cancels_all_goals(self):
        for raw_uuid in (None, [0] * 16):
            with self.subTest(raw_uuid=raw_uuid):
                node = self._node()
                self._goal(node)
                node._active_uuid_raw = raw_uuid
                for _ in range(20):
                    node._publish_latched_zero()
                self._assert_zero(node)
                node._cancel.call_async.assert_not_called()
                self.assertEqual(node._pub.publish.call_count, 3)

    def test_zero_uuid_cannot_take_control(self):
        node = self._node()
        self._status(node, ([0] * 16, self.status.STATUS_EXECUTING))
        self._command(node)
        node._publish_latched_zero()
        self.assertEqual(node._active_uuid, "")
        node._pub.publish.assert_not_called()
        node._cancel.call_async.assert_not_called()

    def test_cancel_service_or_send_recovers_after_burst_without_more_frames(self):
        for failure in ("unavailable", "exception"):
            with self.subTest(failure=failure):
                node = self._node()
                raw_uuid = self._goal(node)
                if failure == "unavailable":
                    node._cancel.service_is_ready.return_value = False
                else:
                    node._cancel.call_async.side_effect = RuntimeError("unavailable")
                for _ in range(30):
                    self._command(node)
                    node._publish_latched_zero()
                self.assertEqual(node._pub.publish.call_count, 3)
                self.assertEqual(node._cancel.service_is_ready.call_count, 1)
                if failure == "unavailable":
                    node._cancel.call_async.assert_not_called()
                else:
                    node._cancel.call_async.assert_called_once()
                before = node._cancel.call_async.call_count
                node._cancel.service_is_ready.return_value = True
                node._cancel.call_async.side_effect = (
                    lambda _request: self._cancel_response(raw_uuid)
                )
                self.now += 1.0
                node._publish_latched_zero()
                node._publish_latched_zero()
                self.assertEqual(node._cancel.call_async.call_count, before + 1)
                self.assertTrue(node._cancel_sent)
                self.assertEqual(node._pub.publish.call_count, 3)
                for call in node._cancel.call_async.call_args_list:
                    self.assertEqual(call.args[0].goal_info.goal_id.uuid, raw_uuid)

    def test_cancel_failure_retries_with_backoff_and_without_more_frames(self):
        for response in ("rejected", "empty", "wrong_uuid", "exception"):
            with self.subTest(response=response):
                node = self._node()
                raw_uuid = self._goal(node)
                if response == "rejected":
                    node._cancel.call_async.side_effect = (
                        lambda _request: self._cancel_response(code=1)
                    )
                elif response == "empty":
                    node._cancel.call_async.side_effect = (
                        lambda _request: self._cancel_response()
                    )
                elif response == "wrong_uuid":
                    node._cancel.call_async.side_effect = (
                        lambda _request: self._cancel_response([2] * 16)
                    )
                else:
                    node._cancel.call_async.side_effect = (
                        lambda _request: self._cancel_response(error=RuntimeError("lost"))
                    )
                for _ in range(30):
                    node._publish_latched_zero()
                    self._command(node)
                self.assertEqual(node._pub.publish.call_count, 3)
                node._cancel.call_async.assert_called_once()
                self.assertFalse(node._cancel_sent)
                self.now += self.module._CANCEL_RETRY_S
                node._publish_latched_zero()
                node._publish_latched_zero()
                self.assertEqual(node._cancel.call_async.call_count, 2)
                self.assertFalse(node._cancel_sent)
                node._cancel.call_async.side_effect = (
                    lambda _request: self._cancel_response(raw_uuid)
                )
                self.now += self.module._CANCEL_RETRY_S
                node._publish_latched_zero()
                node._publish_latched_zero()
                self.assertEqual(node._cancel.call_async.call_count, 3)
                self.assertTrue(node._cancel_sent)
                self.assertEqual(node._pub.publish.call_count, 3)
                for call in node._cancel.call_async.call_args_list:
                    self.assertEqual(call.args[0].goal_info.goal_id.uuid, raw_uuid)

    def test_cancel_rejection_can_recover_without_extending_burst(self):
        node = self._node()
        raw_uuid = self._goal(node)
        node._cancel.call_async.side_effect = [
            self._cancel_response(code=1), self._cancel_response(raw_uuid),
        ]
        node._publish_latched_zero()
        node._publish_latched_zero()
        self.now += self.module._CANCEL_RETRY_S
        for _ in range(20):
            node._publish_latched_zero()
        self.assertEqual(node._pub.publish.call_count, 3)
        self.assertEqual(node._cancel.call_async.call_count, 2)
        self.assertTrue(node._cancel_sent)

    def test_pending_or_accepted_cancel_never_extends_zero_burst(self):
        for accepted in (False, True):
            with self.subTest(accepted=accepted):
                node = self._node()
                raw_uuid = self._goal(node)
                if accepted:
                    node._cancel.call_async.side_effect = (
                        lambda _request: self._cancel_response(raw_uuid)
                    )
                for _ in range(30):
                    node._publish_latched_zero()
                    self._command(node)
                self.assertEqual(node._pub.publish.call_count, 3)
                node._cancel.call_async.assert_called_once()

    def test_terminal_status_stops_burst_and_cancel_retries_immediately(self):
        node = self._node()
        raw_uuid = self._goal(node)
        node._cancel.call_async.side_effect = lambda _request: self._cancel_response(code=1)
        node._publish_latched_zero()
        self._status(node, (raw_uuid, self.status.STATUS_CANCELED))
        for _ in range(20):
            node._publish_latched_zero()
            self._command(node)
        self.assertEqual(node._pub.publish.call_count, 1)
        node._cancel.call_async.assert_called_once()

    def test_cancel_backoff_continues_after_three_zero_frames(self):
        node = self._node()
        raw_uuid = self._goal(node)
        started = self.now
        node._cancel.service_is_ready.return_value = False
        node._publish_latched_zero()
        self.now = started + self.module._CANCEL_RETRY_S / 2
        node._publish_latched_zero()
        self.assertEqual(node._cancel.service_is_ready.call_count, 1)
        self.now = started + self.module._CANCEL_RETRY_S
        node._publish_latched_zero()
        self.assertEqual(node._cancel.service_is_ready.call_count, 2)
        self.assertEqual(node._pub.publish.call_count, 3)
        node._cancel.service_is_ready.return_value = True
        node._cancel.call_async.side_effect = (
            lambda _request: self._cancel_response(raw_uuid)
        )
        self.now = started + self.module._CANCEL_RETRY_S * 1.5
        node._publish_latched_zero()
        node._cancel.call_async.assert_not_called()
        self.now = started + self.module._CANCEL_RETRY_S * 2
        node._publish_latched_zero()
        node._publish_latched_zero()
        self.assertTrue(node._cancel_sent)
        node._cancel.call_async.assert_called_once()
        self.assertEqual(node._pub.publish.call_count, 3)

    def test_pending_cancel_timeout_cleans_up_then_retries_after_backoff(self):
        node = self._node()
        raw_uuid = self._goal(node)
        first, second = Future(), Future()
        node._cancel.call_async.side_effect = [
            first, second, self._cancel_response(raw_uuid),
        ]
        started = self.now
        for _ in range(10):
            node._publish_latched_zero()
        node._cancel.call_async.assert_called_once()
        self.assertIs(node._cancel_future, first)
        self.now = started + self.module._CANCEL_REPLY_TIMEOUT_S
        node._publish_latched_zero()
        node._cancel.remove_pending_request.assert_called_once_with(first)
        self.assertIsNone(node._cancel_future)
        self.assertFalse(node._cancel_sent)
        self.now += self.module._CANCEL_RETRY_S / 2
        node._publish_latched_zero()
        node._cancel.call_async.assert_called_once()
        self.now = started + self.module._CANCEL_REPLY_TIMEOUT_S + self.module._CANCEL_RETRY_S
        node._publish_latched_zero()
        self.assertIs(node._cancel_future, second)
        first.set_result(SimpleNamespace(
            return_code=0,
            goals_canceling=[SimpleNamespace(goal_id=SimpleNamespace(uuid=raw_uuid))],
        ))
        node._publish_latched_zero()
        self.assertIs(node._cancel_future, second)
        self.assertEqual(node._cancel.call_async.call_count, 2)
        second.set_result(SimpleNamespace(return_code=1, goals_canceling=[]))
        node._publish_latched_zero()
        self.assertFalse(node._cancel_sent)
        self.assertIsNone(node._cancel_future)
        self.now += self.module._CANCEL_RETRY_S
        node._publish_latched_zero()
        node._publish_latched_zero()
        self.assertTrue(node._cancel_sent)
        self.assertEqual(node._cancel.call_async.call_count, 3)
        self.assertEqual(node._pub.publish.call_count, 3)
        for call in node._cancel.call_async.call_args_list:
            self.assertEqual(call.args[0].goal_info.goal_id.uuid, raw_uuid)

    def test_retirement_removes_pending_request_and_ignores_late_completion(self):
        for terminal in (self.status.STATUS_SUCCEEDED, self.status.STATUS_CANCELED,
                         self.status.STATUS_ABORTED, None):
            with self.subTest(terminal=terminal):
                node = self._node()
                raw_uuid = self._goal(node)
                node._publish_latched_zero()
                future = node._cancel_future
                self._status(node, *((raw_uuid, terminal),) if terminal is not None else ())
                node._cancel.remove_pending_request.assert_called_once_with(future)
                self.assertIsNone(node._cancel_future)
                future.set_result(SimpleNamespace(return_code=1, goals_canceling=[]))
                self.now += 10.0
                for _ in range(10):
                    node._publish_latched_zero()
                    self._feedback(node, raw_uuid)
                    self._command(node)
                self.assertEqual(node._active_uuid, "")
                self.assertEqual(node._pub.publish.call_count, 1)
                node._cancel.call_async.assert_called_once()

    def test_late_cancel_reply_cannot_affect_a_new_goal(self):
        node = self._node()
        self._goal(node)
        future = Future()
        node._cancel.call_async.side_effect = lambda _request: future
        node._publish_latched_zero()
        self._depth(node)
        new_uuid = self._goal(node, 2)
        node._cancel.remove_pending_request.assert_called_once_with(future)
        future.set_result(SimpleNamespace(return_code=1, goals_canceling=[]))
        node._publish_latched_zero()
        self._command(node)
        self.assertEqual(node._active_uuid, bytes(new_uuid).hex())
        self.assertEqual(node.guard.latched_reason, "")
        self.assertFalse(node._cancel_sent)
        self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)

    def test_depth_disabled_preserves_command_and_idle_behavior(self):
        node = self._node(
            ROBONIX_GUARD_DEPTH_TOPIC="",
            ROBONIX_GUARD_DEPTH_TIMEOUT_S="nan",
        )
        self._goal(node)
        self.now += 100.0
        self._command(node)
        self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)
        node._pub.publish.reset_mock()
        node._publish_latched_zero()
        node._pub.publish.assert_not_called()
        node._cancel.call_async.assert_not_called()
        self.assertFalse(any(
            call.args[1] == "/test/depth_points"
            for call in node.create_subscription.call_args_list
        ))
        node.guard.latched_reason = "rotation limit"
        node._on_status(SimpleNamespace(status_list=[]))
        self.assertEqual(node.guard.latched_reason, "")

    def test_upstream_rotation_timeout_still_cancels_with_finite_burst(self):
        node = self._node(ROBONIX_GUARD_DEPTH_TOPIC="")
        raw_uuid = self._goal(node)
        command = self._twist()
        command.angular.z = 0.2
        node._on_cmd(command)
        self.assertEqual(node.guard.latched_reason, "")
        self.node_api.get_clock.return_value.now.return_value.nanoseconds = 26_000_000_000
        node._on_cmd(command)
        self.assertEqual(node.guard.latched_reason, "continuous stationary rotation timeout")
        for _ in range(20):
            node._on_cmd(command)
            node._publish_latched_zero()
        self.assertEqual(node._pub.publish.call_count, 4)
        self._assert_zero(node)
        request = node._cancel.call_async.call_args.args[0]
        self.assertEqual(request.goal_info.goal_id.uuid, raw_uuid)
        node._cancel.call_async.assert_called_once()

    def test_active_goal_preserves_planar_speed_limits_and_angular_command(self):
        for value, percentage, expected in ((50.0, True, 0.15),
                                            (0.2, False, 0.2),
                                            (0.0, True, 0.3)):
            with self.subTest(value=value, percentage=percentage):
                node = self._node()
                self._depth(node)
                self._goal(node)
                node._on_speed_limit(SimpleNamespace(speed_limit=value, percentage=percentage))
                command = self._twist()
                command.linear.x, command.linear.y, command.linear.z = 0.6, 0.8, 0.1
                command.angular.x, command.angular.y, command.angular.z = 0.2, 0.3, 0.4
                node._on_cmd(command)
                output = node._pub.publish.call_args.args[0]
                self.assertAlmostEqual(math.hypot(output.linear.x, output.linear.y), expected)
                self.assertAlmostEqual(output.linear.x / output.linear.y, 0.75)
                self.assertEqual(output.linear.z, 0.1)
                self.assertEqual(
                    (output.angular.x, output.angular.y, output.angular.z), (0.2, 0.3, 0.4),
                )
                node._cancel.call_async.assert_not_called()

    def test_fault_and_retirement_preserve_trace_diagnostics(self):
        node = self._node()
        self._depth(node)
        raw_uuid = self._goal(node)
        trace_path = Path(node._trace.name)
        node._on_speed_limit(SimpleNamespace(speed_limit=50.0, percentage=True))
        self._command(node)
        node._cancel.call_async.side_effect = lambda _request: self._cancel_response(code=1)
        self.now += 1.0
        node._publish_latched_zero()
        node._publish_latched_zero()
        self._status(node, (raw_uuid, self.status.STATUS_CANCELED))
        records = [json.loads(line) for line in trace_path.read_text().splitlines()]
        events = {record["event"] for record in records}
        self.assertTrue({
            "goal_active", "speed_limit", "cmd", "depth_timeout", "guard_trip",
            "cancel_requested", "cancel_failed", "status", "goal_inactive", "trace_closed",
        }.issubset(events))
        command = next(record for record in records if record["event"] == "cmd")
        self.assertTrue(command["linear_limited"])
        self.assertAlmostEqual(command["linear"], 0.15)

    def test_depth_timeout_and_topic_are_validated_before_ros_endpoints(self):
        for timeout in ("0", "-1", "nan", "inf", "-inf", "bad"):
            with self.subTest(timeout=timeout):
                self.node_api.create_publisher.reset_mock()
                with self.assertRaises(ValueError):
                    self._node(ROBONIX_GUARD_DEPTH_TIMEOUT_S=timeout)
                self.node_api.create_publisher.assert_not_called()
        with self.assertRaises(ValueError):
            self._node(ROBONIX_GUARD_DEPTH_TOPIC="relative")

    def test_configured_timeout_allows_fresh_data_until_deadline(self):
        node = self._node(ROBONIX_GUARD_DEPTH_TIMEOUT_S="2.5")
        self._depth(node)
        self._goal(node)
        self.now += 2.0
        self._command(node)
        self.assertEqual(node.guard.latched_reason, "")
        self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.2)
        self.now += 0.5
        node._publish_latched_zero()
        self._assert_zero(node)


class RotationGuardTest(unittest.TestCase):
    def test_ros_uuid_octets_are_plain_python_ints(self):
        class ByteLike:
            def __init__(self, value):
                self.value = value

            def __int__(self):
                return self.value

        result = normalize_uuid_octets(ByteLike(value) for value in range(16))
        self.assertEqual(result, list(range(16)))
        self.assertTrue(all(type(value) is int for value in result))

    def test_terminal_rotation_without_progress_latches(self):
        guard = RotationGuard(GuardLimits(terminal_no_progress_s=3.0))
        guard.begin_goal("goal-a")
        guard.observe_odom(0.0)
        self.assertEqual(guard.evaluate(0.0, 0.0, 0.2, (0, 0, 0), (0.1, 0, 1.0)), "")
        guard.observe_odom(0.2)
        reason = guard.evaluate(3.1, 0.0, 0.2, (0, 0, 0.0), (0.1, 0, 1.0))
        self.assertEqual(reason, "terminal yaw made no progress")
        self.assertEqual(guard.evaluate(3.2, 0.0, 0.0, None, None), reason)

    def test_new_action_uuid_is_the_only_reset(self):
        guard = RotationGuard(GuardLimits(terminal_no_progress_s=1.0))
        guard.begin_goal("goal-a")
        guard.evaluate(0.0, 0.0, 0.2, (0, 0, 0), (0.1, 0, 1.0))
        guard.evaluate(1.1, 0.0, 0.2, (0, 0, 0), (0.1, 0, 1.0))
        self.assertTrue(guard.latched_reason)
        guard.begin_goal("goal-a")
        self.assertTrue(guard.latched_reason)
        guard.begin_goal("goal-b")
        self.assertFalse(guard.latched_reason)

    def test_global_guard_catches_recovery_spin_away_from_goal(self):
        guard = RotationGuard(GuardLimits(global_spin_limit_rad=1.0))
        guard.begin_goal("goal-a")
        guard.observe_odom(0.0)
        guard.evaluate(0.0, 0.0, 0.2, None, None)
        guard.observe_odom(0.6)
        guard.observe_odom(1.2)
        reason = guard.evaluate(1.0, 0.0, 0.2, None, None)
        self.assertEqual(reason, "continuous stationary rotation limit")

    def test_global_timeout_means_no_measured_yaw_progress(self):
        guard = RotationGuard(
            GuardLimits(global_spin_timeout_s=2.0, global_spin_limit_rad=4.0)
        )
        guard.begin_goal("goal-a")
        guard.observe_odom(0.0)
        self.assertEqual(guard.evaluate(0.0, 0.0, 0.2, None, None), "")
        guard.observe_odom(0.02)
        self.assertEqual(guard.evaluate(1.0, 0.0, 0.2, None, None), "")
        self.assertEqual(
            guard.evaluate(2.1, 0.0, 0.2, None, None),
            "continuous stationary rotation timeout",
        )

    def test_measured_yaw_progress_extends_global_timeout(self):
        guard = RotationGuard(
            GuardLimits(
                global_spin_timeout_s=2.0,
                global_spin_limit_rad=4.0,
                global_min_progress_rad=0.04,
            )
        )
        guard.begin_goal("goal-a")
        guard.observe_odom(0.0)
        self.assertEqual(guard.evaluate(0.0, 0.0, 0.2, None, None), "")
        for now, yaw in ((1.5, 0.05), (3.0, 0.10), (4.5, 0.15)):
            guard.observe_odom(yaw)
            self.assertEqual(guard.evaluate(now, 0.0, 0.2, None, None), "")

    def test_legitimate_half_turn_is_below_global_limit(self):
        guard = RotationGuard()
        guard.begin_goal("goal-a")
        guard.observe_odom(0.0)
        guard.evaluate(0.0, 0.0, 0.2, None, None)
        for i in range(1, 17):
            guard.observe_odom(i * math.pi / 16.0)
            self.assertEqual(guard.evaluate(i * 0.5, 0.0, 0.2, None, None), "")


if __name__ == "__main__":
    unittest.main()
