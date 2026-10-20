import ast
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from nav2_wrapper.configuration import resolve_velocity_output_topic
from nav2_wrapper.speed_control import speed_settings


ROOT = Path(__file__).resolve().parent


class CallbackFuture(Future):
    def result(self, timeout=None):
        if not self.done():
            raise AssertionError("future.result() must only run in a completion callback")
        return super().result(timeout)


def pose_message():
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="map"),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=0.0),
            orientation=SimpleNamespace(z=0.0, w=1.0),
        ),
    )


class NavigationPathCheckTest(unittest.TestCase):
    def setUp(self):
        functions = {
            "_make_pose", "_goal_status_name", "_canonical_state",
            "_resolve_run_id", "_navigate_impl", "_status_impl", "_cancel_impl",
            "_quat_to_yaw", "_dispatch_goal", "_finish_path_check",
            "_expire_path_checks", "_path_response_cb", "_path_result_cb",
            "_issue_cancel", "_cancel_response_cb", "_goal_response_cb",
        }
        source = ast.parse((ROOT / "nav2_wrapper" / "atlas_bridge.py").read_text())
        definitions = [
            node for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name in functions
        ]
        self.assertEqual({node.name for node in definitions}, functions)
        self.clock = Mock()
        self.clock.monotonic.return_value = 100.0
        self.planner = Mock()
        self.planner.server_is_ready.return_value = True
        self.planning_response = CallbackFuture()
        self.planner.send_goal_async.return_value = self.planning_response
        self.navigator = Mock()
        self.navigation_response = CallbackFuture()
        self.navigator.send_goal_async.return_value = self.navigation_response
        self.restore_speed = Mock()
        self.namespace = {
            "math": math, "uuid": uuid, "time": self.clock,
            "_state_lock": threading.Lock(), "_nav_queue": queue.Queue(),
            "_goal_states": {}, "_goal_handles": {}, "_path_checks": {},
            "_last_run_id": "", "_ros_node": Mock(),
            "_nav_action_client": self.navigator, "_nav_action_ready": True,
            "_path_action_client": self.planner, "_PATH_CHECK_TIMEOUT_S": 5.0,
            "_ComputePathToPose": SimpleNamespace(Goal=SimpleNamespace),
            "_NavigateToPose": SimpleNamespace(Goal=SimpleNamespace),
            "_PoseStamped": pose_message,
            "_GoalStatus": SimpleNamespace(
                STATUS_UNKNOWN=0, STATUS_ACCEPTED=1, STATUS_EXECUTING=2,
                STATUS_CANCELING=3, STATUS_SUCCEEDED=4, STATUS_CANCELED=5,
                STATUS_ABORTED=6,
            ),
            "_nav_diagnostics": [], "_restore_speed_after_run": self.restore_speed,
            "_feedback_cb": Mock(), "_result_cb": Mock(),
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "<navigation>", "exec"), self.namespace)

    def start_check(self, goal=None):
        accepted = self.namespace["_navigate_impl"](goal or pose_message())
        self.assertTrue(accepted["accepted"])
        self.dispatch_next()
        return accepted["run_id"]

    def dispatch_next(self):
        run_id, payload = self.namespace["_nav_queue"].get_nowait()
        self.namespace["_dispatch_goal"](self.namespace["_ros_node"], run_id, payload)

    def accept_planning(self):
        handle = Mock()
        handle.accepted = True
        handle.get_result_async.return_value = CallbackFuture()
        handle.cancel_goal_async.return_value = CallbackFuture()
        self.planning_response.set_result(handle)
        return handle

    def complete_planning(self, handle, status=4, poses=None):
        handle.get_result_async.return_value.set_result(SimpleNamespace(
            status=status,
            result=SimpleNamespace(path=SimpleNamespace(
                poses=[pose_message()] if poses is None else poses,
            )),
        ))

    def assert_no_navigation(self, run_id, state, detail):
        status = self.namespace["_status_impl"](run_id)
        self.assertEqual(status["state"], state)
        self.assertIn(detail, status["detail"])
        self.navigator.send_goal_async.assert_not_called()
        self.assertTrue(self.namespace["_nav_queue"].empty())

    def test_nonempty_successful_plan_is_required_before_navigation(self):
        run_id = self.start_check()
        self.assert_no_navigation(run_id, "RUNNING", "checking path")
        handle = self.accept_planning()
        self.navigator.send_goal_async.assert_not_called()
        self.complete_planning(handle)
        self.assertEqual(self.namespace["_status_impl"](run_id)["state"], "PENDING")
        self.dispatch_next()
        self.navigator.send_goal_async.assert_called_once()
        self.assertEqual(self.namespace["_status_impl"](run_id)["state"], "RUNNING")
        self.assertFalse(self.namespace["_path_checks"])
        self.assertFalse(self.namespace["_goal_handles"])
        self.restore_speed.assert_not_called()

    def test_goal_position_frame_and_explicit_yaw_are_preserved(self):
        for yaw in (0.0, math.pi / 2, -math.pi / 3):
            with self.subTest(yaw=yaw):
                self.setUp()
                goal = pose_message()
                goal.header.frame_id = "odom"
                goal.pose.position.x = -3.5
                goal.pose.orientation.z = math.sin(yaw / 2)
                goal.pose.orientation.w = math.cos(yaw / 2)
                self.start_check(goal)
                planning_goal = self.planner.send_goal_async.call_args.args[0]
                self.assertFalse(planning_goal.use_start)
                self.assertEqual(planning_goal.goal.header.frame_id, "odom")
                self.assertEqual(planning_goal.goal.pose.position.x, -3.5)
                self.assertAlmostEqual(planning_goal.goal.pose.orientation.z, math.sin(yaw / 2))
                handle = self.accept_planning()
                self.complete_planning(handle)
                self.dispatch_next()
                navigation_goal = self.navigator.send_goal_async.call_args.args[0]
                self.assertEqual(navigation_goal.pose, planning_goal.goal)

    def test_unavailable_planner_fails_without_driving(self):
        for planner in (None, self.planner):
            with self.subTest(planner=planner):
                self.namespace["_path_action_client"] = planner
                self.planner.server_is_ready.return_value = False
                run_id = self.start_check()
                self.assert_no_navigation(run_id, "FAILED", "action server not ready")
                self.assertFalse(self.namespace["_path_checks"])
                self.planner.send_goal_async.assert_not_called()

    def test_planning_rejection_and_send_exception_fail_without_driving(self):
        run_id = self.start_check()
        self.planning_response.set_result(SimpleNamespace(accepted=False))
        self.assert_no_navigation(run_id, "FAILED", "precheck rejected")
        self.restore_speed.assert_called_once_with(run_id)
        self.planner.send_goal_async.side_effect = RuntimeError("planner disconnected")
        next_run_id = self.start_check()
        self.assert_no_navigation(next_run_id, "FAILED", "planner disconnected")

    def test_failed_empty_and_malformed_results_never_drive(self):
        for status, poses, detail in (
            (6, [pose_message()], "aborted"),
            (5, [pose_message()], "canceled"),
            (4, [], "empty path"),
        ):
            with self.subTest(status=status, poses=poses):
                self.setUp()
                run_id = self.start_check()
                handle = self.accept_planning()
                self.complete_planning(handle, status, poses)
                self.assert_no_navigation(run_id, "FAILED", detail)
                self.assertFalse(self.namespace["_path_checks"])
                self.assertFalse(self.namespace["_goal_handles"])
        self.setUp()
        run_id = self.start_check()
        handle = self.accept_planning()
        handle.get_result_async.return_value.set_exception(RuntimeError("result disconnected"))
        self.assert_no_navigation(run_id, "FAILED", "result disconnected")

    def test_planning_acceptance_future_error_fails_without_driving(self):
        run_id = self.start_check()
        self.planning_response.set_exception(RuntimeError("acceptance disconnected"))
        self.assert_no_navigation(run_id, "FAILED", "acceptance disconnected")

    def test_precheck_timeout_cancels_accepted_planner_and_ignores_late_success(self):
        run_id = self.start_check()
        handle = self.accept_planning()
        self.clock.monotonic.return_value = 105.0
        self.namespace["_expire_path_checks"]()
        self.assert_no_navigation(run_id, "FAILED", "timed out after 5.0s")
        handle.cancel_goal_async.assert_called_once()
        self.complete_planning(handle)
        self.assert_no_navigation(run_id, "FAILED", "timed out after 5.0s")
        self.assertFalse(self.namespace["_path_checks"])
        self.assertFalse(self.namespace["_goal_handles"])

    def test_timeout_before_acceptance_cancels_late_planning_handle(self):
        run_id = self.start_check()
        self.clock.monotonic.return_value = 105.0
        self.namespace["_expire_path_checks"]()
        handle = self.accept_planning()
        handle.cancel_goal_async.assert_called_once()
        self.assert_no_navigation(run_id, "FAILED", "timed out after 5.0s")
        handle.get_result_async.assert_not_called()
        self.assertFalse(self.namespace["_goal_handles"])

    def test_late_success_is_timed_out_even_before_expiration_tick(self):
        run_id = self.start_check()
        handle = self.accept_planning()
        self.clock.monotonic.return_value = 105.0
        self.complete_planning(handle)
        self.assert_no_navigation(run_id, "FAILED", "timed out after 5.0s")

    def test_cancel_before_dispatch_does_not_plan_or_drive(self):
        accepted = self.namespace["_navigate_impl"](pose_message())
        run_id = accepted["run_id"]
        self.assertTrue(self.namespace["_cancel_impl"](run_id)["accepted"])
        self.dispatch_next()
        self.planner.send_goal_async.assert_not_called()
        self.assert_no_navigation(run_id, "CANCELED", "before dispatch")

    def test_cancel_during_planner_acceptance_never_starts_navigation(self):
        run_id = self.start_check()
        self.assertTrue(self.namespace["_cancel_impl"](run_id)["accepted"])
        handle = self.accept_planning()
        handle.cancel_goal_async.assert_called_once()
        self.complete_planning(handle)
        self.assert_no_navigation(run_id, "CANCELED", "during path precheck")

    def test_cancel_after_planner_acceptance_targets_only_that_run(self):
        run_id = self.start_check()
        handle = self.accept_planning()
        other_response = CallbackFuture()
        self.planner.send_goal_async.return_value = other_response
        other_run_id = self.start_check()
        other_handle = Mock(accepted=True)
        other_handle.get_result_async.return_value = CallbackFuture()
        other_response.set_result(other_handle)
        self.assertTrue(self.namespace["_cancel_impl"](run_id)["accepted"])
        handle.cancel_goal_async.assert_called_once()
        other_handle.cancel_goal_async.assert_not_called()
        self.complete_planning(handle)
        self.assert_no_navigation(run_id, "CANCELED", "during path precheck")
        self.complete_planning(other_handle)
        self.dispatch_next()
        self.navigator.send_goal_async.assert_called_once()
        self.assertEqual(self.namespace["_status_impl"](other_run_id)["state"], "RUNNING")

    def test_failed_or_rejected_planner_cancel_cannot_release_late_success(self):
        for failure in ("exception", "rejected", "response_exception"):
            with self.subTest(failure=failure):
                self.setUp()
                run_id = self.start_check()
                handle = self.accept_planning()
                if failure == "exception":
                    handle.cancel_goal_async.side_effect = RuntimeError("cancel disconnected")
                canceled = self.namespace["_cancel_impl"](run_id)
                self.assertEqual(canceled["accepted"], failure != "exception")
                if failure == "rejected":
                    handle.cancel_goal_async.return_value.set_result(SimpleNamespace(goals_canceling=[]))
                elif failure == "response_exception":
                    handle.cancel_goal_async.return_value.set_exception(RuntimeError("cancel disconnected"))
                self.complete_planning(handle)
                self.assert_no_navigation(run_id, "CANCELED", "during path precheck")

    def test_cancel_after_precheck_before_navigation_dispatch_stays_canceled(self):
        run_id = self.start_check()
        handle = self.accept_planning()
        self.complete_planning(handle)
        self.assertTrue(self.namespace["_cancel_impl"](run_id)["accepted"])
        self.dispatch_next()
        self.assert_no_navigation(run_id, "CANCELED", "before dispatch")

    def test_planner_cancel_wait_is_bounded_by_precheck_deadline(self):
        run_id = self.start_check()
        handle = self.accept_planning()
        self.namespace["_cancel_impl"](run_id)
        self.clock.monotonic.return_value = 105.0
        self.namespace["_expire_path_checks"]()
        self.assert_no_navigation(run_id, "CANCELED", "timed out after 5.0s")
        self.complete_planning(handle)
        self.assert_no_navigation(run_id, "CANCELED", "timed out after 5.0s")

    def test_navigation_acceptance_still_forwards_a_queued_cancel(self):
        run_id = self.start_check()
        handle = self.accept_planning()
        self.complete_planning(handle)
        self.dispatch_next()
        self.assertTrue(self.namespace["_cancel_impl"](run_id)["accepted"])
        navigation_handle = Mock(accepted=True)
        navigation_handle.get_result_async.return_value = CallbackFuture()
        navigation_handle.cancel_goal_async.return_value = CallbackFuture()
        self.navigation_response.set_result(navigation_handle)
        navigation_handle.cancel_goal_async.assert_called_once()
        self.assertIs(self.namespace["_goal_handles"][run_id], navigation_handle)


class RuntimeIntegrationTest(unittest.TestCase):
    def test_guard_depth_freshness_is_enabled_only_for_a_started_adapter(self):
        source = ast.parse((ROOT / "nav2_wrapper" / "atlas_bridge.py").read_text())
        function = next(
            node for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "_spawn_velocity_guard"
        )
        namespace = {
            "os": os, "sys": sys, "math": math,
            "subprocess": Mock(), "threading": Mock(),
            "resolve_velocity_output_topic": resolve_velocity_output_topic,
            "_speed_settings": speed_settings({"dynamic_speed": {
                "max_linear_speed_mps": 0.3, "default_percentage": 100,
            }}),
            "_GUARD_SPEED_LIMIT_TOPIC": "/test/speed_limit",
            "_DEPTH_CLOUD_TOPIC": "/test/depth_points",
            "_pkg_root": ROOT, "_pump_output": Mock(),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<guard-spawn>", "exec"), namespace)
        for enabled, timeout in ((False, None), (True, None), (True, 2.5)):
            with self.subTest(enabled=enabled, timeout=timeout):
                namespace["_depth_cloud_proc"] = Mock() if enabled else None
                namespace["subprocess"].Popen.reset_mock()
                config = {} if timeout is None else {"depth_obstacles": {"source_timeout_s": timeout}}
                with patch.dict(os.environ, {"ROBONIX_GUARD_DEPTH_TOPIC": "/inherited/depth"}):
                    namespace["_spawn_velocity_guard"](config)
                environment = namespace["subprocess"].Popen.call_args.kwargs["env"]
                self.assertEqual(
                    environment["ROBONIX_GUARD_DEPTH_TOPIC"],
                    "/test/depth_points" if enabled else "",
                )
                if enabled:
                    self.assertEqual(float(environment["ROBONIX_GUARD_DEPTH_TIMEOUT_S"]), timeout or 1.0)
        for timeout in (0, -1, math.nan, math.inf, -math.inf, "bad"):
            with self.subTest(timeout=timeout):
                namespace["_depth_cloud_proc"] = Mock()
                namespace["subprocess"].Popen.reset_mock()
                with self.assertRaises(ValueError):
                    namespace["_spawn_velocity_guard"]({"depth_obstacles": {"source_timeout_s": timeout}})
                namespace["subprocess"].Popen.assert_not_called()

    def test_docker_start_mounts_manifest_directory_read_only(self):
        bash_major = int(
            subprocess.run(
                ["bash", "-c", "printf %s ${BASH_VERSINFO[0]}"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        if bash_major < 4:
            self.skipTest("provider Docker wrapper requires Bash 4 or newer")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            deploy = root / "robot deploy"
            fake_bin = root / "bin"
            runtime_proto = root / "runtime-proto"
            package.mkdir()
            deploy.mkdir()
            fake_bin.mkdir()
            runtime_proto.mkdir()
            (runtime_proto / "atlas.proto").write_text(
                'syntax = "proto3";\n', encoding="utf-8"
            )
            proto_staging = package / "rbnx-build" / "proto-staging"
            proto_staging.mkdir(parents=True)
            (proto_staging / "navigation.proto").write_text(
                'syntax = "proto3";\n', encoding="utf-8"
            )
            docker_args = root / "docker.args"
            docker = fake_bin / "docker"
            docker.write_text(
                '#!/usr/bin/env bash\n'
                'if [[ "${1:-}" == run ]]; then\n'
                '  if [[ " $* " == *" --network none "* ]]; then\n'
                '    for arg in "$@"; do\n'
                '      if [[ "$arg" == *:/proto-gen ]]; then\n'
                '        output="${arg%:/proto-gen}"\n'
                '        touch "$output/atlas_pb2.py"\n'
                '        touch "$output/navigation_pb2.py"\n'
                '        touch "$output/robonix_contracts_pb2_grpc.py"\n'
                '      fi\n'
                '    done\n'
                '    exit 0\n'
                '  fi\n'
                '  printf "%s\\n" "$@" > "$DOCKER_ARGS_FILE"\n'
                "fi\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            rbnx = fake_bin / "rbnx"
            rbnx.write_text(
                '#!/usr/bin/env bash\n'
                'if [[ "${1:-}" == path && "${2:-}" == runtime-proto ]]; then\n'
                '  echo "$RUNTIME_PROTO_DIR"\n'
                'else\n'
                '  echo /tmp/robonix-api\n'
                'fi\n',
                encoding="utf-8",
            )
            rbnx.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:/usr/bin:/bin",
                    "DOCKER_ARGS_FILE": str(docker_args),
                    "RUNTIME_PROTO_DIR": str(runtime_proto),
                    "RBNX_PACKAGE_ROOT": str(package),
                    "RBNX_INVOCATION_CWD": str(deploy),
                    "ROBONIX_NAV2_FORCE": "docker",
                }
            )
            subprocess.run(
                ["bash", str(ROOT / "scripts" / "start.sh")],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            args = docker_args.read_text(encoding="utf-8").splitlines()
            self.assertIn(f"RBNX_INVOCATION_CWD={deploy}", args)
            self.assertIn(f"{deploy}:{deploy}:ro", args)
            self.assertIn(
                f"{package}/rbnx-build/codegen/nav2_proto_gen:"
                "/nav2/rbnx-build/codegen/proto_gen:ro",
                args,
            )

    def test_nav_consumes_provider_pinned_canonical_odom(self):
        source = (ROOT / "nav2_wrapper" / "atlas_bridge.py").read_text()
        self.assertIn('"odom",  "robonix/primitive/chassis/odom"', source)
        self.assertIn('providers = dict(cfg.get("provider_ids", {}) or {})', source)
        self.assertIn("provider_id=provider_id", source)
        self.assertNotIn('("odom",  "robonix/service/map/odom"', source)

    def test_pointcloud_projection_is_explicit(self):
        source = (ROOT / "nav2_wrapper" / "atlas_bridge.py").read_text()
        self.assertIn("scan_projection_config(cfg)", source)
        self.assertIn('"rtabmap_util", "lidar_deskewing"', source)
        self.assertIn('"scan:=/scanner/scan_raw"', source)
        self.assertIn('"-m", "nav2_wrapper.scan_filter"', source)

    def test_final_velocity_guard_owns_cmd_vel(self):
        source = (ROOT / "nav2_wrapper" / "atlas_bridge.py").read_text()
        guard = (ROOT / "nav2_wrapper" / "velocity_guard.py").read_text()
        self.assertIn("('cmd_vel', 'cmd_vel_guard_input')", source)
        # The smoother feeds the guard, through the Collision Monitor when
        # the deployment configures one.
        self.assertIn("('cmd_vel_smoothed', '{smoothed}')", source)
        self.assertIn('smoothed = _COLLISION_MONITOR_INPUT if collision_monitor else "cmd_vel_guard_input"', source)
        self.assertIn('ros["cmd_vel_out_topic"] = "cmd_vel_guard_input"', source)
        self.assertIn('"-m", "nav2_wrapper.velocity_guard"', source)
        self.assertIn('output_topic = resolve_velocity_output_topic(cfg)', source)
        self.assertIn('"ROBONIX_VELOCITY_OUTPUT_TOPIC": output_topic', source)
        self.assertIn('output_topic = resolve_velocity_output_topic({})', guard)
        self.assertIn('create_publisher(Twist, output_topic, 10)', guard)
        self.assertNotIn('create_publisher(Twist, "/cmd_vel"', guard)
        start = (ROOT / "scripts" / "start.sh").read_text()
        self.assertIn('"${ROBONIX_VELOCITY_OUTPUT_TOPIC+x}" == "x"', start)
        self.assertIn('"${VELOCITY_OUTPUT_ARGS[@]}"', start)

    def test_dynamic_speed_uses_nav2_live_speed_limit_without_replacing_goal(self):
        source = (ROOT / "nav2_wrapper" / "atlas_bridge.py").read_text()
        guard = (ROOT / "nav2_wrapper" / "velocity_guard.py").read_text()
        manifests = [
            (ROOT / name).read_text()
            for name in (
                "package_manifest.yaml",
                "package_manifest.jetson-docker.yaml",
                "package_manifest.jetson-native.yaml",
            )
        ]

        self.assertIn("from nav2_msgs.msg import SpeedLimit", source)
        self.assertIn(
            "node.create_publisher(\n            _SpeedLimit, settings.topic, 10",
            source,
        )
        self.assertIn(
            "_GUARD_SPEED_LIMIT_TOPIC = "
            '"/robonix/navigation/speed_limit_guard"',
            source,
        )
        self.assertIn("guard_publisher.publish(message)", source)
        self.assertIn("ROBONIX_NAV_GUARD_SPEED_LIMIT_TOPIC", guard)
        self.assertIn("def _speed_channels_available()", source)
        self.assertIn("node.create_timer(1.0, _refresh_speed_subscriber)", source)
        self.assertNotIn("node.create_timer(1.0, _republish_speed_limit)", source)
        self.assertIn("def _adjust_speed_impl(", source)
        self.assertIn("def _set_speed_limit_impl(", source)
        self.assertIn("def _get_speed_limit_impl()", source)
        self.assertIn("_publish_speed_limit(decision.percentage)", source)
        speed_source = source[source.index("def _adjust_speed_impl"):]
        self.assertNotIn("_nav_queue.put", speed_source)
        self.assertIn("ROBONIX_NAV_MAX_LINEAR_SPEED_MPS", source)
        self.assertIn("bounded_linear_velocity(", guard)
        for manifest in manifests:
            self.assertIn("robonix/service/navigation/adjust_speed", manifest)
            self.assertIn(
                "robonix/service/navigation/set_speed_limit",
                manifest,
            )
            self.assertIn(
                "robonix/service/navigation/get_speed_limit",
                manifest,
            )

    def test_controller_output_mux_route_is_opt_in_and_fully_forwarded(self):
        source = (ROOT / "nav2_wrapper" / "atlas_bridge.py").read_text()
        config = (ROOT / "nav2_wrapper" / "configuration.py").read_text()
        spec = (ROOT / "config.spec").read_text()
        self.assertIn(
            "resolve_controller_velocity_output_topic(cfg)", source
        )
        self.assertIn(
            "controller_velocity_output_topic="
            "controller_velocity_output_topic",
            source,
        )
        self.assertIn(
            'cfg["controller_velocity_output_topic"]', config
        )
        self.assertIn("controller_velocity_output_topic:", spec)
        self.assertIn("old_controller", source)
        self.assertIn("('cmd_vel', 'cmd_vel_nav')", source)

    def test_invalid_velocity_topic_is_rejected_before_dependency_discovery(self):
        source = (ROOT / "nav2_wrapper" / "atlas_bridge.py").read_text()
        validation = source.index("resolve_velocity_output_topic(cfg)", source.index("def init"))
        discovery = source.index("_build_remap_args(cfg)", source.index("def init"))
        self.assertLess(validation, discovery)
        self.assertIn('return Err(f"invalid velocity_output_topic: {error}")', source)

    def test_failed_init_cleans_up_nav_children(self):
        source = (ROOT / "nav2_wrapper" / "atlas_bridge.py").read_text()
        self.assertIn(
            'except Exception as e:  # noqa: BLE001\n        _kill_nav2()\n'
            '        return Err(f"spawn nav2 failed: {e}")',
            source,
        )

    def test_cancel_is_latched_before_action_handle_exists(self):
        source = (ROOT / "nav2_wrapper" / "atlas_bridge.py").read_text()
        self.assertIn('state["cancel_requested"] = True', source)
        self.assertIn('state["state"] = "CANCELED"', source)
        self.assertIn("cancel queued until goal acceptance", source)
        self.assertIn("if cancel_requested:\n        _issue_cancel(gh, gid)", source)
        self.assertIn("def _cancel_response_cb", source)

    def test_docker_runtime_supports_interface_bound_cyclonedds(self):
        dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        start = (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")
        self.assertIn("ros-humble-rmw-cyclonedds-cpp", dockerfile)
        self.assertIn('-e CYCLONEDDS_URI="${CYCLONEDDS_URI:-}"', start)
        self.assertIn(
            '-e ROBONIX_PROVIDER_BIND_HOST="${ROBONIX_PROVIDER_BIND_HOST:-0.0.0.0}"',
            start,
        )
        self.assertIn(
            '-e ROBONIX_ADVERTISE_HOST="${ROBONIX_ADVERTISE_HOST:-}"', start
        )

    def test_docker_codegen_matches_and_validates_runtime_protobuf(self):
        dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        start = (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")
        entrypoint = (ROOT / "docker" / "entrypoint.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("'setuptools>=77,<80'", dockerfile)
        self.assertIn("grpcio-tools>=1.60.0", dockerfile)
        self.assertIn('runtime_proto="$(rbnx path runtime-proto)"', start)
        self.assertIn("--network none", start)
        self.assertIn("python3 -m grpc_tools.protoc", start)
        self.assertIn("[importlib.import_module(name) for name in modules]", start)
        self.assertIn("codegen/nav2_proto_gen", start)
        self.assertIn(
            "/nav2/rbnx-build/codegen/proto_gen:ro",
            start,
        )
        self.assertIn("missing runtime-compatible protobuf stubs", entrypoint)

    def test_docker_runtime_stays_on_mcp_v1(self):
        """Keep robonix-api's mcp.server.fastmcp import available."""
        dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("'mcp>=1.28.1,<2'", dockerfile)
        self.assertIn("'fastmcp>=3,<4'", dockerfile)

    def test_codegen_is_mcp_only_for_every_deployment_target(self):
        build = (ROOT / "scripts" / "build.sh").read_text(encoding="utf-8")
        native_start = (ROOT / "scripts" / "start_native.sh").read_text(
            encoding="utf-8"
        )
        docker_entrypoint = (ROOT / "docker" / "entrypoint.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("FLAGS=(--mcp)", build)
        self.assertNotIn("--ros2", build)
        self.assertNotIn("ROS2_IDL", build)
        for runtime_script in (native_start, docker_entrypoint):
            self.assertNotIn("codegen/ros2_idl", runtime_script)

    def test_config_directory_contains_only_the_neutral_template(self):
        names = sorted(path.name for path in (ROOT / "config").glob("*"))
        self.assertEqual(names, ["nav2_params.example.yml"])

    def test_frozen_legacy_assets_are_internal_compatibility_data(self):
        names = sorted(
            path.name for path in (ROOT / "nav2_wrapper" / "legacy_config").glob("*")
        )
        self.assertEqual(
            names,
            [
                "nav2_params.yml",
                "nav2_params_ranger_mini_v3.yml",
                "nav2_params_sim.yml",
                "nav2_params_slam.yml",
                "ranger_mini_v3_navigate.xml",
            ],
        )
        spec = (ROOT / "config.spec").read_text()
        self.assertIn("\nspecVersion: 1\n", spec)
        self.assertIn("\n  params_file:\n", spec)
        readme = (ROOT / "README.md").read_text()
        self.assertNotIn("params_profile:", readme)


if __name__ == "__main__":
    unittest.main()
