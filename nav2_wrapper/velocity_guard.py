"""Final velocity guard and passive per-NavigateToPose trajectory recorder."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from nav2_msgs.msg import SpeedLimit
from nav_msgs.msg import Odometry, Path as NavPath
from sensor_msgs.msg import LaserScan, PointCloud2
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_action_status_default, qos_profile_sensor_data

from nav2_wrapper.cmd_vel_summary import CmdVelSummary
from nav2_wrapper.configuration import resolve_velocity_output_topic, validate_absolute_ros_topic
from nav2_wrapper.rotation_guard_core import GuardLimits, RotationGuard, normalize_uuid_octets
from nav2_wrapper.velocity_limit import bounded_linear_velocity


_STOP_FRAME_LIMIT = 3
_RETIRED_UUID_LIMIT = 128
_LIVE_CANDIDATE_LIMIT = 16
_CANCEL_RETRY_S = 0.5
_CANCEL_REPLY_TIMEOUT_S = 2.0


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _uuid_text(raw) -> str:
    return bytes(raw).hex()


class VelocityGuardNode(Node):
    def __init__(self):
        # Validate before constructing the ROS node or creating any endpoint.
        # An invalid/empty/relative output must fail closed with no publisher.
        output_topic = resolve_velocity_output_topic({})
        max_linear_speed_mps = float(
            os.environ["ROBONIX_NAV_MAX_LINEAR_SPEED_MPS"]
        )
        default_percentage = float(
            os.environ["ROBONIX_NAV_DEFAULT_SPEED_PERCENTAGE"]
        )
        speed_limit_topic = os.environ[
            "ROBONIX_NAV_GUARD_SPEED_LIMIT_TOPIC"
        ]
        depth_topic = os.getenv("ROBONIX_GUARD_DEPTH_TOPIC", "")
        depth_timeout_s = 1.0
        if depth_topic:
            validate_absolute_ros_topic(depth_topic, "ROBONIX_GUARD_DEPTH_TOPIC")
            depth_timeout_s = float(os.getenv("ROBONIX_GUARD_DEPTH_TIMEOUT_S", "1.0"))
            if not math.isfinite(depth_timeout_s) or depth_timeout_s <= 0.0:
                raise ValueError("ROBONIX_GUARD_DEPTH_TIMEOUT_S must be finite and positive")
        if (
            not math.isfinite(max_linear_speed_mps)
            or max_linear_speed_mps <= 0.0
        ):
            raise ValueError(
                "ROBONIX_NAV_MAX_LINEAR_SPEED_MPS must be finite and positive"
            )
        if not 0.0 < default_percentage <= 100.0:
            raise ValueError(
                "ROBONIX_NAV_DEFAULT_SPEED_PERCENTAGE must be in (0, 100]"
            )
        super().__init__("robonix_velocity_guard")
        limits = GuardLimits(
            terminal_xy_m=float(os.getenv("ROBONIX_GUARD_TERMINAL_XY_M", "0.45")),
            terminal_timeout_s=float(os.getenv("ROBONIX_GUARD_TERMINAL_TIMEOUT_S", "15.0")),
            terminal_no_progress_s=float(os.getenv("ROBONIX_GUARD_NO_PROGRESS_S", "3.0")),
            global_spin_timeout_s=float(os.getenv("ROBONIX_GUARD_GLOBAL_TIMEOUT_S", "25.0")),
            global_spin_limit_rad=float(os.getenv("ROBONIX_GUARD_GLOBAL_ROTATION_RAD", "6.783185307")),
        )
        self.guard = RotationGuard(limits)
        self._lock = threading.Lock()
        self._active_uuid = ""
        self._active_uuid_raw = None
        self._active_stamp = 0
        self._acceptance_fence = 0
        self._retired_uuids = set()
        self._live_candidates = {}
        self._fence_closed = False
        self._robot_pose = None
        self._goal_pose = None
        self._last_cmd = (0.0, 0.0)
        self._cmd_summary = CmdVelSummary()
        self._cancel_sent = False
        self._cancel_future = None
        self._cancel_requested_at = None
        self._cancel_retry_at = 0.0
        self._stop_frames_remaining = None
        self._depth_topic = depth_topic
        self._depth_timeout_s = depth_timeout_s
        self._depth_stamp = None
        self._depth_received_at = None
        self._trace = None
        self._trace_dir = Path(os.getenv("ROBONIX_NAV_TRACE_DIR", "/tmp/robonix-nav-traces"))
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        self._scan_log = (self._trace_dir / "scan-anomalies.jsonl").open("a", encoding="utf-8")
        self._front_min_history = deque(maxlen=20)
        self._last_scan_summary_at = 0.0
        self._max_linear_speed_mps = max_linear_speed_mps
        self._speed_percentage = default_percentage
        self._effective_linear_speed_mps = (
            max_linear_speed_mps * default_percentage / 100.0
        )

        self._pub = self.create_publisher(Twist, output_topic, 10)
        self.create_subscription(Twist, "/cmd_vel_guard_input", self._on_cmd, 20)
        self.create_subscription(
            SpeedLimit, speed_limit_topic, self._on_speed_limit, 10
        )
        self.create_subscription(Odometry, "/odom", self._on_odom, 50)
        self.create_subscription(NavPath, "/plan", self._on_plan, 10)
        self.create_subscription(LaserScan, "/scanner/scan", self._on_scan, 20)
        if depth_topic:
            self.create_subscription(
                PointCloud2, depth_topic, self._on_depth, qos_profile_sensor_data
            )
        self.create_subscription(
            GoalStatusArray, "/navigate_to_pose/_action/status", self._on_status,
            qos_profile_action_status_default,
        )
        self.create_subscription(
            NavigateToPose.Impl.FeedbackMessage,
            "/navigate_to_pose/_action/feedback",
            self._on_feedback,
            20,
        )
        self._cancel = self.create_client(CancelGoal, "/navigate_to_pose/_action/cancel_goal")
        self.create_timer(0.05, self._publish_latched_zero)
        self.get_logger().info(
            f"guard active; output={output_topic}; "
            f"max_linear={max_linear_speed_mps:g}m/s; "
            f"default={default_percentage:g}%; traces={self._trace_dir}"
        )

    def _on_speed_limit(self, msg: SpeedLimit) -> None:
        """Track the live Nav2 limit and retain the configured hard maximum."""
        value = float(msg.speed_limit)
        if not math.isfinite(value) or value < 0.0:
            self.get_logger().error(f"ignoring invalid speed limit: {value}")
            return
        with self._lock:
            if value == 0.0:
                percentage = 100.0
                linear_limit_mps = self._max_linear_speed_mps
            elif msg.percentage:
                percentage = min(100.0, max(0.0, value))
                linear_limit_mps = (
                    self._max_linear_speed_mps * percentage / 100.0
                )
            else:
                linear_limit_mps = min(self._max_linear_speed_mps, value)
                percentage = (
                    linear_limit_mps / self._max_linear_speed_mps * 100.0
                )
            self._speed_percentage = percentage
            self._effective_linear_speed_mps = linear_limit_mps
            self._event(
                "speed_limit",
                percentage=percentage,
                linear_limit_mps=linear_limit_mps,
                source_percentage=bool(msg.percentage),
            )

    def _event(self, kind: str, **values) -> None:
        if self._trace is None:
            return
        record = {"t_wall": time.time(), "event": kind, **values}
        self._trace.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._trace.flush()

    def _begin_goal(self, goal_uuid: str, raw_uuid, acceptance_stamp: int) -> None:
        if (
            self._active_uuid or acceptance_stamp <= 0
            or self._live_candidates.get(goal_uuid) != acceptance_stamp
            or self._fence_closed or goal_uuid in self._retired_uuids
        ):
            return
        raw_uuid = normalize_uuid_octets(raw_uuid)
        if not any(raw_uuid):
            return
        self._active_uuid = goal_uuid
        self._active_uuid_raw = raw_uuid
        self._active_stamp = acceptance_stamp
        self._cancel_sent = False
        self._cancel_retry_at = 0.0
        self._stop_frames_remaining = None
        self._robot_pose = None
        self._goal_pose = None
        self._last_cmd = (0.0, 0.0)
        self.guard.begin_goal(goal_uuid)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self._trace = (self._trace_dir / f"{stamp}-{goal_uuid}.jsonl").open("a", encoding="utf-8")
        self._event("goal_active", uuid=goal_uuid)

    def _remember_retired(self, goal_uuid: str, acceptance_stamp: int) -> None:
        if acceptance_stamp != self._acceptance_fence or goal_uuid in self._retired_uuids:
            return
        if len(self._retired_uuids) >= _RETIRED_UUID_LIMIT:
            if not self._fence_closed:
                self.get_logger().error("goal acceptance timestamp exhausted; waiting for a newer stamp")
            self._fence_closed = True
            self._live_candidates.clear()
        else:
            self._retired_uuids.add(goal_uuid)

    def _clear_cancel_request(self) -> None:
        future = self._cancel_future
        self._cancel_future = None
        self._cancel_requested_at = None
        if future is not None:
            self._cancel.remove_pending_request(future)

    def _retire_goal(self, reason: str) -> None:
        self._live_candidates.pop(self._active_uuid, None)
        self._remember_retired(self._active_uuid, self._active_stamp)
        self._event("goal_inactive", uuid=self._active_uuid, reason=reason)
        if self._trace is not None:
            self._event("trace_closed", reason=reason)
            self._trace.close()
            self._trace = None
        self._active_uuid = ""
        self._active_uuid_raw = None
        self._active_stamp = 0
        self._cancel_sent = False
        self._clear_cancel_request()
        self._cancel_retry_at = 0.0
        self._stop_frames_remaining = None
        self._robot_pose = None
        self._goal_pose = None
        self._last_cmd = (0.0, 0.0)
        self.guard.clear_if_idle()

    def _on_status(self, msg: GoalStatusArray) -> None:
        with self._lock:
            self._event(
                "status",
                values=[{"uuid": _uuid_text(s.goal_info.goal_id.uuid), "status": int(s.status)}
                        for s in msg.status_list],
            )
            live_states = (
                GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING,
                GoalStatus.STATUS_CANCELING,
            )
            terminal_states = (
                GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED,
                GoalStatus.STATUS_ABORTED,
            )
            entries = []
            for item in msg.status_list:
                stamp = item.goal_info.stamp.sec * 1_000_000_000 + item.goal_info.stamp.nanosec
                if stamp > 0 and any(item.goal_info.goal_id.uuid):
                    entries.append((_uuid_text(item.goal_info.goal_id.uuid), stamp, item))
            newest_stamp = max((stamp for _, stamp, _ in entries), default=0)
            if self._active_uuid and msg.status_list and newest_stamp < self._active_stamp:
                return
            terminal = {
                uuid for uuid, stamp, item in entries if item.status in terminal_states
            }
            active = {
                uuid: (stamp, item) for uuid, stamp, item in entries
                if item.status in live_states and uuid not in terminal
            }
            eligible = {
                uuid: stamp for uuid, (stamp, item) in active.items()
                if self._live_candidates.get(uuid) == stamp
                or stamp > self._acceptance_fence
                or (
                    stamp == self._acceptance_fence and not self._fence_closed
                    and uuid not in self._retired_uuids
                )
            }
            if newest_stamp > self._acceptance_fence:
                self._acceptance_fence = newest_stamp
                self._retired_uuids.clear()
                self._fence_closed = False
            for uuid, stamp in list(self._live_candidates.items()):
                if uuid not in eligible:
                    self._remember_retired(uuid, stamp)
            for uuid, stamp, item in entries:
                if item.status in terminal_states:
                    self._remember_retired(uuid, stamp)
            if len(eligible) > _LIVE_CANDIDATE_LIMIT:
                if not self._fence_closed:
                    self.get_logger().error("too many live goal candidates; waiting for a newer stamp")
                self._fence_closed = True
            self._live_candidates = {} if self._fence_closed else eligible
            if self._active_uuid:
                if self._active_uuid in terminal:
                    self._retire_goal("terminal_status")
                elif any(
                    uuid == self._active_uuid and stamp == self._active_stamp
                    and item.status in live_states for uuid, stamp, item in entries
                ):
                    return
                else:
                    self._retire_goal("missing_status" if msg.status_list else "empty_status")
            if len(active) == 1:
                uuid, (stamp, item) = next(iter(active.items()))
                self._begin_goal(uuid, item.goal_info.goal_id.uuid, stamp)

    def _on_feedback(self, msg) -> None:
        with self._lock:
            uuid = _uuid_text(msg.goal_id.uuid)
            if not self._active_uuid or uuid != self._active_uuid:
                return
            pose = msg.feedback.current_pose.pose
            self._robot_pose = (pose.position.x, pose.position.y, _yaw(pose.orientation))
            self._event(
                "feedback", pose=self._robot_pose,
                distance_remaining=float(msg.feedback.distance_remaining),
                recoveries=int(msg.feedback.number_of_recoveries),
            )

    def _on_plan(self, msg: NavPath) -> None:
        if not msg.poses:
            return
        with self._lock:
            if not self._active_uuid:
                return
            pose = msg.poses[-1].pose
            self._goal_pose = (pose.position.x, pose.position.y, _yaw(pose.orientation))
            self._event("plan", frame=msg.header.frame_id, points=len(msg.poses), goal=self._goal_pose)

    def _on_odom(self, msg: Odometry) -> None:
        with self._lock:
            pose = msg.pose.pose
            yaw = _yaw(pose.orientation)
            self.guard.observe_odom(yaw)
            self._event(
                "odom", pose=(pose.position.x, pose.position.y, yaw),
                twist=(msg.twist.twist.linear.x, msg.twist.twist.angular.z),
            )

    def _on_scan(self, msg: LaserScan) -> None:
        """Persist sudden/isolated close returns even when no goal is active."""
        front = []
        for index, value in enumerate(msg.ranges):
            angle = msg.angle_min + index * msg.angle_increment
            angle = math.atan2(math.sin(angle), math.cos(angle))
            if (
                abs(angle) <= math.radians(60.0)
                and math.isfinite(value)
                and msg.range_min <= value <= msg.range_max
            ):
                front.append((index, angle, float(value)))
        if not front:
            return
        ordered = sorted(value for _, _, value in front)
        front_min = ordered[0]
        front_p05 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.05))]
        baseline = None
        if self._front_min_history:
            history = sorted(self._front_min_history)
            baseline = history[len(history) // 2]
        sudden = baseline is not None and baseline - front_min >= 0.40 and front_min <= 2.0

        isolated = []
        by_index = {index: value for index, _, value in front}
        for index, angle, value in front:
            if value > 2.0:
                continue
            neighbors = [by_index.get(index + offset) for offset in (-2, -1, 1, 2)]
            close_neighbors = sum(
                other is not None and abs(other - value) <= 0.20 for other in neighbors
            )
            if close_neighbors <= 1:
                isolated.append((round(angle, 5), round(value, 4)))
        now = time.monotonic()
        periodic = now - self._last_scan_summary_at >= 1.0
        if sudden or periodic:
            record = {
                "t_wall": time.time(),
                "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
                "frame": msg.header.frame_id,
                "front_min": round(front_min, 4),
                "front_p05": round(front_p05, 4),
                "baseline_min": None if baseline is None else round(baseline, 4),
                "sudden": sudden,
                "isolated": isolated[:40],
                "front_under_2m": [
                    (round(angle, 5), round(value, 4))
                    for _, angle, value in front if value <= 2.0
                ] if sudden else [],
                "range_min": float(msg.range_min),
                "range_max": float(msg.range_max),
            }
            self._scan_log.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._scan_log.flush()
            self._last_scan_summary_at = now
        self._front_min_history.append(front_min)

    def _on_depth(self, msg: PointCloud2) -> None:
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        with self._lock:
            if stamp > 0 and (self._depth_stamp is None or stamp > self._depth_stamp):
                self._depth_stamp = stamp
                self._depth_received_at = time.monotonic()

    def _check_depth(self) -> str:
        if not self._active_uuid:
            return ""
        if self._depth_topic and not self.guard.latched_reason:
            if self._depth_received_at is None:
                reason = "depth input has not received a sample"
            elif time.monotonic() - self._depth_received_at >= self._depth_timeout_s:
                reason = f"depth input timeout after {self._depth_timeout_s:g}s"
            else:
                return ""
            self.guard.latched_reason = reason
            self.get_logger().error(f"DEPTH GUARD TRIPPED uuid={self._active_uuid}: {reason}")
            self._event("depth_timeout", reason=reason)
        return self.guard.latched_reason

    def _on_cmd(self, msg: Twist) -> None:
        with self._lock:
            if not self._active_uuid or self.guard.latched_reason:
                return
            x, y, limited = bounded_linear_velocity(
                float(msg.linear.x),
                float(msg.linear.y),
                self._effective_linear_speed_mps,
            )
            output = msg
            if limited:
                output = Twist()
                output.linear.x = x
                output.linear.y = y
                output.linear.z = msg.linear.z
                output.angular.x = msg.angular.x
                output.angular.y = msg.angular.y
                output.angular.z = msg.angular.z
            self._last_cmd = (math.hypot(x, y), float(output.angular.z))
            now_s = self.get_clock().now().nanoseconds / 1e9
            reason = self._check_depth() or self.guard.evaluate(
                now_s,
                self._last_cmd[0], self._last_cmd[1], self._robot_pose, self._goal_pose,
            )
            self._event(
                "cmd", linear=self._last_cmd[0], angular=self._last_cmd[1],
                requested_linear=math.hypot(msg.linear.x, msg.linear.y),
                linear_limit_mps=self._effective_linear_speed_mps,
                linear_limited=limited,
                terminal_rotation=self.guard.terminal_rotation,
                continuous_rotation=self.guard.spin_rotation,
                latched=bool(reason),
            )
            line = self._cmd_summary.observe(
                now_s, self._last_cmd[0], self._last_cmd[1], limited, bool(reason))
            if line:
                self.get_logger().info(line)
            if reason:
                self._trip(reason)
            else:
                self._pub.publish(output)

    def _trip(self, reason: str) -> None:
        if not self._active_uuid:
            return
        if self._stop_frames_remaining is None:
            self._stop_frames_remaining = _STOP_FRAME_LIMIT
            self.get_logger().error(f"VELOCITY GUARD TRIPPED uuid={self._active_uuid}: {reason}")
            self._event("guard_trip", reason=reason)
        if self._stop_frames_remaining > 0:
            self._pub.publish(Twist())
            self._stop_frames_remaining -= 1
        self._retry_cancel()

    def _retry_cancel(self) -> None:
        now = time.monotonic()
        if self._cancel_future is not None and self._cancel_future.done():
            future = self._cancel_future
            self._clear_cancel_request()
            try:
                response = future.result()
                self._cancel_sent = response.return_code == 0 and any(
                    list(info.goal_id.uuid) == self._active_uuid_raw
                    for info in response.goals_canceling
                )
                if not self._cancel_sent:
                    raise RuntimeError(f"cancellation rejected (code={response.return_code})")
                self._event("cancel_accepted", uuid=self._active_uuid)
            except Exception as error:
                self._cancel_sent = False
                self._cancel_retry_at = now + _CANCEL_RETRY_S
                self.get_logger().error(f"guard cancel failed uuid={self._active_uuid}: {error}")
                self._event("cancel_failed", uuid=self._active_uuid, reason=str(error))
        elif self._cancel_future is not None:
            if (
                not self._cancel_future.cancelled()
                and now - self._cancel_requested_at < _CANCEL_REPLY_TIMEOUT_S
            ):
                return
            self._clear_cancel_request()
            self._cancel_sent = False
            self._cancel_retry_at = now + _CANCEL_RETRY_S
            self._event("cancel_timeout", uuid=self._active_uuid)
        if self._cancel_sent or now < self._cancel_retry_at:
            return
        if self._active_uuid_raw is None or not any(self._active_uuid_raw):
            return
        self._cancel_retry_at = now + _CANCEL_RETRY_S
        if not self._cancel.service_is_ready():
            return
        request = CancelGoal.Request()
        request.goal_info.goal_id.uuid = self._active_uuid_raw
        try:
            self._cancel_future = self._cancel.call_async(request)
        except Exception as error:
            self.get_logger().error(f"guard cancel failed uuid={self._active_uuid}: {error}")
            return
        self._cancel_requested_at = now
        self._cancel_sent = True
        self._event("cancel_requested", uuid=self._active_uuid)

    def _publish_latched_zero(self) -> None:
        with self._lock:
            if not self._active_uuid:
                return
            self._check_depth()
            if self.guard.latched_reason:
                self._trip(self.guard.latched_reason)


def main() -> None:
    rclpy.init()
    node = VelocityGuardNode()
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
