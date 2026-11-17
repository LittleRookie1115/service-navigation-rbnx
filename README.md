# service-navigation-rbnx

Robonix wrapper around a system-installed ROS 2 Nav2 stack. The service
discovers map, odometry, and lidar inputs through Atlas and exposes
`robonix/service/navigation/*` over gRPC and MCP.

## Deployment config

Navigation behavior belongs to the robot deployment, not this provider. Each
robot must carry a complete Nav2 YAML and reference it with `params_file`:

```yaml
service:
  - name: nav2
    url: https://github.com/syswonder/service-navigation-rbnx
    branch: main
    config:
      params_file: config/nav2_params.yaml
      provider_ids:
        map: mapping
        odom: chassis
        scan: lidar
      dynamic_speed:
        max_linear_speed_mps: 0.3
        default_percentage: 75
        step_percentage: 20
        min_percentage: 20
```

Relative paths are resolved from the directory containing
`robonix_manifest.yaml`. `config/nav2_params.example.yml` is a neutral example,
not a robot profile. Copy it into the deploy repository and set the robot's
frames, footprint, velocity and acceleration limits, costmap layers, goal
tolerances, topics, and planner/controller plugins there.

For a 3D lidar, bind `scan_cloud` and declare the adapter explicitly:

```yaml
      params_file: config/nav2_params.yaml
      provider_ids:
        map: mapping
        odom: chassis
        scan_cloud: lidar3d
      scan_projection:
        enabled: true
        target_frame: base_link
        min_height_m: 0.1
        max_height_m: 1.5
        range_max_m: 12.0
```

A 2D lidar sees one plane, so a table top or a shelf stays invisible until the
robot's body meets it. To add a depth camera's view, bind the camera and use
`__ROBONIX_DEPTH_CLOUD_TOPIC__` in the Nav2 YAML:

```yaml
      provider_ids:
        map: mapping
        odom: chassis
        scan: lidar
        depth: head_camera        # provides camera/depth and camera/intrinsics
      depth_obstacles:            # optional; these are the defaults
        stride: 4
        min_range_m: 0.2
        max_range_m: 4.0
        rate_hz: 10
        source_timeout_s: 1.0
```

The wrapper then turns the camera's depth image, with its own intrinsics, into
a PointCloud2 in the camera frame on that topic. Costmap layers such as
`spatio_temporal_voxel_layer` read it and choose the heights that count as
obstacles for the robot, from just above the floor to the top of its body.
Without the token the camera is not used.

Depth projection and Collision Monitor integration are upstream features.
The final velocity guard checks depth freshness and publishes chassis commands
only while it owns an active `NavigateToPose` UUID reported by action status.
Startup and idle depth gaps produce no chassis writes; idle commands and
feedback are ignored. The action status subscription uses the transient-local
action preset, so a late-joining guard receives the retained live or terminal
snapshot. A current UUID keeps ownership while the latest status still reports
it as nonterminal, even if a pending goal is also marked EXECUTING. Terminal,
empty, or owner-missing status retires it. Nonempty snapshots whose newest
acceptance predates the current owner's acceptance are ignored before changing
ownership or history. Initial snapshots with multiple live
UUIDs leave the guard silent until a unique eligible live goal is reported;
feedback cannot choose an owner.

Positive `goal_info.stamp` acceptance timestamps bound retirement history.
The guard remembers at most 128 retired UUIDs at the newest acceptance stamp;
older unknown UUIDs cannot acquire ownership. It also carries at most 16
eligible live candidates across newer stamps, so a newer failed pending goal
does not exclude an older still-live goal, including in a retained snapshot.
A disappeared or terminal candidate loses that eligibility. Equal-stamp goals
are allowed until retirement or candidate capacity overflows; overflow blocks
new acquisition until a newer acceptance stamp supplies an eligible goal.
An already-owned UUID still reported live keeps ownership. Zero or negative
acceptance stamps cannot acquire control. After resetting the action server's
source clock, restart the navigation provider to reset this acceptance fence.
Evicted retirement history cannot be revived by old status or feedback.

When depth is enabled, the watchdog requires samples with strictly increasing,
positive source timestamps. Missing samples or no newer sample within
`source_timeout_s` latch the active goal and request cancellation of that exact
UUID. The timeout must be finite and positive; it uses monotonic elapsed time,
not simulation time. Both incoming commands and the 50 ms timer check it, so
stopping does not depend on another velocity command. A depth or rotation
fault sends at most three zero frames, starting immediately, then stops all
chassis publication, including on incoming commands. Cancellation can retry
after the burst if the service is unavailable, sending fails, or the reply
rejects that UUID, with a 0.5-second retry interval and no further chassis
writes. Only one cancellation request can be pending; after two seconds
without a reply it is removed from the client before another attempt.
Retirement also removes any pending request and stops both the burst and
retries. A late completion cannot affect a new owner. This guard does not
acquire an idle chassis lock or interfere with another driver's idle commands.

Depth recovery does not release the faulted goal; a new UUID and fresh samples
are required. Timestamps are retained across goals, so replayed or out-of-order
samples cannot refresh the watchdog. Restart the navigation provider after
resetting the source clock.

This watchdog detects stream freshness, not obstacles inside the camera's
minimum range or invalid geometry in an otherwise updating stream. It does not
replace obstacle retention, collision checking, or physical stop verification.

A `collision_monitor` section in the same YAML starts Nav2's Collision Monitor
between the velocity smoother and the final velocity guard. Its zones stop or
slow the robot before anything its sources see, whatever the controller
planned; `__ROBONIX_SCAN_TOPIC__` and `__ROBONIX_DEPTH_CLOUD_TOPIC__` can be
its sources. The wrapper sets its input and output topics.

Optional `bt_xml_file` points to a deploy-owned BehaviorTree XML. Existing
`params_profile` deployments remain supported and emit a migration warning;
new deployments should not use that field. See `config.spec` for every
accepted instance field and default.

The final velocity guard publishes to `/cmd_vel` by default for compatibility.
Set `config.velocity_output_topic` to a fully-qualified non-motion sink such as
`/robonix/nomotion/cmd_vel` while integrating a physical robot. The
`ROBONIX_VELOCITY_OUTPUT_TOPIC` environment variable is the fallback when the
config field is absent; an explicit empty, relative, or malformed topic fails
startup before the guard creates any ROS endpoint.

Dynamic speed config uses Nav2-compatible SI semantics.
`max_linear_speed_mps` is the actual hard planar limit
`sqrt(vx^2 + vy^2)` in m/s after considering both the selected controller's
`max_speed_xy` and stricter per-axis limits. The final velocity guard
independently enforces this linear limit. `default_percentage` scales that
maximum, so the example starts at `0.225 m/s`. `step_percentage` is an additive
number of percentage points. Angular constraints such as DWB's
`max_vel_theta` stay in the deploy-owned Nav2 YAML; a controller may
proportionally adjust its internal kinematics when it processes a Nav2 speed
limit, but Robonix does not expose a separate angular-speed policy here.

`adjust_speed` handles `faster`, `slower`, and `normal`;
`set_speed_limit` applies an explicit percentage; and `get_speed_limit` reads
the current and configured state. By default a mutation belongs to the
selected active run and automatically restores the session limit when that run
terminates. `persist=true` deliberately changes the provider-session limit
across navigation runs. None of these operations restart Navigation, cancel a
goal, or resubmit it.

## Runtime

At `Driver(CMD_INIT)`, the wrapper:

1. resolves the selected Atlas providers;
2. resolves and materializes the deployment-owned Nav2 YAML;
3. starts an optional PointCloud2-to-LaserScan adapter, and the depth cloud
   when the Nav2 YAML uses it;
4. starts Nav2, and the Collision Monitor when the YAML configures one, and
   waits for the `navigate_to_pose` action server;
5. connects to Nav2's live `speed_limit` subscriber;
6. exposes navigate, status, cancel, and dynamic speed capabilities.

Missing required providers return `deferred`. Invalid config or a Nav2 startup
failure returns `error` and tears down every child process.

## Build and tests

Navigation generates only its Atlas MCP bindings on every deployment target.
It deliberately does not generate, build, or source a Robonix ROS 2 IDL
overlay: the provider talks to Nav2 through the ROS 2 interfaces supplied by
the selected Humble installation, while its own public capability transport is
gRPC/MCP. Jetson native builds source only the system ROS 2 installation and
the locally built terminal-controller plugin overlay.

```bash
bash scripts/build.sh
python3 -m unittest -v \
  test_configuration.py \
  test_runtime_integration.py \
  test_rotation_guard.py \
  test_scan_filter.py \
  test_speed_control.py \
  test_velocity_limit.py
```

The MCP envelope cases in `test_runtime_integration.py` require FastMCP, the
Robonix Python API, and generated Navigation bindings. They skip when those
dependencies are unavailable. After building the package, include the API
source in the test environment:

```bash
PYTHONPATH="$(rbnx path robonix-api)${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m unittest -v test_runtime_integration.NavigationMcpEnvelopeTest
```

Jetson native builds require ROS 2 Humble and Nav2 packages compatible with the
host JetPack image. Docker manifests remain available for simulator and CI
deployments.

## Protocol responses

The `navigate`, `navigate/status`, and `navigate/cancel` capabilities return the
same typed response fields over gRPC and MCP. Business rejections are completed
MCP tool calls (`isError=false`) with a negative contract response:

- `navigate` before ROS 2 initialization returns `accepted=false`, an empty
  `run_id`, and a diagnostic `detail`.
- `navigate/status` for an unknown run returns `known=false`, `state="FAILED"`,
  and `detail="unknown run_id"`. Check `known` before interpreting `state`.
- `navigate/cancel` for an unknown or already terminal run returns
  `accepted=false` and a diagnostic `detail`.

Unexpected exceptions and transport failures remain errors. Clients must inspect
`accepted` or `known` after a completed call to distinguish a contract rejection
from an accepted request or known run.
