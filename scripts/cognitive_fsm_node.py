#!/usr/bin/env python3
import json
import math
import time
import threading
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, Twist, PolygonStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.srv import CancelGoal

from go2_nav2_bridge.action import ExecuteCognitiveIntent
from llama_cpp import Llama, LlamaGrammar
from fsm_grammar import FSMState, INTENT_GBNF_GRAMMAR, build_qwen_intent_prompt


class CognitiveFSMNode(Node):
    def __init__(self):
        super().__init__('cognitive_fsm_node')

        # Parameters
        self.declare_parameter(
            'model_path',
            '/home/unitree/go2_nav2_ws/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf',
        )
        self.declare_parameter('n_gpu_layers', -1)
        self.declare_parameter('n_ctx', 512)
        self.declare_parameter('target_timeout_sec', 90.0)
        self.declare_parameter('image_width', 640.0)
        self.declare_parameter('center_tolerance_px', 35.0)
        self.declare_parameter('servo_kp', 0.0018)
        self.declare_parameter('servo_max_yaw', 0.45)
        self.declare_parameter('servo_min_yaw', 0.18)
        self.declare_parameter('bbox_stale_timeout_sec', 2.0)
        self.declare_parameter('initial_observe_sec', 11.5)
        self.declare_parameter('search_yaw_speed', 0.45)
        self.declare_parameter('search_step_sec', 1.16)
        self.declare_parameter('search_dwell_sec', 3.0)
        self.declare_parameter('max_search_steps', 13)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        n_gpu_layers = self.get_parameter('n_gpu_layers').get_parameter_value().integer_value
        n_ctx = self.get_parameter('n_ctx').get_parameter_value().integer_value
        self.target_timeout_sec = (
            self.get_parameter('target_timeout_sec').get_parameter_value().double_value
        )
        self.image_width = (
            self.get_parameter('image_width').get_parameter_value().double_value
        )
        self.center_tolerance_px = (
            self.get_parameter('center_tolerance_px').get_parameter_value().double_value
        )
        self.servo_kp = self.get_parameter('servo_kp').get_parameter_value().double_value
        self.servo_max_yaw = (
            self.get_parameter('servo_max_yaw').get_parameter_value().double_value
        )
        self.servo_min_yaw = (
            self.get_parameter('servo_min_yaw').get_parameter_value().double_value
        )
        self.bbox_stale_timeout_sec = (
            self.get_parameter('bbox_stale_timeout_sec').get_parameter_value().double_value
        )
        self.initial_observe_sec = (
            self.get_parameter('initial_observe_sec').get_parameter_value().double_value
        )
        self.search_yaw_speed = (
            self.get_parameter('search_yaw_speed').get_parameter_value().double_value
        )
        self.search_step_sec = (
            self.get_parameter('search_step_sec').get_parameter_value().double_value
        )
        self.search_dwell_sec = (
            self.get_parameter('search_dwell_sec').get_parameter_value().double_value
        )
        self.max_search_steps = (
            self.get_parameter('max_search_steps').get_parameter_value().integer_value
        )

        # Load GPU-accelerated LLM once at node startup
        self.get_logger().info(f'Loading GGUF model onto GPU from: {model_path}')
        self.grammar = LlamaGrammar.from_string(INTENT_GBNF_GRAMMAR)
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )
        self.get_logger().info('GPU LLM initialized and ready.')

        self.parse_intent('stop')  # dummy warm-up call

        # Thread-safe state & perception tracking
        self.cb_group = ReentrantCallbackGroup()
        self.state = FSMState.IDLE
        self.active_target: str = ''
        self.cancel_requested: bool = False

        self.data_lock = threading.Lock()
        self.latest_bbox: Optional[PolygonStamped] = None
        self.latest_bbox_mono_time: float = 0.0
        self.bbox_seq: int = 0
        self.latest_goal_pose: Optional[PoseStamped] = None

        # Publishers
        self.vocab_pub = self.create_publisher(String, '/update_yolo_vocabulary', 10)
        self.projector_enable_pub = self.create_publisher(Bool, '/projector_enable', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.bbox_sub = self.create_subscription(
            PolygonStamped,
            '/tracker/tracked_target',
            self._bbox_callback,
            qos_profile_sensor_data,
            callback_group=self.cb_group,
        )
        self.goal_pose_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self._goal_pose_callback,
            10,
            callback_group=self.cb_group,
        )

        # Nav2 Action Client & Cancel Service
        self.nav2_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose', callback_group=self.cb_group
        )
        self.cancel_goal_client = self.create_client(
            CancelGoal, 'navigate_to_pose/_action/cancel_goal', callback_group=self.cb_group
        )

        # Topic trigger for quick CLI testing
        self.text_cmd_sub = self.create_subscription(
            String,
            '/natural_language_command',
            self._text_command_callback,
            10,
            callback_group=self.cb_group,
        )

        # Custom ROS 2 Action Server
        self._action_server = ActionServer(
            self,
            ExecuteCognitiveIntent,
            'execute_cognitive_intent',
            execute_callback=self._execute_action_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self.cb_group,
        )

        self.get_logger().info('CognitiveFSMNode started in state: IDLE')

    def _transition_state(self, new_state: FSMState, goal_handle=None):
        self.get_logger().info(f'FSM Transition: {self.state.value} -> {new_state.value}')
        self.state = new_state
        if goal_handle is not None:
            feedback = ExecuteCognitiveIntent.Feedback()
            feedback.current_state = new_state.value
            goal_handle.publish_feedback(feedback)

    def _bbox_callback(self, msg: PolygonStamped):
        if len(msg.polygon.points) < 2:
            return
        with self.data_lock:
            if self.active_target and msg.header.frame_id == self.active_target:
                self.latest_bbox = msg
                self.latest_bbox_mono_time = time.monotonic()
                self.bbox_seq += 1

    def _goal_pose_callback(self, msg: PoseStamped):
        with self.data_lock:
            self.latest_goal_pose = msg

    def _publish_zero_vel(self):
        self.cmd_vel_pub.publish(Twist())

    def _set_projector_enabled(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.projector_enable_pub.publish(msg)

    def parse_intent(self, command: str) -> dict:
        """Run GBNF-constrained inference on Qwen2.5-Coder-1.5B."""
        prompt = build_qwen_intent_prompt(command)
        t0 = time.perf_counter()
        output = self.llm(
            prompt,
            max_tokens=64,
            temperature=0.0,
            grammar=self.grammar,
            stop=['<|im_end|>'],
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        raw_text = output['choices'][0]['text'].strip()
        self.get_logger().info(f'LLM parsed in {latency_ms:.1f} ms: {raw_text}')
        return json.loads(raw_text)

    def _emergency_stop(self):
        """Publish zero velocity, gate projector, and cancel any active Nav2 goal."""
        self.cancel_requested = True
        self._publish_zero_vel()
        self._set_projector_enabled(False)
        if self.cancel_goal_client.service_is_ready():
            self.cancel_goal_client.call_async(CancelGoal.Request())

    def _text_command_callback(self, msg: String):
        """Allows direct CLI testing via /natural_language_command."""
        threading.Thread(
            target=self.run_fsm_pipeline, args=(msg.data, None), daemon=True
        ).start()

    def _wait_for_detection(self, timeout_sec: float, goal_handle=None) -> bool:
        """Wait up to timeout_sec for any matching bounding box (bbox_seq > 0)."""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self.cancel_requested or (
                goal_handle is not None and goal_handle.is_cancel_requested
            ):
                return False
            with self.data_lock:
                if self.bbox_seq > 0 and self.latest_bbox is not None:
                    return True
            time.sleep(0.05)
        with self.data_lock:
            return self.bbox_seq > 0 and self.latest_bbox is not None

    def _execute_motion_pulse(
        self,
        lin_x: float,
        ang_z: float,
        duration_sec: float,
        goal_handle=None,
        stop_on_detection: bool = True,
    ) -> bool:
        """Publish a bounded differential-drive Twist pulse while respecting cancels/detections."""
        twist = Twist()
        twist.linear.x = lin_x
        twist.angular.z = ang_z
        pulse_end = time.monotonic() + duration_sec
        while rclpy.ok() and time.monotonic() < pulse_end:
            if self.cancel_requested or (
                goal_handle is not None and goal_handle.is_cancel_requested
            ):
                self._publish_zero_vel()
                return False
            if stop_on_detection:
                with self.data_lock:
                    if self.bbox_seq > 0 and self.latest_bbox is not None:
                        self._publish_zero_vel()
                        time.sleep(0.25)
                        return True
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.05)
        self._publish_zero_vel()
        time.sleep(0.35)  # Allow chassis to settle before inference
        return True

    def _run_mode_b_active_perception(self, target: str, goal_handle=None) -> bool:
        """Mode B: Pure in-place yaw oscillations (+20 -> -40 deg) followed by 360-deg scan."""
        if self._wait_for_detection(self.initial_observe_sec, goal_handle):
            self.get_logger().info(
                f'[MODE B] Target "{target}" already visible in initial FOV. Skipping active perception.'
            )
            return True

        self._transition_state(FSMState.MODE_B_ACTIVE_PERCEPTION, goal_handle)

        # Stage B1: Pure in-place forward yaw oscillations (+20 deg left -> -40 deg right)
        yaw_glances = [
            ("+20 deg left glance", self.search_yaw_speed, 0.78),
            ("-40 deg right glance", -self.search_yaw_speed, 1.55),
        ]

        for label, wz, dur in yaw_glances:
            self.get_logger().info(
                f'[MODE B - WIGGLE] Executing {label} (in-place) for "{target}"...'
            )
            if not self._execute_motion_pulse(0.0, wz, dur, goal_handle):
                return False
            if self._wait_for_detection(self.search_dwell_sec, goal_handle):
                self.get_logger().info(
                    f'[MODE B - WIGGLE] Acquired "{target}" during {label}! Handing off to Mode A.'
                )
                return True

        # Stage B2: Pure in-place 360-deg step-and-scan (~30 deg per step)
        self.get_logger().info(
            f'[MODE B - SCAN] Forward glances did not reveal "{target}". '
            f'Starting in-place 360-deg step-and-scan ({self.max_search_steps} steps)...'
        )
        for step_idx in range(1, self.max_search_steps + 1):
            self.get_logger().info(
                f'[MODE B - SCAN] Step {step_idx}/{self.max_search_steps} for "{target}"...'
            )
            if not self._execute_motion_pulse(
                0.0, self.search_yaw_speed, self.search_step_sec, goal_handle
            ):
                return False
            if self._wait_for_detection(self.search_dwell_sec, goal_handle):
                self.get_logger().info(
                    f'[MODE B - SCAN] Acquired "{target}" on step {step_idx}! Handing off to Mode A.'
                )
                return True

        self._publish_zero_vel()
        self.get_logger().warn(
            f'[MODE B] Completed in-place yaw wiggle and 360-deg scan without detecting "{target}".'
        )
        return False

    def _run_mode_a_visual_servoing(self, target: str, goal_handle=None) -> bool:
        """Mode A: Step-and-settle horizontal centering matched to ~1.5s YOLO inference rate."""
        self._transition_state(FSMState.MODE_A_VISUAL_SERVOING, goal_handle)
        u_center_ref = self.image_width / 2.0
        start_time = time.monotonic()
        last_log_time = 0.0
        last_processed_seq = 0
        last_fresh_time = time.monotonic()
        last_err_x = float('inf')


        while rclpy.ok():
            if self.cancel_requested or (
                goal_handle is not None and goal_handle.is_cancel_requested
            ):
                self._publish_zero_vel()
                return False

            now = time.monotonic()
            if (now - start_time) > self.target_timeout_sec:
                self._publish_zero_vel()
                self.get_logger().warn(
                    f'[MODE A] Timeout ({self.target_timeout_sec:.1f}s) centering "{target}".'
                )
                return False

            with self.data_lock:
                bbox = self.latest_bbox
                seq = self.bbox_seq

            # Wait strictly for a NEW detection frame after each step settles
            if bbox is None or seq == last_processed_seq:
                if (
                    last_processed_seq > 0
                    and abs(last_err_x) <= self.center_tolerance_px
                    and (now - last_fresh_time) > 4.5
                    and bbox is not None
                ):
                    self._publish_zero_vel()
                    self.get_logger().warn(
                        f'[MODE A] Detection flickered for >4.5s after centering; '
                        f'proceeding with last centered "{target}" detection!'
                    )
                    return True
                if (now - last_log_time) > 2.0:
                    self.get_logger().info(
                        f'[MODE A] Waiting for next fresh "{target}" detection...'
                    )
                    last_log_time = now
                time.sleep(0.05)
                continue

            last_processed_seq = seq
            last_fresh_time = now
            x_min = bbox.polygon.points[0].x
            x_max = bbox.polygon.points[1].x
            u_target = (x_min + x_max) / 2.0
            err_x = u_center_ref - u_target  # +err_x => target is left => +angular.z (CCW)
            last_err_x = err_x

            if abs(err_x) <= self.center_tolerance_px:
                self._publish_zero_vel()
                self.get_logger().info(
                    f'[MODE A] Target "{target}" centered at u={u_target:.1f}px '
                    f'(err={err_x:+.1f}px <= ±{self.center_tolerance_px:.0f}px). Ready!'
                )
                time.sleep(0.20)
                return True

            # Compute bounded discrete step pulse (~4 deg to ~14 deg per observation)
            yaw_speed = 0.35
            pulse_sec = max(0.18, min(0.55, abs(err_x) * 0.0022))
            yaw_cmd = math.copysign(yaw_speed, err_x)

            self.get_logger().info(
                f'[MODE A] Stepping toward "{target}": u={u_target:.1f}px, '
                f'err={err_x:+.1f}px -> pulse {yaw_cmd:+.2f} rad/s for {pulse_sec:.2f}s'
            )

            twist = Twist()
            twist.angular.z = yaw_cmd
            pulse_end = time.monotonic() + pulse_sec
            while rclpy.ok() and time.monotonic() < pulse_end:
                if self.cancel_requested or (
                    goal_handle is not None and goal_handle.is_cancel_requested
                ):
                    self._publish_zero_vel()
                    return False
                self.cmd_vel_pub.publish(twist)
                time.sleep(0.05)

            # Immediately stop and wait for chassis + next YOLO frame to settle
            self._publish_zero_vel()
            time.sleep(0.35)

        self._publish_zero_vel()
        return False

    def run_fsm_pipeline(
        self, command_text: str, goal_handle=None
    ) -> Tuple[bool, str, str, str]:
        """Parse intent, update YOLO-World, run Mode A centering, and enable projection."""
        self.cancel_requested = False
        self._transition_state(FSMState.PARSING_INTENT, goal_handle)
        try:
            intent = self.parse_intent(command_text)
        except Exception as exc:
            err_msg = f'Intent parsing failed: {exc}'
            self.get_logger().error(err_msg)
            self._transition_state(FSMState.FAILED, goal_handle)
            self._transition_state(FSMState.IDLE)
            return False, '', '', err_msg

        action = intent.get('action', '')
        target = intent.get('target', '').strip()

        self._transition_state(FSMState.UPDATING_PERCEPTION, goal_handle)
        with self.data_lock:
            self.active_target = '' if (action == 'stop' or not target) else target
            self.latest_bbox = None
            self.latest_bbox_mono_time = 0.0
            self.bbox_seq = 0
            self.latest_goal_pose = None

        self._set_projector_enabled(False)

        vocab_msg = String()
        if action == 'stop' or not target:
            vocab_msg.data = ''
            self.vocab_pub.publish(vocab_msg)
            self._emergency_stop()
            self.get_logger().info('Stop intent received. Cleared YOLO-World vocabulary (STANDBY).')
            self._transition_state(FSMState.COMPLETED, goal_handle)
            self._transition_state(FSMState.IDLE)
            return True, action, target, 'Cleared YOLO-World target and stopped.'

        vocab_msg.data = target
        self.vocab_pub.publish(vocab_msg)
        self.get_logger().info(f'Published target vocabulary to YOLO-World: "{target}"')

        # Execute Mode B: Active Viewpoint Perspective Shift + 360-deg Search Scan
        found = self._run_mode_b_active_perception(target, goal_handle)
        if not found:
            self._publish_zero_vel()
            self._transition_state(FSMState.FAILED, goal_handle)
            self._transition_state(FSMState.IDLE)
            return False, action, target, f'Could not find "{target}" during active perception.'

        # Execute Mode A: Horizontal Visual Servoing Alignment
        aligned = self._run_mode_a_visual_servoing(target, goal_handle)
        if not aligned:
            self._publish_zero_vel()
            self._transition_state(FSMState.FAILED, goal_handle)
            self._transition_state(FSMState.IDLE)
            return False, action, target, f'Failed or canceled while aligning to "{target}".'

        # Hand off to C++ Target Projector -> Nav2
        self._transition_state(FSMState.NAVIGATING, goal_handle)
        self._set_projector_enabled(True)
        self.get_logger().info(
            f'Enabled 3D target projection for centered target "{target}". Handing off to Nav2.'
        )

        self._transition_state(FSMState.COMPLETED, goal_handle)
        self._transition_state(FSMState.IDLE)
        return True, action, target, f'Centered and dispatched navigation goal for "{target}".'

    # --- ROS 2 Action Server Callbacks ---
    def _goal_callback(self, goal_request):
        self.get_logger().info(f'Received Action Goal: "{goal_request.command}"')
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().warn('Action Goal cancel requested.')
        self._emergency_stop()
        return CancelResponse.ACCEPT

    def _execute_action_callback(self, goal_handle):
        success, action, target, message = self.run_fsm_pipeline(
            goal_handle.request.command, goal_handle
        )
        result = ExecuteCognitiveIntent.Result()
        result.success = success
        result.extracted_action = action
        result.extracted_target = target
        result.message = message

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = CognitiveFSMNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



