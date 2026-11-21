#!/usr/bin/env python3
import json
import time
import threading
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Twist
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
            '/home/unitree/go2_nav2_ws/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
        )
        self.declare_parameter('n_gpu_layers', -1)
        self.declare_parameter('n_ctx', 512)
        self.declare_parameter('target_timeout_sec', 10.0)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        n_gpu_layers = self.get_parameter('n_gpu_layers').get_parameter_value().integer_value
        n_ctx = self.get_parameter('n_ctx').get_parameter_value().integer_value
        self.target_timeout_sec = (
            self.get_parameter('target_timeout_sec').get_parameter_value().double_value
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

        self.parse_intent("stop") # dummy warm-up call

        # Thread-safe state tracking
        self.cb_group = ReentrantCallbackGroup()
        self.state = FSMState.IDLE
        self.latest_goal_pose: Optional[PoseStamped] = None
        self.goal_lock = threading.Lock()

        # Publishers
        self.vocab_pub = self.create_publisher(String, '/update_yolo_vocabulary', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers (receives 3D projected target pose in map frame)
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

    def _goal_pose_callback(self, msg: PoseStamped):
        with self.goal_lock:
            self.latest_goal_pose = msg

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
        """Publish zero velocity and cancel any active Nav2 goal."""
        self.cmd_vel_pub.publish(Twist())
        if self.cancel_goal_client.service_is_ready():
            self.cancel_goal_client.call_async(CancelGoal.Request())

    def _text_command_callback(self, msg: String):
        """Allows direct CLI testing via /natural_language_command."""
        threading.Thread(
            target=self.run_fsm_pipeline, args=(msg.data, None), daemon=True
        ).start()

    def run_fsm_pipeline(
        self, command_text: str, goal_handle=None
    ) -> Tuple[bool, str, str, str]:
        """Minimal Stage 1: Parse intent via LLM and publish target to YOLO-World."""
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
        vocab_msg = String()
        if action == 'stop' or not target:
            vocab_msg.data = ''
            self.vocab_pub.publish(vocab_msg)
            self.get_logger().info('Stop intent received. Cleared YOLO-World vocabulary (STANDBY).')
            self._transition_state(FSMState.COMPLETED, goal_handle)
            self._transition_state(FSMState.IDLE)
            return True, action, target, 'Cleared YOLO-World target (STANDBY).'

        vocab_msg.data = target
        self.vocab_pub.publish(vocab_msg)
        self.get_logger().info(f'Published target vocabulary to YOLO-World: "{target}"')

        self._transition_state(FSMState.COMPLETED, goal_handle)
        self._transition_state(FSMState.IDLE)
        return True, action, target, f'Sent target "{target}" to YOLO-World.'

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

