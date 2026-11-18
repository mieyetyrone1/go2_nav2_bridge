#!/usr/bin/env python3
from enum import Enum


class FSMState(str, Enum):
    IDLE = "IDLE"
    PARSING_INTENT = "PARSING_INTENT"
    UPDATING_PERCEPTION = "UPDATING_PERCEPTION"
    # Dedicated hooks for upcoming feature branches:
    MODE_B_ACTIVE_PERCEPTION = "MODE_B_ACTIVE_PERCEPTION"
    MODE_A_VISUAL_SERVOING = "MODE_A_VISUAL_SERVOING"
    MODE_C_LONG_RANGE_STAGING = "MODE_C_LONG_RANGE_STAGING"
    NAVIGATING = "NAVIGATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# Strict GBNF grammar guaranteeing syntactically valid JSON output
INTENT_GBNF_GRAMMAR = r"""
root   ::= "{" ws "\"action\"" ws ":" ws action "," ws "\"target\"" ws ":" ws string ws "}"
action ::= "\"navigate\"" | "\"stop\"" | "\"search\""
string ::= "\"" (
             [^"\\] |
             "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
           )* "\""
ws     ::= [ \t\n]*
"""


def build_qwen_intent_prompt(user_command: str) -> str:
    """Construct a ChatML prompt for Qwen2.5-Coder-1.5B-Instruct."""
    system_msg = (
        "You are an intent parser for a quadruped mobile robot. "
        "Extract the user's requested action ('navigate', 'search', or 'stop') "
        "and the physical target object (lowercase noun phrase, or empty string if stopping). "
        "Respond ONLY with valid JSON matching the schema."
    )
    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_command.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

