"""Text that tells the AI what it remembers about a person."""

MEMORY_HEADER = (
    "Things the user told you in earlier conversations "
    "(use them only if relevant to the current request):"
)


def build_memory_prompt(memories: list[str], saved: int, limit: int) -> str:
    """Describe the person's memory and list the memories relevant now."""
    lines = [
        "You have long-term memory of this user. "
        f"Saved messages: {saved} (only their latest {limit} are kept, "
        "older ones are forgotten automatically). "
        "If they ask what or how much you remember about them, tell them this."
    ]
    if memories:
        lines += ["", MEMORY_HEADER, *(f"- {memory}" for memory in memories)]
    return "\n".join(lines)
