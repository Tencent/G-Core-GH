import re


def contains_renderable_field(s: str, key: str) -> bool:
    """Check whether ``s`` contains a renderable field named ``key``.

    Args:
        s: The string to inspect.
        key: Name of the renderable field.

    Returns:
        True if ``s`` contains patterns like ``{key}``, ``{key:format}``,
        ``{key.attr}``, ``{key[index]}``, etc.
    """
    if not isinstance(s, str):
        raise TypeError("Input 's' must be a string.")
    if not isinstance(key, str):
        raise TypeError("Input 'key' must be a string.")

    pattern = r"\{" + re.escape(key) + r"(?!\w).*\}"
    return re.search(pattern, s) is not None
