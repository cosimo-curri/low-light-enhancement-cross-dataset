from __future__ import annotations

from typing import Any


def group_sort_key(value: Any) -> tuple[int, int | str]:
    value = "" if value is None else str(value)

    if value.isdigit():
        return (0, int(value))

    return (1, value)