"""Streaming PII redactor with a bounded holdback buffer.

Invariant: the concatenated output of feed()/flush() equals redact(full_text),
whatever the chunk boundaries are.
"""

import re

REDACTED = "[REDACTED]"

EMAIL = r"[\w.%+-]{1,64}@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,24}"
SSN = r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"
CARD = r"(?<!\d)\d(?:[ -]?\d){12,15}(?!\d)"
PII = re.compile(f"{EMAIL}|{SSN}|{CARD}")
MAX_MATCH = 64 + 1 + 255 + 1 + 24  # longest possible email; SSN and card are shorter

# Text at the end of the buffer that could still grow into (or be) a match.
# Each pattern accepts every prefix of the matching PII pattern (a superset is fine).
PARTIAL = re.compile(
    r"(?:"
    r"[\w.%+-]{1,64}(?:@[A-Za-z0-9.-]{0,280})?"  # email prefix
    r"|[\d-]{1,11}"  # SSN prefix
    r"|\d(?:[ -]?\d){0,15}[ -]?"  # card prefix
    r")\Z"
)
# A held suffix is at most one partial match plus one complete match that overlaps it.
MAX_HOLDBACK = 2 * MAX_MATCH


def redact(text: str) -> str:
    return PII.sub(REDACTED, text)


class StreamRedactor:
    def __init__(self) -> None:
        self._held = ""
        self._prev = ""  # last released char, so lookbehinds see real context

    def feed(self, text: str, final: bool = False) -> str:
        buf = self._held + text
        partial = None if final else PARTIAL.search(buf, max(0, len(buf) - MAX_MATCH))
        cut = partial.start() if partial else len(buf)

        out, pos = [], 0
        offset = len(self._prev)
        for m in PII.finditer(self._prev + buf, offset):
            start, end = m.start() - offset, m.end() - offset
            if end > cut:
                cut = min(cut, start)  # never split a complete match
                break
            out += [buf[pos:start], REDACTED]
            pos = end
        out.append(buf[pos:cut])

        if cut:
            self._prev = buf[cut - 1]
        self._held = buf[cut:]
        return "".join(out)

    def flush(self) -> str:
        """Release (redacted) everything still held. Call once at stream end."""
        return self.feed("", final=True)
