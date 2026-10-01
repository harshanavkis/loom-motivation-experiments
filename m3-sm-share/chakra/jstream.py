#!/usr/bin/env python3
"""Stream-parse large trace JSONs (Kineto device traces, PyTorch host ETs) with the stdlib only.

Both formats are one top-level object holding one large array ("traceEvents" for Kineto, "nodes" for
the host ET) plus small scalar/object fields before and after it. stream_array() reads the file in
16 MiB chunks and decodes one array element at a time with json.JSONDecoder.raw_decode (C scanner),
so the raw text is never held in memory as a whole; only the elements the caller keeps are retained.
"""
import json, re

_WS = re.compile(r"[\s,]*")


def stream_array(path, key, meta=None, chunk=1 << 24):
    """Yield the elements of the top-level array `key`; top-level fields before/after it go into `meta`."""
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        buf = f.read(chunk)
        pat = '"%s"' % key
        i = buf.find(pat)
        while i < 0:
            more = f.read(chunk)
            if not more: raise ValueError(f"{path}: key {key!r} not found")
            buf += more; i = buf.find(pat)
        head = buf[:i].rstrip()
        if meta is not None:
            if head.endswith(","): head = head[:-1]
            meta.update(json.loads(head + "}"))   # also asserts the key was found at top level
        buf = buf[buf.index("[", i) + 1:]; pos = 0
        while True:
            pos = _WS.match(buf, pos).end()
            if pos >= len(buf):
                more = f.read(chunk)
                if not more: raise ValueError(f"{path}: truncated array {key!r}")
                buf = buf[pos:] + more; pos = 0; continue
            if buf[pos] == "]": break
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = f.read(chunk)
                if not more: raise
                buf = buf[pos:] + more; pos = 0; continue
            yield obj
            pos = end
            if pos > chunk: buf = buf[pos:]; pos = 0
        tail = (buf[pos + 1:] + f.read()).strip()
        if meta is not None:
            if tail.startswith(","): tail = tail[1:]
            if tail.strip() != "}": meta.update(json.loads("{" + tail))


def load_kineto(path, cats=None):
    """Kineto JSON as a dict like json.load() would return, but streamed; traceEvents keeps only events
    whose "cat" is in `cats` (all events if cats is None)."""
    meta = {}
    ev = [e for e in stream_array(path, "traceEvents", meta) if cats is None or e.get("cat") in cats]
    meta["traceEvents"] = ev
    return meta
