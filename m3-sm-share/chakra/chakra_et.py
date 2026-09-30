"""Minimal pure-Python decoder for Chakra ET protobuf files (schema/protobuf/et_def.proto,
mlcommons/chakra). File = varint-length-prefixed GlobalMetadata, then varint-length-prefixed Node
messages. Only the wire format is needed; no protobuf library required."""
import struct

NODE_TYPES = {0: "INVALID", 1: "METADATA", 2: "MEM_LOAD", 3: "MEM_STORE", 4: "COMP",
              5: "COMM_SEND", 6: "COMM_RECV", 7: "COMM_COLL", 8: "STORAGE", 9: "DATA_OP"}
COLL_TYPES = {0: "ALL_REDUCE", 1: "REDUCE", 2: "ALL_GATHER", 3: "GATHER", 4: "SCATTER",
              5: "BROADCAST", 6: "ALL_TO_ALL", 7: "REDUCE_SCATTER", 8: "REDUCE_SCATTER_BLOCK",
              9: "BARRIER"}


def _varint(b, i):
    r = 0; s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s
        if c < 0x80:
            return r, i
        s += 7


def _fields(b):
    """yield (field_no, wire_type, value) for one message buffer"""
    i = 0; n = len(b)
    while i < n:
        k, i = _varint(b, i)
        f, wt = k >> 3, k & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 2:
            L, i = _varint(b, i); v = b[i:i + L]; i += L
        elif wt == 1:
            v = struct.unpack_from("<d", b, i)[0]; i += 8  # only doubles used as fixed64 here
        elif wt == 5:
            v = struct.unpack_from("<f", b, i)[0]; i += 4
        else:
            raise ValueError("wire type %d" % wt)
        yield f, wt, v


def _zz(v):
    return (v >> 1) ^ -(v & 1)


def _attr(b):
    name = None; val = None
    for f, wt, v in _fields(b):
        if f == 1:
            name = v.decode()
        elif f in (7, 9, 11, 13):  # int32/int64/uint32/uint64
            val = v if v < (1 << 63) else v - (1 << 64)
        elif f in (15, 17):
            val = _zz(v)
        elif f == 27:
            val = bool(v)
        elif f == 29:
            val = v.decode(errors="replace")
        elif f in (3, 5, 19, 21, 23, 25):
            val = v
        elif f in (8, 10, 12, 14):  # packed int lists inside XxxList{values=1}
            out = []
            for f2, wt2, v2 in _fields(v):
                if wt2 == 2:
                    j = 0
                    while j < len(v2):
                        x, j = _varint(v2, j); out.append(x)
                else:
                    out.append(v2)
            val = out
        elif f == 30:
            val = [v2.decode() for f2, wt2, v2 in _fields(v)]
        else:
            val = ("raw", f)
    return name, val


def _packed(v, wt):
    if wt == 0:
        return [v]
    out = []; j = 0
    while j < len(v):
        x, j = _varint(v, j); out.append(x)
    return out


def parse_node(b):
    n = {"ctrl_deps": [], "data_deps": [], "attr": {}, "type": 0, "start": 0, "dur": 0, "name": ""}
    for f, wt, v in _fields(b):
        if f == 1: n["id"] = v
        elif f == 2: n["name"] = v.decode(errors="replace")
        elif f == 3: n["type"] = v
        elif f == 4: n["ctrl_deps"] += _packed(v, wt)
        elif f == 5: n["data_deps"] += _packed(v, wt)
        elif f == 6: n["start"] = v
        elif f == 7: n["dur"] = v
        elif f == 10:
            k, val = _attr(v); n["attr"][k] = val
    return n


def read_et(path, parse_attrs=True):
    """returns (global_metadata_attrs, iterator of node dicts)"""
    data = open(path, "rb").read()
    L, i = _varint(data, 0)
    meta = {"version": None}
    for f, wt, v in _fields(data[i:i + L]):
        if f == 1: meta["version"] = v.decode()
        elif f == 2:
            k, val = _attr(v); meta[k] = val
    i += L

    def gen(i=i):
        n = len(data)
        while i < n:
            L, i = _varint(data, i)
            yield parse_node(data[i:i + L])
            i += L
    return meta, gen()
