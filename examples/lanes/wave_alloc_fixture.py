"""The synthetic Xbox compute shader the Lanes self-test loads for the
wave-alloc self-test.

It is a small hand-written console program: a constant input, a wave32 masked
DPP exclusive-prefix (the pattern the translator lowers to `WavePrefixSum`),
and an `s_endpgm`. Not taken from any game.
"""
import hashlib
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.reference.scarlett_worker_fixture import container, pack  # noqa: E402

CODE = (
    0x7e000281,  # v_mov_b32 v0, 1
    0xbe803dc1,  # s_or_saveexec_b32 s0, -1
    0xd5010001,  # v_cndmask_b32 v1, 0, v0, s0
    0x00020080,
    0x7e040301,  # v_mov_b32 v2, v1
    0x4a0404fa,  # v_add_nc_u32 v2, 250, v2 (DPP row_shr)
    0xff091102,
    0x4a0404fa,
    0xff011202,
    0x4a0404fa,
    0xff011402,
    0x4a0404fa,
    0xff011802,
    0xbefe03ff,  # s_mov_b32 ExecLo, 0xffff0000
    0xffff0000,
    0xd7780003,  # v_permlanex16_b32 v3, v2, -1, -1
    0x03058302,
    0x4a040503,  # v_add_nc_u32 v2, v3, v2
    0xbefe03c1,  # s_mov_b32 ExecLo, -1
    0x4c040302,  # v_sub_nc_u32 v2, v2, v1
    0xbefe0300,  # s_mov_b32 ExecLo, s0
    0x7e080302,  # v_mov_b32 v4, v2
    0xbf810000,  # s_endpgm
)
# Root signature 1.0 with no parameters, samplers or flags.
EMPTY_ROOT = pack([1, 0, 24, 0, 24, 0])


def shader():
    base = container(list(CODE), width=32)
    count = struct.unpack_from("<I", base, 28)[0]
    offsets = struct.unpack_from(f"<{count}I", base, 32)
    chunks = [base[start:start + 8 + struct.unpack_from("<I", base, start + 4)[0]] for start in offsets]
    chunks.append(b"RTS0" + pack([len(EMPTY_ROOT)]) + EMPTY_ROOT)
    start = 32 + 4 * len(chunks)
    table = []
    for chunk in chunks:
        table.append(start)
        start += len(chunk)
    return b"DXBC" + bytes(16) + pack([1, start, len(chunks), *table]) + b"".join(chunks)


def header(data):
    values = ",".join(str(byte) for byte in data)
    return ("// Generated from wave_alloc_fixture.py.\n"
            f"constexpr unsigned char wave_alloc_shader[] = {{{values}}};\n"
            "// The name its translation is cached under.\n"
            f"constexpr wchar_t wave_alloc_shader_digest[] = L\"{hashlib.sha256(data).hexdigest()}\";\n")


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(header(shader()))
