"""The synthetic Xbox vertex and pixel programs of the `raster-tables` self-test.

Both are small hand-written console programs, not taken from any game. The
vertex program draws a fullscreen triangle. The pixel program reads one dword
through table 1 of a root with two descriptor tables, from the table pointer in
s9 at slot 2, and writes it to a uint render target. Its FSSM records and PSV
binding name that descriptor `t3, space2`, a register the title's root does not
declare, so the pipeline needs the root its tables are reflected into.
"""
from pathlib import Path
import struct
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.reference.portable_link_fixture import VERTEX_CODE, pack  # noqa: E402

# gfx1030, checked with `llvm-mc -mcpu=gfx1030 -show-encoding`.
PIXEL_CODE = (
    0xBE800309,              # s_mov_b32 s0, s9          table 1 pointer
    0xBE810380,              # s_mov_b32 s1, 0
    0xF4080100, 0xFA000040,  # s_load_dwordx4 s[4:7], s[0:1], 0x40   slot 2
    0xBF8CC07F,              # s_waitcnt lgkmcnt(0)
    0xE0300000, 0x80010000,  # buffer_load_dword v0, off, s[4:7], 0
    0xBF8C3F70,              # s_waitcnt vmcnt(0)
    0xF8001801, 0x00000000,  # exp mrt0 v0, off, off, off done vm
    0xBF810000,              # s_endpgm
)
# FSSM rows [class, dword offset, width or register, register or shape, location].
# The user-data row puts the descriptor in table 1 (s9) at dword 16; the used row
# names it a raw SRV, register 3, with its table in the location's high half.
PIXEL_USER = ([3, 16, 1, 3, 0x109],)
PIXEL_USED = ([0x300, 16, 3, 1, 0x1_0904],)
# PSV binding [type, space, lower, upper]: raw SRV t3, space2.
PIXEL_RESOURCES = ([4, 2, 3, 3],)
# The value the pixel program must read, and what the other descriptors hold.
EXPECTED = 0x7AB1E001
DECOY = 0xD0000000

ABSENT = 0xDEADBEEFDEADBEEF
ROWS_AT, CODE_AT = 0x240, 0x280


def image(slot, code, user=(), used=(), inputs=0):
    """An FSSM 1.1 image with one stage descriptor at 0x68, after the directory."""
    data = bytearray(CODE_AT)
    struct.pack_into("<Q", data, 0, 6)
    for n in range(12):
        struct.pack_into("<Q", data, 8 + 8 * n, 0x68 if n == slot else ABSENT)
    struct.pack_into("<QQ", data, 0x68, CODE_AT, 4 * len(code))
    # Descriptor fields at 0xd8: user-data and used-resource row offsets and counts.
    used_at = ROWS_AT + 20 * len(user)
    struct.pack_into("<QQII", data, 0xd8, ROWS_AT if user else 0, used_at if used else 0, len(user), len(used))
    for index, row in enumerate((*user, *used)):
        struct.pack_into("<5I", data, ROWS_AT + 20 * index, *row)
    struct.pack_into("<I", data, 0x238, inputs)  # SPI_PS_INPUT_ENA
    data.extend(pack(code))
    return bytes(data)


def signature(rows):
    """Signature part rows: (name, index, system value, component type, register, mask, read/write mask)."""
    data = bytearray(pack([len(rows), 8]))
    names = bytearray()
    for name, index, sysval, kind, register, mask, used in rows:
        data.extend(pack([8 + 24 * len(rows) + len(names), index, sysval, kind, register, mask | used << 8]))
        names.extend(name.encode("ascii") + b"\0")
    return bytes(data + names)


def validation(stage, resources=()):
    info = bytearray(48)
    info[24] = stage
    psv = pack([len(info)]) + info + pack([len(resources)])
    if resources:
        psv += pack([16, *(word for row in resources for word in row)])
    return psv


def container(version, fssm_image, parts):
    fssm = b"FSSM" + pack([0x101]) + struct.pack("<Q", len(fssm_image)) + fssm_image
    compressed = zlib.compress(fssm)
    section = b"XBCH" + pack([3]) + struct.pack("<QQ", len(compressed), len(fssm)) + compressed
    xdat = b"XDAT" + pack([2, 36, 13, 60, len(section), 7, 60 + ((len(section) + 3) // 4) * 4, 4]) + section
    xdat += bytes((-len(xdat)) % 4) + b"NOBC"
    body = pack([version, (24 + len(xdat)) // 4]) + b"DXIL" + pack([0x100, 16 + len(xdat) - 4, 4]) + xdat
    chunks = [(b"DXIL", body), *parts]
    start = 32 + 4 * len(chunks)
    offsets, data = [], bytearray()
    for tag, part in chunks:
        offsets.append(start + len(data))
        data.extend(tag + pack([len(part)]) + part)
        data.extend(bytes((-len(data)) % 4))
    return b"DXBC" + bytes(16) + pack([1, start + len(data), len(chunks), *offsets]) + data


def vertex():
    return container(0x10060, image(7, VERTEX_CODE), [
        (b"ISGN", signature([("SV_VertexID", 0, 6, 1, 0, 1, 1)])),
        (b"OSGN", signature([("SV_Position", 0, 1, 3, 0, 15, 0), ("TEXCOORD", 0, 0, 3, 1, 15, 0)])),
        (b"PSV0", validation(1)),
    ])


def pixel():
    return container(0x60, image(5, PIXEL_CODE, PIXEL_USER, PIXEL_USED, inputs=2), [
        (b"ISGN", signature([("SV_Position", 0, 1, 3, 0, 15, 0)])),
        (b"OSGN", signature([("SV_Target", 0, 64, 1, 0, 1, 0)])),
        (b"PSV0", validation(0, PIXEL_RESOURCES)),
    ])


def header(vertex_data, pixel_data):
    def array(name, data):
        return f"constexpr unsigned char {name}[] = {{{','.join(str(byte) for byte in data)}}};\n"
    return ("// Generated from raster_fixture.py.\n"
            + array("raster_vertex", vertex_data) + array("raster_pixel", pixel_data)
            + f"constexpr UINT raster_expected = {EXPECTED:#x}, raster_decoy = {DECOY:#x};\n")


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(header(vertex(), pixel()))
