"""The synthetic Xbox raster programs of the `lane-spill-interp` self-test.

All are small hand-written console programs, not taken from any game. The
pixel program spills a lane-spill area with `v_writelane_b32` and then reads
that register only as the position operand of an f16 raster interpolator,
whose position is never emitted. The translation must accept that program:
the spill register's only vector use is dead, so its translation must not
refuse the whole shader because of it. Any live use of the spill register
as a vector remains refused. The interp writes a register the program never
reads, so the translation must keep the spill and interpolator validated and
write a constant to the target: a stand-in pipeline leaves the target at its
clear value, while the translated one reaches the expected constant.

The f16 interpolator opcodes are not encodable by llvm-mc (its parser rejects
the f16 interpolator attribute), so the words are hand-encoded from the RDNA
2 ISA bit layout, the same words the translation's decoder and the file's
tests verify. The spill and writeout words are from llvm-mc.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.lanes.raster_fixture import image, signature, validation, container  # noqa: E402

# gfx1030; the f16 interpolator pair is hand-encoded (see module docstring):
#   first word:  Vop3 class 0x35, opcode Vop3 834 (v_interp_p1ll_f16), dst v14.
#   second word: src0 = v128 (selector 384 = 0x180: attr 0, channel 2, high 1,
#   which is what makes src0's low 9 bits the position register), factor v0
#   (selector 256 = 0x100), p2 partial s0 (selector 0).
PIXEL_CODE = (
    0xD7610080, 0x02010000,  # v_writelane_b32 v128, s0, 0   spill area (llvm-mc)
    0xD742000E, 0x00020180,  # v_interp_p1ll_f16 v14, v128, v0, s0   position is dead
    0x7E0802FF, 0x7AB1E003,  # v_mov_b32 v4, 0x7AB1E003   constant result (llvm-mc)
    0xF8180001, 0x00000004,  # exp mrt0 v4, off, off, off done vm
    0xBF810000,              # s_endpgm
)
# The constant the pixel program writes to the target: 0x7AB1E003.
EXPECTED = 0x7AB1E003


def pixel():
    return container(0x60, image(5, PIXEL_CODE, (), (), inputs=1), [
        (b"ISGN", signature([
            ("SV_Position", 0, 1, 3, 0, 15, 0),
            ("TEXCOORD", 1, 0, 3, 1, 15, 15),
        ])),
        (b"OSGN", signature([("SV_Target", 0, 64, 1, 0, 1, 0)])),
        (b"PSV0", validation(0)),
    ])


def declaration(name, data):
    values = ",".join(str(byte) for byte in data)
    return f"constexpr unsigned char {name}[] = {{{values}}};\n"


def header():
    return ("// Generated from spill_fixture.py.\n"
            + declaration("lane_spill_pixel", pixel())
            + f"constexpr UINT lane_spill_expected = {EXPECTED:#x};\n")


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(header())
