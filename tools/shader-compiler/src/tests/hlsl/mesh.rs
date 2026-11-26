use super::*;

#[test]
fn launch_requires_the_packed_group_and_wave64_lane_definitions() {
    let words: [u32; 10] = [
        0xd7650000, 0x000100c1, // v0 = mbcnt_lo(-1, 0)
        0x7e000505, // s0 = readfirstlane(v5)
        0x9381ff03, 0x00040018, // s1 = s3[27:24]
        0xd7660000, 0x000200c1, // v0 = mbcnt_hi(-1, v0)
        0x900a8800, // s10 = s0 >> 8
        0xd7460001, 0x04010c01, // v1 = (s1 << 6) + v0
    ];
    let check = |words: &[u32]| {
        let bytes = words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>();
        validate_launch(&disassemble(&bytes), 8, Opnd::Vgpr(1))
    };
    assert!(check(&words).is_ok());
    let mut wrong_wave = words;
    wrong_wave[4] = 0x00040017;
    assert!(check(&wrong_wave).is_err());
    // An s_load_dwordx4 into s[0:3] clobbers both intermediate values.
    let mut clobbered = words.to_vec();
    clobbered.splice(7..7, [0xf408000d, 0xfa000000]);
    assert!(check(&clobbered).is_err());
    let mut wrong_group = words;
    wrong_group[7] = 0x900a8700;
    assert!(check(&wrong_group).is_err());
}

#[test]
fn only_uniform_values_can_bypass_a_wave_broadcast() {
    let validate = |words: &[u32]| {
        let bytes = words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>();
        { let body = disassemble(&bytes); validate_lane_ops(&body, &waterfall_reads(&body), false, false) }
    };
    // Entry v5 holds the same packed group identifier in every lane.
    assert!(validate(&[0x7e000505]).is_ok());
    // Counting preceding lanes overwrites v5 with a lane-dependent value.
    assert!(validate(&[0xd7650005, 0x000100c1, 0x7e000505]).is_err());
    // A new constant in v5 is safe to broadcast.
    assert!(validate(&[0x7e0a0281, 0x7e000505]).is_ok());
    // Shared memory would require synchronization among logical Xbox waves.
    assert!(validate(&[0xd8380100, 0x00020100]).is_err());
}

#[test]
fn class5_mesh_pointer_is_eight_above_the_user_location() {
    assert_class5_mesh_pointer();
}

#[test]
fn class5_descriptor_load_uses_the_guest_heap_index() {
    assert_class5_descriptor_load();
}

/// One register holds both output counts, so the compiler checks it once and
/// combines each mask with itself. Assembled with llvm-mc for wave64.
#[test]
fn one_count_for_vertices_and_primitives_is_recognized() {
    let words: [u32; 70] = [
        0x7e3c0280, 0xbf068014, 0x8580807e, 0xb5940080, 0x8582807e, 0x88800000,
        0x88820202, 0x88800002, 0xd5010001, 0x00010280, 0x7e000501, 0x87000081,
        0xbf068100, 0xbe80047e, 0x8582807e, 0xd5010001, 0x00090214, 0x8984c102,
        0x7e0c0501, 0x7da40080, 0xbf88000d, 0x8f078c06, 0x8aea027e, 0x88060607,
        0xbefc0306, 0xbf900009, 0xbf8c0000, 0xbf870006, 0xbf8a0000, 0x7e027081,
        0xf80008c0, 0x00000000, 0xf8000941, 0x00000001, 0xbf8cff0f, 0xbefe0400,
        0xbf8a0000, 0x7d880014, 0x8780046a, 0xbe862400, 0xbf880012, 0xd5480001,
        0x02291513, 0xd5480002, 0x02292913, 0x363026ff, 0x000003ff, 0x7d860214,
        0xd4c30002, 0x00020414, 0xd4c30000, 0x00023014, 0x8882026a, 0x88800200,
        0xd5010001, 0x0001ff13, 0x80000000, 0xf8000941, 0x00000001, 0xbf8cff0f,
        0xbefe0406, 0x7d880014, 0x8780046a, 0xbe802400, 0xbf880004, 0xf80008cf,
        0x33323130, 0xf800020f, 0x1211100f, 0xbf810000,
    ];
    let recognize = |words: &[u32]| {
        let bytes = words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>();
        super::super::mesh_exports::recognize(&disassemble(&bytes), 128, None)
    };
    let found = recognize(&words).unwrap_or_else(|reason| panic!("{reason}"));
    assert_eq!((found.vertices, found.primitives), (Opnd::Sgpr(20), Opnd::Sgpr(20)));
    assert_eq!(found.lane_index, Opnd::Vgpr(0));
    assert_eq!((found.primitive_exports.len(), found.vertex_exports.len()), (0, 2));
    // A larger count limit is a different program.
    let mut limit = words;
    assert_eq!(limit[3], 0xb5940080);
    limit[3] = 0xb5940081;
    assert!(recognize(&limit).is_err());
    // The self-combinations must be of the mask itself.
    let mut mixed = words;
    assert_eq!(mixed[5], 0x88800000);
    mixed[5] = 0x88800200;
    assert!(recognize(&mixed).is_err());
}
