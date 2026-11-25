use super::*;

fn recognizes(words: &[u32], group: [u32; 3]) -> bool {
    let isa: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    prepare(group, &mut crate::rdna2::disassemble(&isa))
}

#[test]
fn partial_group_reads_only_proven_launched_lanes() {
    // In a 7x3 group, registers and inline operands can select lanes 0..20.
    const SET: u32 = 0xbe98_0394; // s_mov_b32 s24,20
    const READ: [u32; 2] = [0xd760_001c, 0x0000_3109]; // s28,v9,s24
    const END: u32 = 0xbf81_0000;
    let check = |words: &[u32]| {
        let isa: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut ops = crate::rdna2::disassemble(&isa);
        let before = ops.iter().map(|op| op.mnemonic.clone()).collect::<Vec<_>>();
        let accepted = constant_reads([7, 3, 1], &mut ops, true);
        if accepted {
            assert!(ops.iter().any(|op| op.mnemonic == "tb_indexed_readlane"));
            assert!(!ops.iter().any(|op| op.mnemonic == "v_readlane_b32"));
        } else {
            assert!(ops.iter().map(|op| &op.mnemonic).eq(&before));
        }
        accepted
    };
    assert!(check(&[SET, READ[0], READ[1], END]));
    assert!(!check(&[SET + 1, READ[0], READ[1], END]));
    assert!(check(&[READ[0], 0x0001_2909, END])); // inline lane 20
    assert!(!check(&[READ[0], 0x0001_2b09, END])); // inline lane 21
    // Equal incoming values at a forward branch join prove the selector.
    assert!(check(&[SET, 0xbf84_0001, SET, READ[0], READ[1], END]));
    assert!(!check(&[SET, 0xbf84_0001, SET + 1, READ[0], READ[1], END]));
    assert!(check(&[SET, 0xbf84_0001, SET - 1, READ[0], READ[1], END]));
    // Keep the runtime selector when both CSELECT operands are in range.
    assert!(check(&[0x8518_8a94, 0xbe99_0318, READ[0], 0x0000_3309, END]));
    assert!(!check(&[0x8518_9594, READ[0], READ[1], END])); // 20 or 21
    assert!(!check(&[0x8518_0594, READ[0], READ[1], END])); // 20 or unknown
    // CMOV keeps the previous destination when SCC is clear.
    const CMOV_20: u32 = 0xbe98_0594; // s_cmov_b32 s24,20
    assert!(check(&[SET - 10, CMOV_20, READ[0], READ[1], END]));
    assert!(!check(&[CMOV_20, READ[0], READ[1], END]));
    assert!(!check(&[SET + 1, CMOV_20, READ[0], READ[1], END]));
    assert!(!check(&[SET, CMOV_20 + 1, READ[0], READ[1], END]));
    assert!(!check(&[SET, 0xbe98_0505, READ[0], READ[1], END])); // source s5 unknown
    assert!(check(&[0x8718_8f05, READ[0], READ[1], END])); // s5 & 15
    assert!(!check(&[0x8718_9f05, READ[0], READ[1], END])); // s5 & 31
    assert!(!check(&[0x8718_8f05, 0x8018_a018, READ[0], READ[1], END])); // add 32
    // The first read overwrites the selector with the selected VGPR value.
    assert!(!check(&[SET, 0xd760_0018, READ[1], READ[0], READ[1], END]));
}

#[test]
fn partial_group_selector_tracks_each_word_of_a_scalar_pair() {
    const OLD: u32 = 0xbe99_038a; // s_mov_b32 s25,10
    const NEW: u32 = 0xbe9b_0394; // s_mov_b32 s27,20
    const MOV: u32 = 0xbe98_041a; // s_mov_b64 s[24:25],s[26:27]
    const CMOV: u32 = 0xbe98_061a; // conditional copy of the same pair
    const SELECT: u32 = 0x8598_181a; // select s[26:27] or s[24:25]
    let reads_high = |setup: &[u32]| {
        let words = [setup, &[0xd760_001c, 0x0000_3309, 0xbf81_0000]].concat();
        let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        constant_reads([7, 3, 1], &mut crate::rdna2::disassemble(&bytes), true)
    };
    assert!(reads_high(&[NEW, MOV]));
    assert!(!reads_high(&[OLD, MOV])); // unknown source overwrites known high word
    for op in [CMOV, SELECT] {
        assert!(reads_high(&[OLD, NEW, op])); // low words are unknown
        assert!(!reads_high(&[NEW, op]));
        assert!(!reads_high(&[OLD, op]));
        assert!(!reads_high(&[OLD, NEW + 1, op]));
        assert!(!reads_high(&[OLD + 11, NEW, op]));
    }
    assert!(reads_high(&[0xbe98_0487])); // high word of inline 7 is zero
    assert!(!reads_high(&[OLD, 0xbe98_04c1])); // high word of -1 is out of range
}

#[test]
fn empty_wave_fallback_cannot_be_entered_from_another_edge() {
    let pruned = |words: &[u32]| {
        let isa: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut ops = crate::rdna2::disassemble(&isa);
        skip_empty_wave_fallback(&mut ops)
            && !ops.iter().any(|op| op.mnemonic == "v_readlane_b32")
    };
    let words = [
        0xbf800000, 0xbeea037e, 0xbf870002, 0xd760000b, 0x00000d01, 0xbf800000, 0xbf810000,
    ];
    assert!(pruned(&words));
    for (index, word) in [
        (0, 0xbefe0380),
        (1, 0xbeea0380),
        (2, 0xbf860002),
        (5, 0xbf82fffd),
        (5, 0xbf82fffb),
        (5, 0xbe802000),
    ] {
        let mut changed = words;
        changed[index] = word;
        assert!(!pruned(&changed));
    }
}

#[test]
fn lane_election_requires_an_intact_full_wave_predicate() {
    let words = [
        0xd4850004, 0x00010081, // v_cmp_ne_i32 s[4:5], 1, 0
        0xbf800000, // s_nop
        0xbed01404, // s_ff1_i32_b64 s80, s[4:5]
        0xd766000c, 0x00020cc1, // v_mbcnt_hi_u32_b32
        0xd7600000, 0x0000a104, // v_readlane_b32 s0, v4, s80
        0xbf810000,
    ];
    assert!(recognizes(&words, [16, 16, 1]));
    assert!(!recognizes(&words, [32, 1, 1]));
    for change in [0xbe850380, 0xbf820001, 0xbefe047e] {
        let mut changed = words;
        changed[2] = change; // clobber mask high, branch, write EXEC
        assert!(!recognizes(&changed, [16, 16, 1]));
    }
    let mut equal = words;
    equal[1] = 0x00010080; // empty mask
    assert!(!recognizes(&equal, [16, 16, 1]));
}

fn compute_shader(group: [u32; 3], words: &[u32], fspx_config: Option<[u32; 7]>) -> crate::ShaderExtract {
    let isa: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    crate::ShaderExtract {
        stage: crate::Stage::Compute,
        shader_model: (6, 0),
        required_features: None,
        bitcode_stripped: true,
        fspx_config,
        ps09_pixel_inputs: None,
        fssm: Vec::new(),
        entry: Some("main".into()),
        isa_data: isa.clone(),
        isa,
        isa_offset_in_fssm: 0,
        psv_threads: Some(group),
        flat_inputs: Vec::new(),
        flat_outputs: Vec::new(),
        isa_preamble: [0, group[0], group[1], group[2]],
        has_input_sig: false,
        has_output_sig: false,
        has_psv0: false,
        has_root_sig: false,
        input_sig: Vec::new(),
        output_sig: Vec::new(),
        resources: Vec::new(),
        gs_info: None,
        mesh_info: None,
        primitive_sig: Vec::new(),
        root_sig: None,
    }
}

const fn readlane_const(dst: u8, vsrc: u8, lane: u32) -> [u32; 2] {
    [
        0xd760_0000 | dst as u32,
        ((128 + lane) << 9) | 256 | vsrc as u32,
    ]
}

const fn readlane_sgpr(dst: u8, vsrc: u8, s_lane: u8) -> [u32; 2] {
    [
        0xd760_0000 | dst as u32,
        ((s_lane as u32) << 9) | 256 | vsrc as u32,
    ]
}

const fn writelane_const(vdst: u8, ssrc: u8, lane: u32) -> [u32; 2] {
    [
        0xd761_0000 | vdst as u32,
        ((128 + lane) << 9) | ssrc as u32,
    ]
}

#[test]
fn native_wave32_constant_reads_use_wave_read_lane_at_when_all_lanes_active() {
    for lane in [0u32, 15, 31] {
        let [w0, w1] = readlane_const(7, 1, lane);
        let words = [
            0x7e02_0285, // v_mov_b32 v1, 5
            w0, w1,      // v_readlane_b32 s7, v1, lane
            0x7e04_0207, // v_mov_b32 v2, s7
            0xbf81_0000, // s_endpgm
        ];
        let ex = compute_shader([32, 1, 1], &words, None);
        let tr = crate::hlsl::translate(&ex).expect("native wave32 constant readlane");
        assert_eq!(tr.required_wave_size, Some(32));
        assert!(tr.hlsl.contains("[WaveSize(32)]"), "{}", tr.hlsl);
        assert!(
            tr.hlsl.contains(&format!("s7 = WaveReadLaneAt(v1, {lane}u);")),
            "{}",
            tr.hlsl
        );
        assert!(!tr.hlsl.contains("groupshared uint tb_swizzle"), "{}", tr.hlsl);
        assert!(!tr.hlsl.contains("GroupMemoryBarrierWithGroupSync"), "{}", tr.hlsl);
    }
}

#[test]
fn wave32_masked_exec_readlane_stages_through_groupshared_and_ignores_exec() {
    // Lane 31 writes v1, then v_cmpx_lt_u32 disables lane 31 in EXEC,
    // remaining lanes overwrite v1, then v_readlane_b32 reads lane 31 of v1.
    let [r0, r1] = readlane_const(7, 1, 31);
    let words = [
        0x7e02_0300, // v_mov_b32 v1, v0
        0x7e04_029f, // v_mov_b32 v2, 31
        0x7da2_0500, // v_cmpx_lt_u32 exec, v0, v2 (lane 31 -> EXEC = 0)
        0x7e02_0280, // v_mov_b32 v1, 0 (active lanes overwrite v1; lane 31 keeps 31)
        0xbf88_0002, // s_cbranch_execz +2 (lane 31 skips past readlane)
        r0, r1,      // v_readlane_b32 s7, v1, 31
        0x7e06_0207, // v_mov_b32 v3, s7
        0xbf81_0000, // s_endpgm
    ];
    let ex = compute_shader([32, 1, 1], &words, None);
    let tr = crate::hlsl::translate(&ex).expect("masked wave32 readlane");
    assert_eq!(tr.required_wave_size, None);
    assert!(tr.hlsl.contains("groupshared uint tb_swizzle[32];"), "{}", tr.hlsl);
    assert!(tr.hlsl.contains("tb_swizzle[tb_lane] = v1;"), "{}", tr.hlsl);
    assert!(
        tr.hlsl.contains("tb_swizzle[(tb_lane & ~31u) | (0x1fu & 31u)]"),
        "{}",
        tr.hlsl
    );
    assert!(!tr.hlsl.contains("WaveReadLaneAt("), "{}", tr.hlsl);
    // Staging must be hoisted BEFORE the skip guard `if (!(j...))`.
    let stage_pos = tr.hlsl.find("tb_swizzle[tb_lane] = v1;").unwrap();
    let guard_pos = tr.hlsl.find("if (!(j").unwrap();
    assert!(stage_pos < guard_pos, "staging must precede skip guard:\n{}", tr.hlsl);
}

#[test]
fn wave32_multiwave_group_uses_32_lane_stride_and_restored_exec_uses_native() {
    // 64 threads with s_and_saveexec_b32 (two 32-lane waves):
    // First readlane happens while EXEC is narrowed -> groupshared stride 32.
    // Second readlane happens after s_mov_b32 exec_lo, s4 restores full EXEC -> native WaveReadLaneAt.
    let [r_masked0, r_masked1] = readlane_const(7, 1, 31);
    let [r_full0, r_full1] = readlane_const(8, 1, 15);
    let words = [
        0x7e02_0300, // v_mov_b32 v1, v0
        0x7e04_029f, // v_mov_b32 v2, 31
        0x7d82_0500, // v_cmp_lt_u32 vcc, v0, v2
        0xbe84_3c6a, // s_and_saveexec_b32 s4, vcc_lo
        r_masked0, r_masked1, // v_readlane_b32 s7, v1, 31 (EXEC narrowed)
        0xbefe_0304, // s_mov_b32 exec_lo, s4 (restores full EXEC)
        r_full0, r_full1,     // v_readlane_b32 s8, v1, 15 (EXEC full again)
        0xbf81_0000,
    ];
    let ex = compute_shader([64, 1, 1], &words, None);
    let tr = crate::hlsl::translate(&ex).expect("multi-wave wave32 readlane");
    assert_eq!(tr.required_wave_size, Some(32));
    assert!(tr.hlsl.contains("[WaveSize(32)]"), "{}", tr.hlsl);
    assert!(tr.hlsl.contains("groupshared uint tb_swizzle[64];"), "{}", tr.hlsl);
    assert!(
        tr.hlsl.contains("tb_swizzle[(tb_lane & ~31u) | (0x1fu & 31u)]"),
        "{}",
        tr.hlsl
    );
    assert!(tr.hlsl.contains("s8 = WaveReadLaneAt(v1, 15u);"), "{}", tr.hlsl);
}

#[test]
fn portable_wave64_constant_reads_cross_wave32_boundary_without_wavesize64() {
    for group in [[64, 1, 1], [128, 1, 1]] {
        let [a0, a1] = readlane_const(6, 1, 31);
        let [b0, b1] = readlane_const(7, 1, 47);
        let [c0, c1] = readlane_const(8, 1, 63);
        let words = [
            0x7e02_0300, // v_mov_b32 v1, v0
            a0, a1,
            b0, b1,
            c0, c1,
            0xbf81_0000,
        ];
        let ex = compute_shader(group, &words, None);
        let tr = crate::hlsl::translate(&ex).expect("portable wave64 readlane");
        assert_eq!(tr.required_wave_size, None);
        assert!(!tr.hlsl.contains("[WaveSize(64)]"), "{}", tr.hlsl);
        assert!(
            tr.hlsl.contains(&format!("groupshared uint tb_swizzle[{}];", group[0])),
            "{}",
            tr.hlsl
        );
        assert!(tr.hlsl.contains("tb_swizzle[(tb_lane & ~63u) | (0x1fu & 63u)]"), "{}", tr.hlsl);
        assert!(tr.hlsl.contains("tb_swizzle[(tb_lane & ~63u) | (0x2fu & 63u)]"), "{}", tr.hlsl);
        assert!(tr.hlsl.contains("tb_swizzle[(tb_lane & ~63u) | (0x3fu & 63u)]"), "{}", tr.hlsl);
    }
}

#[test]
fn constant_readlane_coexists_with_sgpr_spills_and_uniform_loops() {
    // Models a shader with SGPR spills on v72, a uniform loop containing
    // v_readlane_b32 s7, v1, 31, and spill restore from v72.
    let [wl0, wl1] = writelane_const(72, 8, 0);
    let [rl_spill0, rl_spill1] = readlane_const(9, 72, 0);
    let [rl_lane0, rl_lane1] = readlane_const(7, 1, 31);
    let words = [
        0xbe88_0383,          // s_mov_b32 s8, 3
        wl0, wl1,             // v_writelane_b32 v72, s8, 0
        0xbe8a_0384,          // s_mov_b32 s10, 4 (uniform loop counter)
        0x7e02_0300,          // L_loop: v_mov_b32 v1, v0
        rl_lane0, rl_lane1,   // v_readlane_b32 s7, v1, 31
        0x818a_810a,          // s_sub_i32 s10, s10, 1
        0xbf07_800a,          // s_cmp_lg_u32 s10, 0
        0xbf85_fffa,          // s_cbranch_scc1 L_loop (-6 dwords = 24 bytes back to 0x10)
        rl_spill0, rl_spill1, // v_readlane_b32 s9, v72, 0
        0xbf81_0000,
    ];
    let ex = compute_shader([64, 1, 1], &words, None);
    let tr = crate::hlsl::translate(&ex).expect("spill + uniform loop + readlane");
    assert!(tr.hlsl.contains("tb_s10000"), "{}", tr.hlsl);
    assert!(tr.hlsl.contains("tb_swizzle[(tb_lane & ~63u) | (0x1fu & 63u)]"), "{}", tr.hlsl);
}

#[test]
fn refuses_out_of_range_unproven_dynamic_and_non_uniform_loop_readlanes() {
    // 1. Out-of-range constant lane 32 in a 32-thread group.
    let [o32_0, o32_1] = readlane_const(7, 1, 32);
    let err = crate::hlsl::translate(&compute_shader([32, 1, 1], &[o32_0, o32_1, 0xbf81_0000], None))
        .err()
        .expect("lane 32 on 32-thread group must be refused");
    assert!(err.contains("out of range"), "{err}");

    // 2. Out-of-range constant lane 64 in a 64-thread group.
    let [o64_0, o64_1] = readlane_const(7, 1, 64);
    let err = crate::hlsl::translate(&compute_shader([64, 1, 1], &[o64_0, o64_1, 0xbf81_0000], None))
        .err()
        .expect("lane 64 on 64-thread group must be refused");
    assert!(err.contains("out of range"), "{err}");

    // 3. Lane 32 in a Wave32 64-thread group (s_and_saveexec_b32 present).
    let err = crate::hlsl::translate(&compute_shader(
        [64, 1, 1],
        &[0xbe84_3c6a, o32_0, o32_1, 0xbf81_0000],
        None,
    ))
    .err()
    .expect("lane 32 in wave32 shader must be refused");
    assert!(err.contains("conflicting wave32 and wave64"), "{err}");

    // 4. Unproven dynamic SGPR lane selector in both 32- and 64-thread groups.
    let [dyn0, dyn1] = readlane_sgpr(7, 1, 5);
    for group in [[32, 1, 1], [64, 1, 1]] {
        let err = crate::hlsl::translate(&compute_shader(group, &[dyn0, dyn1, 0xbf81_0000], None))
            .err()
            .expect("unproven dynamic SGPR selector must be refused");
        assert!(err.contains("upper-bound proof"), "{err}");
    }

    // 5. Dynamic SGPR selector masked above limit (`s5 & 63` in 32-thread group).
    let err = crate::hlsl::translate(&compute_shader(
        [32, 1, 1],
        &[0x8705_bf06, dyn0, dyn1, 0xbf81_0000], // s_and_b32 s5, s6, 63
        None,
    ))
    .err()
    .expect("upper bound 63 on 32-thread group must be refused");
    assert!(err.contains("exceeds limit 32"), "{err}");

    // 6. Non-uniform loop (s_cbranch_vccnz back-edge) enclosing v_readlane_b32.
    let [r0, r1] = readlane_const(7, 1, 31);
    let err = crate::hlsl::translate(&compute_shader(
        [64, 1, 1],
        &[
            0x7e02_0300, // L_loop: v_mov_b32 v1, v0
            r0, r1,      // v_readlane_b32 s7, v1, 31
            0xbf87_fffc, // s_cbranch_vccnz L_loop (-4 dwords)
            0xbf81_0000,
        ],
        None,
    ))
    .err()
    .expect("readlane inside non-uniform VCC loop must be refused");
    assert!(err.contains("non-uniform loop"), "{err}");

    // 7. Non-uniform loop whose SCC depends on v_readlane_b32 itself.
    let err = crate::hlsl::translate(&compute_shader(
        [64, 1, 1],
        &[
            0x7e02_0300, // L_loop: v_mov_b32 v1, v0
            r0, r1,      // v_readlane_b32 s7, v1, 31
            0xbf07_8007, // s_cmp_lg_u32 s7, 0 (tainted SCC!)
            0xbf85_fffb, // s_cbranch_scc1 L_loop (-5 dwords)
            0xbf81_0000,
        ],
        None,
    ))
    .err()
    .expect("readlane inside lane-dependent SCC loop must be refused");
    assert!(err.contains("non-uniform loop"), "{err}");

    // 8. Early s_endpgm before v_readlane_b32.
    let err = crate::hlsl::translate(&compute_shader(
        [64, 1, 1],
        &[0xbf88_0001, 0xbf81_0000, r0, r1, 0xbf81_0000],
        None,
    ))
    .err()
    .expect("readlane after early s_endpgm must be refused");
    assert!(err.contains("early s_endpgm"), "{err}");

    // 9. Spill VGPR read of unwritten lane still refused by lower_lane_spills.
    let [wl0, wl1] = writelane_const(72, 8, 0);
    let [bad_rl0, bad_rl1] = readlane_const(9, 72, 1);
    let err = crate::hlsl::translate(&compute_shader(
        [64, 1, 1],
        &[wl0, wl1, bad_rl0, bad_rl1, 0xbf81_0000],
        None,
    ))
    .err()
    .expect("unwritten spill lane must stay refused");
    assert!(err.contains("v72 lane 1 read before any v_writelane"), "{err}");
}
