//! Emit synthetic RDNA2 constant-lane `v_readlane_b32` compute shaders through
//! the real HLSL translator for DXC/DXV validation and D3D12 GPU readback.
use tuxbox_xshader::{hlsl, ShaderExtract, Stage};

fn shader(group: [u32; 3], words: &[u32]) -> ShaderExtract {
    let isa: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ShaderExtract {
        stage: Stage::Compute,
        shader_model: (6, 0),
        required_features: None,
        bitcode_stripped: true,
        fspx_config: None,
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

const fn readlane(dst: u8, vsrc: u8, lane: u32) -> [u32; 2] {
    [
        0xd760_0000 | dst as u32,
        ((128 + lane) << 9) | 256 | vsrc as u32,
    ]
}

const fn writelane(vdst: u8, ssrc: u8, lane: u32) -> [u32; 2] {
    [
        0xd761_0000 | vdst as u32,
        ((128 + lane) << 9) | ssrc as u32,
    ]
}

fn instrument(hlsl: &str, lanes: u32, end_marker: &str, result_reg: &str) -> String {
    let init_src = "    v0 = gtid.x; v1 = gtid.y; v2 = gtid.z;";
    // Load per-lane v1 value from output[gtid.x] so the host verifies exact
    // arbitrary bit patterns (not predictable thread IDs alone).
    let init_dst = "    v0 = gtid.x; v1 = output[gtid.x]; v2 = gtid.z;";
    assert_eq!(hlsl.matches(init_src).count(), 1, "missing init in:\n{hlsl}");
    assert_eq!(hlsl.matches(end_marker).count(), 1, "missing {end_marker} in:\n{hlsl}");
    let store = if lanes <= 64 {
        format!("    output[{lanes}u + gtid.x] = {result_reg};\n{end_marker}")
    } else {
        format!("    GroupMemoryBarrierWithGroupSync();\n    output[gtid.x] = {result_reg};\n{end_marker}")
    };
    let body = hlsl
        .replace(init_src, init_dst)
        .replace(end_marker, &store);
    format!("RWBuffer<uint> output : register(u0);\n{body}")
}

fn main() {
    let case = std::env::args().nth(1).expect("case name");
    let (group, words, result_reg, expect_wave32): ([u32; 3], Vec<u32>, &str, bool) = match case.as_str() {
        "wave32_active_0" => {
            let [r0, r1] = readlane(7, 1, 0);
            ([32, 1, 1], vec![r0, r1, 0xbf81_0000], "s7", true)
        }
        "wave32_active_15" => {
            let [r0, r1] = readlane(7, 1, 15);
            ([32, 1, 1], vec![r0, r1, 0xbf81_0000], "s7", true)
        }
        "wave32_active_31" => {
            let [r0, r1] = readlane(7, 1, 31);
            ([32, 1, 1], vec![r0, r1, 0xbf81_0000], "s7", true)
        }
        "wave32_masked_31" => {
            // Mask lane 31 OFF in EXEC, overwrite active lanes' v1 with sentinel
            // 0xdead0000, skip lane 31 via s_cbranch_execz, then read lane 31 of v1.
            // Restore EXEC afterwards so ALL 32 lanes store s7 (or store s7 directly).
            let [r0, r1] = readlane(7, 1, 31);
            (
                [32, 1, 1],
                vec![
                    0x7e04_029f,             // v_mov_b32 v2, 31
                    0x7d82_0500,             // v_cmp_lt_u32 vcc, v0, v2
                    0xbe84_3c6a,             // s_and_saveexec_b32 s4, vcc_lo (lane 31 EXEC = 0)
                    0x7e02_02ff, 0xdead0000, // v_mov_b32 v1, 0xdead0000 (lane 31 keeps input!)
                    r0, r1,                  // v_readlane_b32 s7, v1, 31 (MUST read pre-mask v1!)
                    0xbefe_0304,             // s_mov_b32 exec_lo, s4 (restore EXEC)
                    0xbf81_0000,
                ],
                "s7",
                false,
            )
        }
        "wave32_masked_branch_31" => {
            // Same as above, PLUS s_cbranch_execz skips lane 31 around the readlane!
            // Active lanes 0..30 MUST still receive lane 31's pre-mask v1 in s7;
            // lane 31 took the skip branch so its s7 stays 0.
            let [r0, r1] = readlane(7, 1, 31);
            (
                [32, 1, 1],
                vec![
                    0xbe87_0380,             // s_mov_b32 s7, 0
                    0x7e04_029f,             // v_mov_b32 v2, 31
                    0x7d82_0500,             // v_cmp_lt_u32 vcc, v0, v2
                    0xbe84_3c6a,             // s_and_saveexec_b32 s4, vcc_lo
                    0x7e02_02ff, 0xdead0000, // v_mov_b32 v1, 0xdead0000
                    0xbf88_0002,             // s_cbranch_execz +2 (skip r0, r1)
                    r0, r1,                  // v_readlane_b32 s7, v1, 31
                    0xbefe_0304,             // s_mov_b32 exec_lo, s4
                    0xbf81_0000,
                ],
                "s7",
                false,
            )
        }
        "wave32_64t_masked_31" => {
            // 64 threads, Wave32 (two independent 32-lane waves):
            // Wave 0 (0..31) reads thread 31's pre-mask v1; Wave 1 (32..63) reads thread 63's!
            let [r0, r1] = readlane(7, 1, 31);
            (
                [64, 1, 1],
                vec![
                    0x3604_009f,             // v_and_b32 v2, 31, v0 (lane in wave32)
                    0x7e06_029f,             // v_mov_b32 v3, 31
                    0x7d82_0702,             // v_cmp_lt_u32 vcc, v2, v3 (lane 31 of EACH wave -> 0)
                    0xbe84_3c6a,             // s_and_saveexec_b32 s4, vcc_lo
                    0x7e02_02ff, 0xbad00000, // v_mov_b32 v1, 0xbad00000
                    r0, r1,                  // v_readlane_b32 s7, v1, 31
                    0xbefe_0304,             // s_mov_b32 exec_lo, s4
                    0xbf81_0000,
                ],
                "s7",
                false,
            )
        }
        "wave64_64t_lane31" => {
            let [r0, r1] = readlane(7, 1, 31);
            ([64, 1, 1], vec![r0, r1, 0xbf81_0000], "s7", false)
        }
        "wave64_64t_lane47" => {
            let [r0, r1] = readlane(7, 1, 47);
            ([64, 1, 1], vec![r0, r1, 0xbf81_0000], "s7", false)
        }
        "wave64_64t_masked_lane63" => {
            let [r0, r1] = readlane(7, 1, 63);
            (
                [64, 1, 1],
                vec![
                    0x7e04_02bf,             // v_mov_b32 v2, 63
                    0x7d82_0500,             // v_cmp_lt_u32 vcc, v0, v2
                    0xbe84_246a,             // s_and_saveexec_b64 s[4:5], vcc (lane 63 EXEC = 0)
                    0x7e02_02ff, 0xdeadbeef, // v_mov_b32 v1, 0xdeadbeef (lane 63 keeps input!)
                    r0, r1,                  // v_readlane_b32 s7, v1, 63
                    0xbefe_0404,             // s_mov_b64 exec, s[4:5]
                    0xbf81_0000,
                ],
                "s7",
                false,
            )
        }
        "wave64_128t_lane31" => {
            let [r0, r1] = readlane(7, 1, 31);
            ([128, 1, 1], vec![r0, r1, 0xbf81_0000], "s7", false)
        }
        "wave64_128t_lane63" => {
            let [r0, r1] = readlane(7, 1, 63);
            ([128, 1, 1], vec![r0, r1, 0xbf81_0000], "s7", false)
        }
        "wave64_64t_spill_and_loop_lane31" => {
            // SGPR spill on v72 + uniform loop reading lane 31 of v1, adding spilled value.
            let [wl0, wl1] = writelane(72, 8, 0);
            let [rl_s0, rl_s1] = readlane(9, 72, 0);
            let [rl_l0, rl_l1] = readlane(7, 1, 31);
            (
                [64, 1, 1],
                vec![
                    0xbe88_038a,          // s_mov_b32 s8, 10
                    wl0, wl1,             // v_writelane_b32 v72, s8, 0
                    0xbe8a_0383,          // s_mov_b32 s10, 3
                    rl_l0, rl_l1,         // L_loop: v_readlane_b32 s7, v1, 31
                    0x818a_810a,          // s_sub_i32 s10, s10, 1
                    0xbf07_800a,          // s_cmp_lg_u32 s10, 0
                    0xbf85_fffb,          // s_cbranch_scc1 L_loop
                    rl_s0, rl_s1,         // v_readlane_b32 s9, v72, 0
                    0x8007_0907,          // s_add_u32 s7, s7, s9 (lane31(v1) + 10)
                    0xbf81_0000,
                ],
                "s7",
                false,
            )
        }
        other => panic!("unknown readlane probe case: {other}"),
    };
    let end_off = (words.len() - 1) * 4;
    let end_marker = format!("    // {end_off:#06x} s_endpgm");
    let tr = hlsl::translate(&shader(group, &words)).unwrap_or_else(|e| panic!("{case}: {e}"));
    if expect_wave32 {
        assert_eq!(tr.required_wave_size, Some(32), "{case} must use WaveSize(32)");
    } else {
        assert_eq!(tr.required_wave_size, None, "{case} must be portable (no WaveSize)");
    }
    print!("{}", instrument(&tr.hlsl, group[0], &end_marker, result_reg));
}
