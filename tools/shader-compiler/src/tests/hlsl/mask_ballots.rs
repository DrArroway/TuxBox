use super::*;
use crate::{hlsl::translate, ShaderExtract, Stage};

mod fixture {
    include!("../../../examples/support/wave_guard_fixture.rs");
}

fn insn(mnemonic: &str, enc: Enc, dst: Vec<Opnd>, src: Vec<Opnd>) -> Insn {
    Insn { offset: 0, dwords: 1, enc, op: 0, raw: vec![0], mnemonic: mnemonic.into(), dst, src, aux: Default::default() }
}

fn program(mut ops: Vec<Insn>) -> Vec<Insn> {
    // Something that tells the wave's width.
    ops.insert(0, scalar("s_mov_b32", 100, vec![Opnd::ExecLo]));
    ops.push(Insn { op: 1, ..insn("s_endpgm", Enc::Sopp, vec![], vec![]) });
    for (i, op) in ops.iter_mut().enumerate() { op.offset = i * 4; }
    ops
}

fn scalar(mnemonic: &str, dst: u8, src: Vec<Opnd>) -> Insn {
    insn(mnemonic, if src.len() == 1 { Enc::Sop1 } else { Enc::Sop2 }, vec![Opnd::Sgpr(dst)], src)
}

fn compare() -> Insn {
    insn("v_cmp_gt_u32", Enc::Vopc, vec![Opnd::VccLo], vec![Opnd::Vgpr(0), Opnd::Vgpr(1)])
}

fn count(source: Opnd) -> Insn { scalar("s_bcnt1_i32_b32", 20, vec![source]) }

fn branch(op: u32, over: i64) -> Insn {
    let mut branch = insn("s_cbranch", Enc::Sopp, vec![], vec![]);
    (branch.op, branch.aux.imm) = (op, over);
    branch
}

/// Index of the definitions a program's counts take their ballots from.
fn ballots(ops: Vec<Insn>) -> Option<Vec<(usize, u8)>> {
    let found = find(&program(ops))?;
    Some(found.defs.iter().flat_map(|(&at, regs)| regs.iter().map(move |&r| (at, r))).collect())
}

#[test]
fn a_mask_is_captured_where_it_is_computed() {
    assert_eq!(ballots(vec![compare(), count(Opnd::VccLo)]), Some(vec![(1, 106)]));
    // Through copies and EXEC, past the point where EXEC is narrowed.
    let saved = ballots(vec![
        compare(),
        scalar("s_and_saveexec_b32", 10, vec![Opnd::VccLo]),
        scalar("s_mov_b32", 11, vec![Opnd::ExecLo]),
        insn("v_cmpx_eq_u32", Enc::Vopc, vec![Opnd::ExecLo], vec![Opnd::Sgpr(9), Opnd::Vgpr(1)]),
        count(Opnd::Sgpr(11)),
    ]);
    assert_eq!(saved, Some(vec![(3, 11)]));
    // Both of the masks that can reach the count, one of them empty.
    let gathered = ballots(vec![
        scalar("s_mov_b32", 11, vec![Opnd::ConstInt(0)]),
        compare(),
        scalar("s_or_b32", 11, vec![Opnd::Sgpr(11), Opnd::VccLo]),
        count(Opnd::Sgpr(11)),
        branch(5, -4),
    ]);
    assert_eq!(gathered, Some(vec![(3, 11)]));
}

#[test]
fn numbers_are_not_masks() {
    for ops in [
        vec![scalar("s_mov_b32", 11, vec![Opnd::Sgpr(0)]), count(Opnd::Sgpr(11))],
        vec![scalar("s_mov_b32", 11, vec![Opnd::Literal(0xff)]), count(Opnd::Sgpr(11))],
        vec![scalar("s_add_u32", 11, vec![Opnd::Sgpr(0), Opnd::Sgpr(1)]), count(Opnd::Sgpr(11))],
        // Every lane, running or not.
        vec![scalar("s_mov_b32", 11, vec![Opnd::ConstInt(-1)]), count(Opnd::Sgpr(11))],
        // Part number.
        vec![compare(), scalar("s_and_b32", 11, vec![Opnd::VccLo, Opnd::Sgpr(0)]), count(Opnd::Sgpr(11))],
        // A mask on one path only.
        vec![
            scalar("s_mov_b32", 11, vec![Opnd::Sgpr(0)]),
            branch(4, 2),
            compare(),
            scalar("s_mov_b32", 11, vec![Opnd::VccLo]),
            count(Opnd::Sgpr(11)),
        ],
    ] {
        assert_eq!(ballots(ops), None);
    }
}

#[test]
fn counts_read_the_ballot() {
    // v_cmp_lt_u32 vcc, 1000, v0; s_bcnt1 s20, vcc; v_mov v10, s20
    let words = |bcnt: u32| vec![0x7d82_00ff, 1000, bcnt, 0x7e14_0214];
    for (width, bcnt, save) in [(32, 0xbe94_0f6a, 0xbe8c_037eu32), (64, 0xbe94_106a, 0xbe8c_047e)] {
        let (mut ex, _, _): (ShaderExtract, _, _) = fixture::shader(width);
        assert!(matches!(ex.stage, Stage::Compute));
        let words = [vec![save], words(bcnt), vec![0xbf81_0000]].concat();
        ex.isa = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        ex.isa_data = ex.isa.clone();
        let shader = translate(&ex).unwrap();
        assert!(shader.hlsl.contains("WaveActiveBallot((vcc_lo & 1u) != 0u)"), "{}", shader.hlsl);
        let wanted = if width == 32 { "countbits(tb_bal106)" } else { "(countbits(tb_bal106) + countbits(tb_bal107))" };
        assert!(shader.hlsl.contains(wanted), "{}", shader.hlsl);
        // Two mask words hold a wave of any size, one holds 32 lanes.
        assert_eq!(shader.required_wave_size, (width == 32).then_some(32));
        assert_eq!(shader.profile, "cs_6_5");
    }
}
