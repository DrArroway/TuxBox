//! Lane masks read as numbers.
//!
//! One invocation holds only its own bit of a lane mask. Counting a mask's lanes, finding
//! the first one or counting those before this lane reads all of them, often after EXEC has
//! been narrowed to one elected lane. Take the ballot where the mask is computed, while
//! every lane that can be in it is there, and let each lane keep it.
use super::{Emitter, Enc, Insn, Opnd};
use std::collections::{BTreeMap, BTreeSet};

const VCC: u8 = 106;

/// The scalar register an operand names. VCC follows the ordinary registers.
pub(super) fn register(o: &Opnd) -> Option<u8> {
    match *o {
        Opnd::Sgpr(n) => Some(n),
        Opnd::VccLo => Some(VCC),
        Opnd::VccHi => Some(VCC + 1),
        _ => None,
    }
}

pub(super) fn name(register: u8) -> String { format!("tb_bal{register}") }

/// First scalar register written and how many.
fn written(ins: &Insn, wide: bool) -> Option<(u8, u8)> {
    let m = ins.mnemonic.as_str();
    let d = ins.dst.first()?;
    if m.starts_with("s_load_") || m.starts_with("s_buffer_load_") {
        let width = ["x16", "x8", "x4", "x2"].iter().zip([16, 8, 4, 2]).find(|(s, _)| m.ends_with(*s)).map_or(1, |(_, w)| w);
        return Some((register(d)?, width));
    }
    if m.starts_with("s_") {
        if m.starts_with("s_cmp") || m.starts_with("s_bitcmp") { return None; }
        let single = ["s_bcnt", "s_ff", "s_flbit"].iter().any(|p| m.starts_with(p));
        return Some((register(d)?, if m.ends_with("64") && !single { 2 } else { 1 }));
    }
    if m == "v_readlane_b32" {
        return match *d { Opnd::Vgpr(n) | Opnd::Sgpr(n) => Some((n, 1)), _ => register(d).map(|n| (n, 1)) };
    }
    if m.starts_with("v_cmpx") { return None; }
    if m.starts_with("v_") && m != "v_readfirstlane_b32" {
        let mask = ins.dst.iter().find_map(register)?;
        return Some((mask, if wide { 2 } else { 1 }));
    }
    Some((register(d)?, if m.starts_with("tb_") && m.ends_with("64") { 2 } else { 1 }))
}

enum Kind { Mask, Of(Vec<u8>), AllLanes, Number }

/// What an instruction's scalar result is made of.
fn kind(ins: &Insn) -> Kind {
    let m = ins.mnemonic.as_str();
    if m.starts_with("v_cmp") || m.contains("_co_") || m.starts_with("v_div_scale") { return Kind::Mask; }
    if m.contains("saveexec") { return Kind::Mask; }
    let combines = ["s_mov_b", "s_and_b", "s_or_b", "s_xor_b", "s_andn2_b", "s_orn2_b", "s_nand_b", "s_nor_b",
        "s_xnor_b", "s_not_b", "s_wqm_b", "s_cselect_b"];
    if !combines.iter().any(|p| m.strip_prefix(p).is_some_and(|rest| matches!(rest, "32" | "64"))) { return Kind::Number; }
    let (mut exec, mut from) = (false, Vec::new());
    for o in &ins.src {
        match o {
            Opnd::ExecLo => exec = true,
            Opnd::ConstInt(0 | -1) => (),
            _ => match register(o) { Some(n) => from.push(n), None => return Kind::Number },
        }
    }
    match (exec, from.is_empty()) {
        (_, false) => Kind::Of(from),
        (true, true) => Kind::Mask,
        (false, true) if ins.src.contains(&Opnd::ConstInt(-1)) => Kind::AllLanes,
        (false, true) => Kind::Of(Vec::new()),
    }
}

fn predecessors(insns: &[Insn]) -> Vec<Vec<usize>> {
    let at: BTreeMap<i64, usize> = insns.iter().enumerate().map(|(i, x)| (x.offset as i64, i)).collect();
    let mut from = vec![Vec::new(); insns.len()];
    for (i, ins) in insns.iter().enumerate() {
        let sopp = ins.enc == Enc::Sopp;
        if i + 1 < insns.len() && !(sopp && matches!(ins.op, 1 | 2)) { from[i + 1].push(i); }
        if sopp && Emitter::branch_cond(ins.op).is_some() {
            if let Some(&t) = at.get(&(ins.offset as i64 + 4 + 4 * ins.aux.imm)) { from[t].push(i); }
        }
    }
    from
}

struct Reach<'a> { insns: &'a [Insn], from: Vec<Vec<usize>>, wide: bool }

impl Reach<'_> {
    /// The writes of `register` that reach `at`. None when its entry value does.
    fn defs(&self, at: usize, register: u8) -> Option<BTreeSet<usize>> {
        let (mut found, mut seen, mut todo) = (BTreeSet::new(), BTreeSet::new(), vec![at]);
        while let Some(i) = todo.pop() {
            if i == 0 { return None; }
            // Padding after an unconditional branch is never entered.
            for &p in &self.from[i] {
                if !seen.insert(p) { continue; }
                match written(&self.insns[p], self.wide) {
                    Some((first, count)) if (first..first.saturating_add(count)).contains(&register) => { found.insert(p); }
                    _ => todo.push(p),
                }
            }
        }
        Some(found)
    }

    /// The definitions of a mask read at `at`, if every value that reaches it is one.
    fn mask(&self, at: usize, register: u8) -> Option<BTreeSet<usize>> {
        let direct = self.defs(at, register)?;
        let (mut seen, mut todo, mut rooted) = (BTreeSet::new(), direct.iter().copied().collect::<Vec<_>>(), false);
        while let Some(d) = todo.pop() {
            if !seen.insert(d) { continue; }
            match kind(&self.insns[d]) {
                Kind::Number => return None,
                // Counted itself, it holds lanes that are not running.
                Kind::AllLanes if direct.contains(&d) => return None,
                Kind::AllLanes => (),
                Kind::Mask => rooted = true,
                Kind::Of(regs) => for n in regs { todo.extend(self.defs(d, n)?); },
            }
        }
        rooted.then_some(direct)
    }
}

fn counted(ins: &Insn) -> bool {
    matches!(ins.mnemonic.as_str(), "s_bcnt1_i32_b32" | "s_bcnt1_i32_b64" | "s_ff1_i32_b32" | "s_ff1_i32_b64"
        | "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32")
}

pub(super) struct Ballots {
    /// Registers to take a ballot of after an instruction.
    pub defs: BTreeMap<usize, BTreeSet<u8>>,
    /// Instructions that read ballots instead of their mask operand.
    pub uses: BTreeSet<usize>,
    pub width: u32,
}

impl Ballots {
    /// The variables the ballots are kept in.
    pub(super) fn names(&self) -> Vec<String> {
        let words = u8::from(self.width == 64);
        self.defs.values().flatten().flat_map(|&n| [n, n + words]).map(name).collect()
    }
}

/// Masks of a compute program that are read as numbers.
pub(super) fn find(insns: &[Insn]) -> Option<Ballots> {
    if insns.iter().any(|i| matches!(i.mnemonic.as_str(), "s_setpc_b64" | "s_swappc_b64" | "s_call_b64")
        || i.mnemonic.starts_with("s_movrel")) { return None; }
    let masks = |bits: &str| insns.iter().any(|i| i.mnemonic.starts_with("s_") && i.mnemonic.ends_with(bits)
        && (i.mnemonic.contains("saveexec") || i.src.contains(&Opnd::ExecLo) || i.dst.contains(&Opnd::ExecLo)));
    let wide = match (masks("_b32"), masks("_b64")) { (true, false) => false, (false, true) => true, _ => return None };
    let reach = Reach { insns, from: predecessors(insns), wide };
    let mut found = Ballots { defs: BTreeMap::new(), uses: BTreeSet::new(), width: if wide { 64 } else { 32 } };
    for (i, ins) in insns.iter().enumerate().filter(|(_, i)| counted(i)) {
        let Some(source) = ins.src.first() else { continue };
        // The emitter already takes the ballot of EXEC for the other two.
        if matches!(source, Opnd::ExecLo | Opnd::ExecHi) {
            if ins.mnemonic.starts_with("s_bcnt1") || ins.mnemonic == "s_ff1_i32_b32" { found.uses.insert(i); }
            continue;
        }
        let Some(register) = register(source) else { continue };
        let Some(defs) = reach.mask(i, register) else {
            continue
        };
        for d in defs {
            let (first, count) = written(&insns[d], wide).expect("a definition writes");
            found.defs.entry(d).or_default().insert(if count == 2 { first } else { register });
        }
        found.uses.insert(i);
    }
    (!found.uses.is_empty()).then_some(found)
}

/// The text of a mask word read by instruction `at`.
pub(super) fn word(em: &Emitter<'_>, at: usize, o: &Opnd) -> Option<String> {
    let ballots = em.mask_ballots.as_ref().filter(|b| b.uses.contains(&at))?;
    match o {
        Opnd::ExecLo => Some("WaveActiveBallot((exec_lo & 1u) != 0u).x".into()),
        Opnd::ExecHi => Some(if ballots.width == 64 { "WaveActiveBallot((exec_lo & 1u) != 0u).y" } else { "0u" }.into()),
        _ => register(o).map(name),
    }
}

/// Statements that take the ballots due after instruction `at`.
pub(super) fn capture(em: &Emitter<'_>, at: usize) -> Vec<String> {
    let Some(ballots) = &em.mask_ballots else { return Vec::new() };
    ballots.defs.get(&at).into_iter().flatten().map(|&n| {
        let mask = if n == VCC { "vcc_lo".to_owned() } else { format!("s{n}") };
        if ballots.width == 64 {
            format!("{{ uint4 tb_b = WaveActiveBallot(({mask} & 1u) != 0u); {} = tb_b.x; {} = tb_b.y; }}", name(n), name(n + 1))
        } else {
            format!("{} = WaveActiveBallot(({mask} & 1u) != 0u).x;", name(n))
        }
    }).collect()
}

#[cfg(test)]
#[path = "../tests/hlsl/mask_ballots.rs"]
mod tests;
