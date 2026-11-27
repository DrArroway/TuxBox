//! Lower ProjectScarlett's inlined console ray walks to native ray queries.
//! The title compiles each query into a loop over raw console BVH nodes. The
//! loop is replaced by one native query against the scene the runtime binds
//! in its private space; the code around it keeps the guest ISA.
use super::{packed_root::matches_program, Emitter, Enc, Insn, Opnd, ShaderExtract, Stage};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Walk {
    /// CS_IrradianceCache_ShadowTrace: one occlusion ray per cache texel.
    IrradianceShadow,
    /// CS_IrradianceCache_ShadowTrace_WithFar: the same ray against the near
    /// scene, then, if nothing was hit, its remainder against the far scene.
    IrradianceShadowFar,
}

pub(crate) fn identify(ex: &ShaderExtract) -> Option<Walk> {
    if ex.stage != Stage::Compute || ex.psv_threads != Some([32, 1, 1]) {
        return None;
    }
    match ex.entry.as_deref()? {
        "CS_IrradianceCache_ShadowTrace_ad5ad774_Wrapped"
            if matches_program(&ex.isa, 2832, 0x43fba5aebffdef56) =>
        {
            Some(Walk::IrradianceShadow)
        }
        "CS_IrradianceCache_ShadowTrace_WithFar_ad5ad774_Wrapped"
            if matches_program(&ex.isa, 4992, 0x41caa1a510538f9d) =>
        {
            Some(Walk::IrradianceShadowFar)
        }
        _ => None,
    }
}

pub(super) const MNEMONIC: &str = "tb_scarlett_ray_query";

/// One walk in the guest ISA.
struct Loop {
    /// The load of the scene's root node pointer and the store that seeds the
    /// shared-memory stack. Both only feed the walk.
    scratch: [usize; 2],
    /// From the walk's first instruction through its back edge. Nothing
    /// outside this range branches into it except to `at`, and nothing in it
    /// branches out.
    first: usize,
    last: usize,
    /// Where the query goes: the instruction after the entry branch, or the
    /// target of a branch from elsewhere.
    at: usize,
    /// World-space ray: origin xyz, direction xyz, maximum distance. The walk
    /// copies these before it starts, so they hold the world ray throughout.
    ray: [u8; 7],
    /// Register holding the minimum distance, if the walk rejects hits closer
    /// than one; otherwise the walk accepts t >= 0.
    min: Option<u8>,
    /// Nonzero after an accepted triangle; zero before the walk.
    hit: u8,
    /// The scene's descriptor: the user SGPR holding the table pointer (the
    /// walk reads it through a copy) and the byte offset in the table.
    table: u8,
    scene: u32,
}

const IRRADIANCE_SHADOW: &[Loop] = &[Loop {
    scratch: [0x017c, 0x0334],
    first: 0x03bc,
    last: 0x0acc,
    at: 0x03c0,
    ray: [9, 7, 8, 17, 5, 45, 12],
    min: None,
    hit: 10,
    table: 5,
    scene: 128,
}];

// The near walk keeps its world ray in v45, v2, v4 (origin), v12, v11, v10
// (direction) and v14 (distance). The far walk starts where the near ray
// leaves a sphere around the cache and runs only if the near walk missed.
const IRRADIANCE_SHADOW_FAR: &[Loop] = &[
    Loop {
        scratch: [0x0384, 0x0418],
        first: 0x0428,
        last: 0x0b34,
        at: 0x042c,
        ray: [45, 2, 4, 12, 11, 10, 14],
        min: None,
        hit: 7,
        table: 5,
        scene: 128,
    },
    Loop {
        scratch: [0x0bc4, 0x0b7c],
        first: 0x0c24,
        last: 0x133c,
        at: 0x0c28,
        ray: [14, 15, 9, 17, 18, 13, 46],
        min: None,
        hit: 7,
        table: 5,
        scene: 96,
    },
];

fn loops(walk: Walk) -> &'static [Loop] {
    match walk {
        Walk::IrradianceShadow => IRRADIANCE_SHADOW,
        Walk::IrradianceShadowFar => IRRADIANCE_SHADOW_FAR,
    }
}

/// Replaces each walk with a single query instruction. The query names the
/// scene by its descriptor-table slot; the emitter resolves the slot to the
/// title's binding.
pub(super) fn prepare(walk: Walk, ops: &mut Vec<Insn>) {
    for l in loops(walk) {
        let at = l.at;
        ops.retain(|op| {
            !(l.scratch.contains(&op.offset) || (op.offset >= l.first && op.offset <= l.last && op.offset != at))
        });
        let query = ops.iter_mut().find(|op| op.offset == at).expect("matched walk ISA");
        query.mnemonic = MNEMONIC.into();
        query.enc = Enc::Vop1;
        query.src = l.ray.iter().map(|&r| Opnd::Vgpr(r)).chain([
            l.min.map_or(Opnd::ConstInt(0), Opnd::Vgpr),
            Opnd::Sgpr(l.table),
            Opnd::Literal(l.scene),
        ]).collect();
        query.dst = vec![Opnd::Vgpr(l.hit)];
    }
}

/// Index of a title acceleration structure in the runtime's private scene
/// space: the title's structures ordered by (space, register).
fn scene_index(ex: &ShaderExtract, space: u32, register: u32) -> Option<usize> {
    let mut titles: Vec<(u32, u32)> =
        ex.resources.iter().filter(|r| r.res_type == 4 && r.lower == r.upper).map(|r| (r.space, r.lower)).collect();
    titles.sort_unstable();
    titles.iter().position(|&t| t == (space, register))
}

pub(super) fn emit(em: &mut Emitter<'_>, ins: &Insn) -> Result<(), String> {
    let v = |n: usize| match ins.src[n] {
        Opnd::Vgpr(r) => Ok(format!("v{r}")),
        _ => Err(format!("{MNEMONIC} at {:#x}: bad ray operand", ins.offset)),
    };
    let (Some(Opnd::Vgpr(hit)), Opnd::Sgpr(table), Opnd::Literal(offset)) = (ins.dst.first(), &ins.src[8], &ins.src[9])
    else {
        return Err(format!("{MNEMONIC} at {:#x}: bad operands", ins.offset));
    };
    let (kind, register, space) = em.table_binding_space(*table, *offset, 32)?;
    let scene = (kind == Some(0)).then(|| scene_index(em.ex, space, register)).flatten().ok_or_else(|| {
        format!("{MNEMONIC} at {:#x}: s{table} + {offset:#x} is not a title acceleration structure ({kind:?} t{register} space{space})", ins.offset)
    })?;
    let min = match ins.src[7] {
        Opnd::Vgpr(r) => format!("asfloat(v{r})"),
        _ => "0.0".into(),
    };
    // The walk accepts the first triangle with minimum <= t < maximum and
    // tests no instance mask. It also reads a facing-cull field from each
    // console instance's flags; the native query culls nothing, which matches
    // the walk for instances that leave that field clear.
    em.line(&format!(
        "if (exec_lo & 1u) {{ v{hit} = tbScarlettOccluded(tbScarlettScene{scene}, asfloat(uint3({}, {}, {})), asfloat(uint3({}, {}, {})), {min}, asfloat({})) ? 1u : 0u; }}",
        v(0)?, v(1)?, v(2)?, v(3)?, v(4)?, v(5)?, v(6)?
    ));
    Ok(())
}

/// Declares the scenes the emitted code reads, at their register in the
/// runtime's private space 900, and the query they share.
pub(super) fn declarations(hlsl: &str) -> String {
    let mut out = String::new();
    for i in 0..8 {
        if hlsl.contains(&format!("tbScarlettScene{i},")) {
            out += &format!("RaytracingAccelerationStructure tbScarlettScene{i} : register(t{i}, space900);\n");
        }
    }
    out + r#"bool tbScarlettOccluded(RaytracingAccelerationStructure scene, float3 origin, float3 direction, float nearT, float farT) {
    if (!(farT > nearT) || !(nearT >= 0.0)) return false;
    RayDesc ray = {origin, nearT, direction, farT};
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xffu, ray);
    query.Proceed();
    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}
"#
}

#[cfg(test)]
#[path = "../tests/hlsl/scarlett_rays.rs"]
mod tests;
