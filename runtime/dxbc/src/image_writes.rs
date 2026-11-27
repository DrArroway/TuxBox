//! Whether a console program can write to an image.
//!
//! A program that indexes the resource heap directly can reach any view in it,
//! so the provider must assume it writes every writable texture there unless
//! the program has no instruction that writes an image at all.

use crate::rdna2::{self, Enc};
use crate::{Dxbc, fssm, xdat};

/// True unless every variant of `shader` is console code that decodes
/// completely and contains no image store or image atomic.
///
/// The answer errs towards true: a desktop shader, a container this reader
/// does not know, an undecoded instruction and an indirect jump all count as
/// a possible image write.
pub fn may_write_images(shader: &[u8]) -> bool {
    writes_no_image(shader).is_none()
}

fn writes_no_image(shader: &[u8]) -> Option<()> {
    let container = Dxbc::parse(shader).ok()?;
    let part = container.chunk(b"DXIL")?;
    let variants = xdat::programs(shader.get(part.offset + 24..part.offset + part.size)?).ok()?;
    if variants.is_empty() {
        return None;
    }
    for program in &variants {
        isa_writes_no_image(&fssm::decode(program).ok()?.isa)?;
    }
    Some(())
}

/// `Some` when the instruction stream is fully decoded, stays inside itself
/// and has no image store or image atomic.
pub fn isa_writes_no_image(isa: &[u8]) -> Option<()> {
    if isa.is_empty() || !isa.len().is_multiple_of(4) {
        return None;
    }
    for op in rdna2::instructions(isa) {
        if !op.is_complete()
            || op.enc == Enc::Unknown
            || op.mnemonic.contains('<')
            // Code reached through a computed address is not in this stream.
            || matches!(
                op.mnemonic.as_str(),
                "s_call_b64" | "s_setpc_b64" | "s_swappc_b64" | "s_rfe_b64" | "s_trap"
            )
            || op.mnemonic.starts_with("image_store")
            || op.mnemonic.starts_with("image_atomic")
        {
            return None;
        }
    }
    Some(())
}

#[cfg(test)]
#[path = "tests/image_writes.rs"]
mod tests;
