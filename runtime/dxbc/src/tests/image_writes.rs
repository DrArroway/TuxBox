use super::{isa_writes_no_image, may_write_images};
use crate::rdna2;

fn isa(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

const ENDPGM: u32 = 0xbf81_0000;
/// MIMG instructions of two dwords with the opcode in bits 18..26.
fn mimg(op: u32) -> [u32; 2] {
    [0xf000_0108 | op << 18, 0]
}

fn mnemonics(words: &[u32]) -> Vec<String> {
    rdna2::instructions(&isa(words)).map(|op| op.mnemonic).collect()
}

#[test]
fn a_program_that_only_loads_images_writes_none() {
    let [load, operands] = mimg(0);
    let words = [load, operands, ENDPGM];
    assert_eq!(mnemonics(&words), ["image_load", "s_endpgm"]);
    assert!(isa_writes_no_image(&isa(&words)).is_some());
}

#[test]
fn every_store_and_atomic_counts_as_a_write() {
    for (op, name) in [
        (8, "image_store"),
        (9, "image_store_mip"),
        (10, "image_store_pck"),
        (11, "image_store_mip_pck"),
        (15, "image_atomic_swap"),
        (16, "image_atomic_cmpswap"),
        (17, "image_atomic_add"),
        (31, "image_atomic_fmax"),
    ] {
        let [first, operands] = mimg(op);
        let words = [first, operands, ENDPGM];
        assert_eq!(mnemonics(&words)[0], name);
        assert!(isa_writes_no_image(&isa(&words)).is_none(), "{name}");
    }
}

#[test]
fn what_cannot_be_read_counts_as_a_write() {
    // Nothing to read, a partial dword, and a truncated final instruction.
    assert!(isa_writes_no_image(&[]).is_none());
    assert!(isa_writes_no_image(&isa(&[ENDPGM])[..3]).is_none());
    assert!(isa_writes_no_image(&isa(&[mimg(0)[0]])).is_none());
    // An opcode the decoder has no name for.
    let unknown = rdna2::instructions(&isa(&[0xf3fc_0108, 0])).next().unwrap();
    assert!(unknown.mnemonic.contains('<'), "{}", unknown.mnemonic);
    assert!(isa_writes_no_image(&isa(&[0xf3fc_0108, 0, ENDPGM])).is_none());
    // Desktop or unknown containers are not console code.
    assert!(may_write_images(&[]));
    assert!(may_write_images(b"DXBC not a container"));
}
