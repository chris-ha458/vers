use crate::{BitVector, BitVectorBuilder, BuildingStrategy, WORD_SIZE};

/// Size of a block in the bitvector. The size is deliberately chosen to fit one block into a
/// AVX256 register, so that we can use SIMD instructions to speed up rank and select queries.
const BLOCK_SIZE: usize = 512;

/// Size of a super block in the bitvector. Super-blocks exist to decrease the memory overhead
/// of block descriptors.
/// Increasing or decreasing the super block size has negligible effect on performance. This
/// means we want to make the super block size as large as possible, as long as the zero-counter
/// in normal blocks still fits in a reasonable amount of bits. So we chose the super block size
/// to be the largest power of two that still fits into a `u16`. Note that blocks store the number
/// of zeros up to the block, so they will never overflow even if the entire vector contains only
/// zeros.
const SUPER_BLOCK_SIZE: usize = 1 << 16;

/// Meta-data for a block. The `zeros` field stores the number of zeros up to the block,
/// beginning from the last super-block boundary. This means the first block in a super-block
/// always stores the number zero, which serves as a sentinel value to avoid special-casing the
/// first block in a super-block (which would be a performance hit due branch prediction failures).
#[derive(Clone, Copy, Debug)]
struct BlockDescriptor {
    zeros: u16,
}

/// Meta-data for a super-block. The `zeros` field stores the number of zeros up to this super-block.
/// This allows the `BlockDescriptor` to store the number of zeros in a much smaller
/// space. The `zeros` field is the number of zeros up to the super-block.
#[derive(Clone, Copy, Debug)]
struct SuperBlockDescriptor {
    zeros: usize,
}

/// A bitvector that supports constant-time rank and select queries and is optimized for fast queries.
/// The bitvector is stored as a vector of `u64`s. The bit-vector stores meta-data for constant-time
/// rank and select queries, which takes sub-linear additional space. The overhead is
/// 2KiB + 4 bytes per 65KiB of user data (3.083%), plus 12 bytes constant overhead.
// TODO update once select-overhead is implemented
#[derive(Clone, Debug)]
pub struct FastBitVector {
    data: Vec<u64>,
    len: usize,
    blocks: Vec<BlockDescriptor>,
    super_blocks: Vec<SuperBlockDescriptor>,
}

impl FastBitVector {
    #[target_feature(enable = "popcnt")]
    unsafe fn naive_rank0(&self, pos: usize) -> usize {
        self.rank(true, pos)
    }

    #[target_feature(enable = "popcnt")]
    unsafe fn naive_rank1(&self, pos: usize) -> usize {
        self.rank(false, pos)
    }

    // I measured 5-10% improvement with this. I don't know why it's not inlined by default, the
    // branch elimination profits alone should make it worth it.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn rank(&self, zero: bool, pos: usize) -> usize {
        #[allow(unused_variables)]
        let index = pos / WORD_SIZE;
        let block_index = pos / BLOCK_SIZE;
        let super_block_index = pos / SUPER_BLOCK_SIZE;
        let mut rank = 0;

        // at first add the number of zeros/ones before the current super block
        rank += if zero {
            self.super_blocks[super_block_index].zeros
        } else {
            (super_block_index * SUPER_BLOCK_SIZE) - self.super_blocks[super_block_index].zeros
        };

        // then add the number of zeros/ones before the current block
        rank += if zero {
            self.blocks[block_index].zeros as usize
        } else {
            ((block_index % (SUPER_BLOCK_SIZE / BLOCK_SIZE)) * BLOCK_SIZE)
                - self.blocks[block_index].zeros as usize
        };

        // naive popcount of blocks
        for &i in &self.data[(block_index * BLOCK_SIZE) / WORD_SIZE..index] {
            rank += if zero {
                i.count_zeros() as usize
            } else {
                i.count_ones() as usize
            };
        }

        rank += if zero {
            (!self.data[index] & ((1 << (pos % WORD_SIZE)) - 1)).count_ones() as usize
        } else {
            (self.data[index] & ((1 << (pos % WORD_SIZE)) - 1)).count_ones() as usize
        };

        rank
    }
}

impl BitVector for FastBitVector {
    fn rank0(&self, pos: usize) -> usize {
        unsafe { self.naive_rank0(pos) }
    }

    fn rank1(&self, pos: usize) -> usize {
        unsafe { self.naive_rank1(pos) }
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl BuildingStrategy for FastBitVector {
    type Vector = FastBitVector;

    fn build(mut builder: BitVectorBuilder<Self>) -> FastBitVector {
        // Construct the block descriptor meta data. Each block descriptor contains the number of
        // zeros in the super-block, up to but excluding the block.
        let mut blocks = Vec::with_capacity(builder.len / BLOCK_SIZE + 1);
        let mut super_blocks = Vec::with_capacity(builder.len / SUPER_BLOCK_SIZE + 1);

        let mut total_zeros: usize = 0;
        let mut current_zeros: u32 = 0;
        for (idx, &word) in builder.words.iter().enumerate() {
            // if we moved past a block boundary, append the block information for the previous
            // block and reset the counter if we moved past a super-block boundary.
            if idx % (BLOCK_SIZE / WORD_SIZE) == 0 {
                if idx % (SUPER_BLOCK_SIZE / WORD_SIZE) == 0 {
                    total_zeros += current_zeros as usize;
                    current_zeros = 0;
                    super_blocks.push(SuperBlockDescriptor { zeros: total_zeros });
                }

                // this cannot overflow because the only block where it could (the last in a super-
                // block) is not added to the list of blocks
                #[allow(clippy::cast_possible_truncation)]
                blocks.push(BlockDescriptor {
                    zeros: current_zeros as u16,
                });
            }

            // count the zeros in the current word and add them to the counter
            // the last word may contain padding zeros, which should not be counted,
            // but since we do not append the last block descriptor, this is not a problem
            current_zeros += word.count_zeros();
        }

        // pad the internal vector to be block-aligned, so SIMD operations don't try to read
        // past the end of the vector. Note that this does not affect the content of the vector,
        // because those bits are not considered part of the vector.
        while builder.words.len() % (BLOCK_SIZE / WORD_SIZE) != 0 {
            builder.words.push(0);
        }

        FastBitVector {
            data: builder.words,
            len: builder.len,
            blocks,
            super_blocks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::{Distribution, Uniform};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_append_bit() {
        let mut bv = BitVectorBuilder::<FastBitVector>::new();
        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        let bv = bv.build();

        assert_eq!(bv.data[..1], vec![0b110]);
    }

    #[test]
    fn test_sample_data() {
        let mut bv = BitVectorBuilder::<FastBitVector>::with_capacity(LENGTH);
        let mut rng = StdRng::from_seed([
            0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4,
            5, 6, 7,
        ]);
        let sample = Uniform::new(0, 2);
        static LENGTH: usize = 4 * SUPER_BLOCK_SIZE;

        for _ in 0..LENGTH {
            bv.append_bit(sample.sample(&mut rng) as u8);
        }

        let bv = bv.build();

        assert_eq!(bv.len(), LENGTH);

        for _ in 0..100 {
            let rnd_index = rng.gen_range(0..LENGTH);
            let actual_rank1 = bv.rank1(rnd_index);
            let actual_rank0 = bv.rank0(rnd_index);

            let data = &bv.data;
            let mut expected_rank1 = 0;
            let mut expected_rank0 = 0;

            let data_index = rnd_index / WORD_SIZE;
            let bit_index = rnd_index % WORD_SIZE;

            for i in 0..data_index {
                expected_rank1 += data[i].count_ones() as usize;
                expected_rank0 += data[i].count_zeros() as usize;
            }

            if bit_index > 0 {
                expected_rank1 += (data[data_index] & (1 << bit_index) - 1).count_ones() as usize;
                expected_rank0 += (!data[data_index] & (1 << bit_index) - 1).count_ones() as usize;
            }

            assert_eq!(actual_rank1, expected_rank1);
            assert_eq!(actual_rank0, expected_rank0);
        }
    }

    #[test]
    fn test_append_bit_long() {
        let bv = BitVectorBuilder::<FastBitVector>::new();
        crate::common_tests::test_append_bit_long(bv, SUPER_BLOCK_SIZE);
    }

    #[test]
    fn test_rank() {
        let bv = BitVectorBuilder::<FastBitVector>::new();
        crate::common_tests::test_rank(bv);
    }

    #[test]
    fn test_multi_words() {
        let bv = BitVectorBuilder::<FastBitVector>::new();
        crate::common_tests::test_multi_words(bv);
    }

    #[test]
    fn test_only_zeros() {
        let bv = BitVectorBuilder::<FastBitVector>::new();
        crate::common_tests::test_only_zeros(bv, SUPER_BLOCK_SIZE, WORD_SIZE);
    }

    #[test]
    fn test_only_ones() {
        let bv = BitVectorBuilder::<FastBitVector>::new();
        crate::common_tests::test_only_ones(bv, SUPER_BLOCK_SIZE, WORD_SIZE);
    }
}
