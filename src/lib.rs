use std::ops::Rem;

/// Size of a word in the bitvector.
const WORD_SIZE: usize = 64;

/// Size of a block in the bitvector. The size is chosen so we have enough room for the L1 counters
/// in super blocks left if we store 8 10-bit L2-counters in their 128 bit word
const BLOCK_SIZE: usize = 512;

/// Size of a super block in the bitvector. Super-blocks exist to decrease the memory overhead
/// of block descriptors. The size is chosen deliberately so we can store 44 bit of L1 counter and
/// 8 10-bit L2 counters in a 128 bit word.
const SUPER_BLOCK_SIZE: usize = 1 << 12;

/// Maximum size of a bitvector. This is the maximum size of a bitvector that can be stored in
/// memory. This limitation is due to the fact that super-blocks only reserve 44 bit for the
/// zero-counter. This can in theory be increased by an L0-block with 2^44 bits covered.
const MAX_SIZE: u64 = 1 << 44;

const L2_COUNTER_LANE: usize = 84;

const L2_COUNTER_LANE_MASK: u128 = 0x0000_0000_000f_ffff_ffff_ffff_ffff_ffff;

const L2_COUNTER_SIZE: usize = 12;

const L2_COUNTER_MASK: usize = 0xfff;

/// Meta-data for a super-block. The `zeros` field stores the number of zeros up to this super-block.
/// This allows the `BlockDescriptor` to store the number of zeros in a much smaller
/// space. The `zeros` field is the number of zeros up to the super-block.
#[derive(Clone, Copy, Debug)]
struct InterleavedBlockDescriptor {
    interleaved: u128,
}

/// A bitvector that supports constant-time rank and select queries. The bitvector is stored as
/// a vector of `u64`s. The last word is not necessarily full, in which case the remaining bits
/// are set to 0. The bit-vector stores meta-data for constant-time rank and select queries, which
/// takes sub-linear additional space.
#[derive(Clone, Debug)]
pub struct BitVector {
    data: Vec<u64>,
    len: usize,
    interleaved_blocks: Vec<InterleavedBlockDescriptor>,
}

impl BitVector {
    /// Return the 0-rank of the bit at the given position. The 0-rank is the number of
    /// 0-bits in the vector up to but excluding the bit at the given position.
    ///
    /// # Parameters
    /// - `pos`: The position of the bit to return the rank of.
    pub fn rank0(&self, pos: usize) -> usize {
        unsafe { self.naive_rank0(pos) }
    }

    /// Return the 1-rank of the bit at the given position. The 1-rank is the number of
    /// 1-bits in the vector up to but excluding the bit at the given position.
    ///
    /// # Parameters
    /// - `pos`: The position of the bit to return the rank of.
    pub fn rank1(&self, pos: usize) -> usize {
        unsafe { self.naive_rank1(pos) }
    }

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
    #[inline(always)]
    fn rank(&self, zero: bool, pos: usize) -> usize {
        let index = pos / WORD_SIZE;
        let super_block_index = pos / SUPER_BLOCK_SIZE;
        let mut rank = 0usize;

        // at first add the number of zeros/ones before the current super block
        rank += if zero {
            (self.interleaved_blocks[super_block_index].interleaved >> L2_COUNTER_LANE) as usize
        } else {
            (super_block_index * SUPER_BLOCK_SIZE) - (self.interleaved_blocks[super_block_index].interleaved >> L2_COUNTER_LANE) as usize
        };

        // then add the number of zeros/ones before the current block by extracting the information
        // from the interleaved block descriptor. The information is stored in reverse order to
        // avoid a boundary check for the first block that would mess with the branch prediction.
        let block_index = (pos % SUPER_BLOCK_SIZE) / BLOCK_SIZE;
        let shift_index = L2_COUNTER_LANE - (L2_COUNTER_SIZE * block_index);
        let zeros_in_block = ((self.interleaved_blocks[super_block_index].interleaved & L2_COUNTER_LANE_MASK) >> shift_index) as usize & L2_COUNTER_MASK;

        rank += if zero {
            zeros_in_block as usize
        } else {
            block_index * BLOCK_SIZE - zeros_in_block
        };

        // naive popcount of remaining words
        for &i in &self.data[((super_block_index * SUPER_BLOCK_SIZE) + (block_index * BLOCK_SIZE)) / WORD_SIZE..index] {
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

    /// Return the length of the vector, i.e. the number of bits it contains.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A builder for `BitVector`s. This is used to efficiently construct a `BitVector` by appending
/// bits to it. Once all bits have been appended, the `BitVector` can be built using the `build`
/// method. If the number of bits to be appended is known in advance, it is recommended to use
/// `with_capacity` to avoid re-allocations. If bits are already available in little endian u64
/// words, those words can be appended using `append_word`.
#[derive(Clone, Debug)]
pub struct BitVectorBuilder {
    words: Vec<u64>,
    len: usize,
}

impl BitVectorBuilder {
    /// Create a new empty `BitVectorBuilder`.
    pub fn new() -> BitVectorBuilder {
        BitVectorBuilder {
            words: Vec::new(),
            len: 0,
        }
    }

    /// Create a new empty `BitVectorBuilder` with the specified initial capacity to avoid
    /// re-allocations.
    pub fn with_capacity(capacity: usize) -> BitVectorBuilder {
        BitVectorBuilder {
            words: Vec::with_capacity(capacity),
            len: 0,
        }
    }

    /// Append a bit to the vector.
    pub fn append_bit<T: Rem + From<u8>>(&mut self, bit: T)
        where
            T::Output: Into<u64>,
    {
        let bit: u64 = (bit % T::from(2u8)).into();

        if self.len % WORD_SIZE == 0 {
            self.words.push(0);
        }

        self.words[self.len / WORD_SIZE] |= bit << (self.len % WORD_SIZE);
        self.len += 1;
    }

    /// Append a word to the vector. The word is assumed to be in little endian, i.e. the least
    /// significant bit is the first bit. It is a logical error to append a word if the vector is
    /// not 64-bit aligned (i.e. has not a length that is a multiple of 64). If the vector is not
    /// 64-bit aligned, the last word already present will be padded with zeros,without
    /// affecting the length, meaning the bit-vector is corrupted afterwards.
    pub fn append_word(&mut self, word: u64) {
        debug_assert!(self.len % WORD_SIZE == 0);
        self.words.push(word);
        self.len += WORD_SIZE;
    }

    /// Build the `BitVector` from all bits that have been appended so far. This will consume the
    /// `BitVectorBuilder`.
    pub fn build(mut self) -> BitVector {
        if self.len > MAX_SIZE.try_into().unwrap_or(usize::MAX) {
            panic!("BitVector cannot be larger than {} bits", MAX_SIZE);
        }

        // Construct the block descriptor meta data. Each block descriptor contains the number of
        // zeros in the super-block, up to but excluding the block.
        // let mut blocks = Vec::with_capacity(self.len / BLOCK_SIZE + 1);
        let mut super_blocks = Vec::with_capacity(self.len / SUPER_BLOCK_SIZE + 1);

        let mut total_zeros_before: u128 = 0;
        let mut current_super_block_zeros: u32 = 0;
        let mut l2_counter_lane: u128 = 0;
        for (idx, &word) in self.words.iter().enumerate() {
            // if we moved past a block boundary, append the block information for the previous
            // block to the counter lane, or append a super-block descriptor if we moved past
            // a super-block boundary. Reset the counters in the latter case.
            if idx > 0 && idx % (SUPER_BLOCK_SIZE / WORD_SIZE) == 0 {
                let interleaved = (total_zeros_before << L2_COUNTER_LANE) | l2_counter_lane;
                super_blocks.push(InterleavedBlockDescriptor { interleaved });
                total_zeros_before += current_super_block_zeros as u128;
                current_super_block_zeros = 0;
                l2_counter_lane = 0;
            } else {
                if idx > 0 && idx % (BLOCK_SIZE / WORD_SIZE) == 0 {
                    l2_counter_lane = (l2_counter_lane << L2_COUNTER_SIZE) | (current_super_block_zeros as u128);
                }
            }

            // count the zeros in the current word and add them to the block counter
            // the last word may contain padding zeros, which should not be counted,
            // but since we do not append the last block descriptor, this is not a problem
            current_super_block_zeros += word.count_zeros();
        }

        // push last incomplete block. It is missing the last L2 counter, but this will never be
        // accessed, so we can ignore it
        let interleaved = (total_zeros_before << L2_COUNTER_LANE) | l2_counter_lane;
        super_blocks.push(InterleavedBlockDescriptor { interleaved });

        // pad the internal vector to be block-aligned, so SIMD operations don't try to read
        // past the end of the vector. Note that this does not affect the content of the vector,
        // because those bits are not considered part of the vector.
        while self.words.len() % (BLOCK_SIZE / WORD_SIZE) != 0 {
            self.words.push(0);
        }

        BitVector {
            data: self.words,
            len: self.len,
            interleaved_blocks: super_blocks,
        }
    }
}

impl Default for BitVectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::Distribution;
    use rand::distributions::Uniform;
    use rand::Rng;

    #[test]
    fn test_append_bit() {
        let mut bv = BitVectorBuilder::new();
        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        let bv = bv.build();

        assert_eq!(bv.data[..1], vec![0b110]);
    }

    #[test]
    fn test_append_bit_long() {
        let mut bv = BitVectorBuilder::new();

        let len = SUPER_BLOCK_SIZE + 1;
        for _ in 0..len {
            bv.append_bit(0u8);
            bv.append_bit(1u8);
        }

        let bv = bv.build();

        assert_eq!(bv.len(), len * 2);
        assert_eq!(bv.rank0(2 * len - 1), len);
        assert_eq!(bv.rank1(2 * len - 1), len - 1);
    }

    #[test]
    fn test_rank() {
        let mut bv = BitVectorBuilder::new();
        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        let bv = bv.build();

        // first bit must always have rank 0
        assert_eq!(bv.rank0(0), 0);
        assert_eq!(bv.rank1(0), 0);

        assert_eq!(bv.rank1(2), 1);
        assert_eq!(bv.rank1(3), 2);
        assert_eq!(bv.rank1(4), 2);
        assert_eq!(bv.rank0(3), 1);
    }

    #[test]
    fn test_multi_words() {
        let mut bv = BitVectorBuilder::new();
        bv.append_word(0);
        bv.append_bit(0u8);
        bv.append_bit(1u8);
        bv.append_bit(1u8);
        let bv = bv.build();

        // if BLOCK_SIZE is changed, we need to update this test case
        assert_eq!(bv.data.len(), BLOCK_SIZE / WORD_SIZE);

        assert_eq!(bv.rank0(63), 63);
        assert_eq!(bv.rank0(64), 64);
        assert_eq!(bv.rank0(65), 65);
        assert_eq!(bv.rank0(66), 65);
    }

    #[test]
    fn test_super_block() {
        let mut bv = BitVectorBuilder::with_capacity(LENGTH);
        let mut rng = rand::thread_rng();
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
    fn test_only_zeros() {
        let mut bv = BitVectorBuilder::new();
        for _ in 0..2 * (SUPER_BLOCK_SIZE / WORD_SIZE) {
            bv.append_word(0);
        }
        bv.append_bit(0u8);
        let bv = bv.build();

        assert_eq!(bv.len(), 2 * SUPER_BLOCK_SIZE + 1);

        for i in 0..bv.len() {
            assert_eq!(bv.rank0(i), i);
            assert_eq!(bv.rank1(i), 0);
        }
    }

    #[test]
    fn test_only_ones() {
        let mut bv = BitVectorBuilder::new();
        for _ in 0..2 * (SUPER_BLOCK_SIZE / WORD_SIZE) {
            bv.append_word(u64::MAX);
        }
        bv.append_bit(1u8);
        let bv = bv.build();

        assert_eq!(bv.len(), 2 * SUPER_BLOCK_SIZE + 1);

        for i in 0..bv.len() {
            assert_eq!(bv.rank0(i), 0);
            assert_eq!(bv.rank1(i), i);
        }
    }
}
