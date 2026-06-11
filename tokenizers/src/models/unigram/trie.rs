use crate::tokenizer::Result;
use ahash::AHashMap;
use std::convert::TryInto;
use std::sync::Arc;
use yada::unit::{Unit, UNIT_SIZE};
use yada::{builder::DoubleArrayBuilder, DoubleArray};

const YADA_MAX_VALUE: u32 = 0x7fff_ffff;

pub(crate) struct Trie {
    inner: TrieInner,
}

enum TrieInner {
    Yada(DoubleArray<Arc<[u8]>>),
    Fallback(FallbackTrie),
}

impl Clone for Trie {
    fn clone(&self) -> Self {
        let inner = match &self.inner {
            TrieInner::Yada(da) => TrieInner::Yada(da.clone()),
            TrieInner::Fallback(trie) => TrieInner::Fallback(trie.clone()),
        };
        Self { inner }
    }
}

impl Trie {
    pub(crate) fn build<'a>(entries: impl Iterator<Item = (&'a [u8], u32)>) -> Result<Self> {
        let raw = entries
            .map(|(key, value)| (key.to_vec(), value))
            .collect::<Vec<_>>();

        // yada cannot represent these inputs, so fall back to the hash-map trie:
        // - an empty keyset,
        // - keys containing a NUL byte (yada uses `0x00` as the leaf terminator),
        // - values larger than `YADA_MAX_VALUE` (2^31 - 1).
        if raw.is_empty()
            || raw.iter().any(|(key, _)| key.contains(&0))
            || raw.iter().any(|(_, value)| *value > YADA_MAX_VALUE)
        {
            return Ok(Self::fallback(raw));
        }

        // Deduplicate to the unique, bytewise-sorted keyset that
        // `DoubleArrayBuilder::build` requires. Keep the value of the *last*
        // occurrence of each key so the fast path matches the fallback trie's
        // last-write-wins behaviour (which mirrors `token_to_ids` insertion
        // order in `Unigram`). Empty keys are dropped: yada rejects them.
        let mut last_value: AHashMap<&[u8], u32> = AHashMap::with_capacity(raw.len());
        for (key, value) in &raw {
            if key.is_empty() {
                continue;
            }
            last_value.insert(key.as_slice(), *value);
        }

        if last_value.is_empty() {
            return Ok(Self::fallback(raw));
        }

        let mut dedup: Vec<(&[u8], u32)> = last_value.into_iter().collect();
        dedup.sort_unstable_by(|(left_key, _), (right_key, _)| left_key.cmp(right_key));

        let bytes = match DoubleArrayBuilder::build(&dedup) {
            Ok(bytes) => bytes,
            Err(_) => return Ok(Self::fallback(raw)),
        };
        let bytes = Arc::<[u8]>::from(bytes);
        let da = match DoubleArray::new(bytes) {
            Ok(da) => da,
            Err(_) => return Ok(Self::fallback(raw)),
        };

        Ok(Self {
            inner: TrieInner::Yada(da),
        })
    }

    pub(crate) fn common_prefix_search<'a>(&'a self, suffix: &'a [u8]) -> SearchIter<'a> {
        match &self.inner {
            TrieInner::Yada(da) => SearchIter::Yada(YadaIter {
                bytes: &da.0,
                suffix,
                unit_id: 0,
                pos: 0,
            }),
            TrieInner::Fallback(trie) => SearchIter::Fallback(trie.common_prefix_search(suffix)),
        }
    }

    fn fallback(entries: Vec<(Vec<u8>, u32)>) -> Self {
        let mut trie = FallbackTrie::default();
        for (key, value) in entries {
            trie.push(&key, value);
        }
        Self {
            inner: TrieInner::Fallback(trie),
        }
    }

    #[cfg(test)]
    pub(crate) fn build_fallback_for_tests<'a>(
        entries: impl Iterator<Item = (&'a [u8], u32)>,
    ) -> Self {
        Self::fallback(
            entries
                .map(|(key, value)| (key.to_vec(), value))
                .collect::<Vec<_>>(),
        )
    }

    #[cfg(test)]
    pub(crate) fn is_fallback_for_tests(&self) -> bool {
        matches!(self.inner, TrieInner::Fallback(_))
    }

    #[cfg(test)]
    pub(crate) fn storage_bytes_for_tests(&self) -> Option<usize> {
        match &self.inner {
            TrieInner::Yada(da) => Some(da.0.len()),
            TrieInner::Fallback(_) => None,
        }
    }
}

pub(crate) enum SearchIter<'a> {
    Yada(YadaIter<'a>),
    Fallback(FallbackIter<'a>),
}

impl Iterator for SearchIter<'_> {
    type Item = (u32, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Yada(iter) => iter.next(),
            Self::Fallback(iter) => iter.next(),
        }
    }
}

pub(crate) struct YadaIter<'a> {
    bytes: &'a [u8],
    suffix: &'a [u8],
    unit_id: usize,
    pos: usize,
}

impl Iterator for YadaIter<'_> {
    type Item = (u32, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.suffix.len() {
            let unit = get_yada_unit(self.bytes, self.unit_id)?;
            // `self.pos < self.suffix.len()` is guaranteed by the loop, so direct
            // indexing cannot panic. A NUL byte can never be a valid edge in a
            // yada trie (it is the leaf terminator), so stop the walk: continuing
            // would let yada treat it as a terminator and report spurious matches.
            let byte = self.suffix[self.pos];
            if byte == 0 {
                return None;
            }
            self.pos += 1;

            self.unit_id = (unit.offset() ^ self.unit_id as u32 ^ byte as u32) as usize;
            let unit = get_yada_unit(self.bytes, self.unit_id)?;
            if unit.label() != byte as u32 {
                return None;
            }
            if unit.has_leaf() {
                let leaf_id = (unit.offset() ^ self.unit_id as u32) as usize;
                let leaf_unit = get_yada_unit(self.bytes, leaf_id)?;
                if !leaf_unit.is_leaf() {
                    return None;
                }
                return Some((leaf_unit.value(), self.pos));
            }
        }
        None
    }
}

fn get_yada_unit(bytes: &[u8], index: usize) -> Option<Unit> {
    let start = index.checked_mul(UNIT_SIZE)?;
    let end = start.checked_add(UNIT_SIZE)?;
    let raw = bytes.get(start..end)?.try_into().ok()?;
    Some(Unit::from_u32(u32::from_le_bytes(raw)))
}

/// Hash-map trie used whenever the input cannot be represented by yada's
/// double-array (empty vocab, keys containing NUL, or values above
/// `YADA_MAX_VALUE`). This is a permanent, correctness-preserving fallback, not
/// deprecated code.
#[derive(Clone, Default)]
struct FallbackTrie {
    root: FallbackNode,
}

impl FallbackTrie {
    fn push(&mut self, key: &[u8], value: u32) {
        let mut node = &mut self.root;
        for byte in key {
            node = node.children.entry(*byte).or_default();
        }
        node.value = Some(value);
    }

    fn common_prefix_search<'a>(&'a self, suffix: &'a [u8]) -> FallbackIter<'a> {
        FallbackIter {
            node: Some(&self.root),
            suffix,
            pos: 0,
        }
    }
}

#[derive(Clone, Default)]
struct FallbackNode {
    value: Option<u32>,
    children: AHashMap<u8, FallbackNode>,
}

pub(crate) struct FallbackIter<'a> {
    node: Option<&'a FallbackNode>,
    suffix: &'a [u8],
    pos: usize,
}

impl Iterator for FallbackIter<'_> {
    type Item = (u32, usize);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let node = self.node?;
            let byte = *self.suffix.get(self.pos)?;
            self.pos += 1;

            let child = node.children.get(&byte)?;
            self.node = Some(child);
            if let Some(value) = child.value {
                return Some((value, self.pos));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(entries: &[(&[u8], u32)]) -> Trie {
        Trie::build(entries.iter().map(|(key, value)| (*key, *value))).unwrap()
    }

    #[test]
    fn prefix_order() {
        let trie = build(&[(b"a", 1), (b"ab", 2), (b"abc", 3)]);

        assert_eq!(
            trie.common_prefix_search(b"abcd").collect::<Vec<_>>(),
            vec![(1, 1), (2, 2), (3, 3)]
        );
    }

    #[test]
    fn last_wins_dedup() {
        // The value decreases across duplicates, so this distinguishes
        // last-write-wins from a max-wins policy: the last occurrence (1) must
        // win in both the yada fast path and the hash-map fallback.
        let entries: &[(&[u8], u32)] = &[(b"a", 7), (b"a", 1), (b"ab", 2)];

        let yada = build(entries);
        assert!(!yada.is_fallback_for_tests());
        assert_eq!(
            yada.common_prefix_search(b"ab").collect::<Vec<_>>(),
            vec![(1, 1), (2, 2)]
        );

        let fallback =
            Trie::build_fallback_for_tests(entries.iter().map(|(key, value)| (*key, *value)));
        assert!(fallback.is_fallback_for_tests());
        assert_eq!(
            fallback.common_prefix_search(b"ab").collect::<Vec<_>>(),
            vec![(1, 1), (2, 2)]
        );
    }

    #[test]
    fn large_value_uses_fallback() {
        let trie = build(&[(b"a", YADA_MAX_VALUE + 1)]);

        assert!(trie.is_fallback_for_tests());
        assert_eq!(
            trie.common_prefix_search(b"a").collect::<Vec<_>>(),
            vec![(YADA_MAX_VALUE + 1, 1)]
        );
    }

    #[test]
    fn empty_vocab_uses_fallback_and_matches_nothing() {
        let trie = Trie::build(std::iter::empty()).unwrap();

        assert!(trie.is_fallback_for_tests());
        assert_eq!(trie.common_prefix_search(b"a").collect::<Vec<_>>(), vec![]);
    }

    #[test]
    fn empty_token_is_ignored_for_matching() {
        let trie = build(&[(b"", 0), (b"a", 1)]);

        assert_eq!(
            trie.common_prefix_search(b"a").collect::<Vec<_>>(),
            vec![(1, 1)]
        );
    }

    #[test]
    fn nul_input_matches_legal_prefix_only() {
        let trie = build(&[(b"a", 1), (b"ab", 2)]);

        assert_eq!(
            trie.common_prefix_search(b"a\0b").collect::<Vec<_>>(),
            vec![(1, 1)]
        );
    }

    #[test]
    fn nul_input_does_not_match_across_nul() {
        let trie = build(&[(b"ab", 2)]);

        assert!(!trie.is_fallback_for_tests());
        assert_eq!(
            trie.common_prefix_search(b"a\0b").collect::<Vec<_>>(),
            vec![]
        );
        assert_eq!(
            trie.common_prefix_search(b"\0ab").collect::<Vec<_>>(),
            vec![]
        );

        let trie = build(&[(b"ab", 1), (b"abc", 2), (b"x", 3), (b"xyz", 4)]);

        assert!(!trie.is_fallback_for_tests());
        assert_eq!(
            trie.common_prefix_search(b"ab\0c").collect::<Vec<_>>(),
            vec![(1, 2)]
        );
        assert_eq!(
            trie.common_prefix_search(b"x\0yz").collect::<Vec<_>>(),
            vec![(3, 1)]
        );
    }

    #[test]
    fn nul_key_uses_fallback_and_matches_byte_exact() {
        let trie = build(&[(b"a", 1), (b"a\0b", 2)]);

        assert!(trie.is_fallback_for_tests());
        assert_eq!(
            trie.common_prefix_search(b"a\0b").collect::<Vec<_>>(),
            vec![(1, 1), (2, 3)]
        );
        assert_eq!(
            trie.common_prefix_search(b"a\0c").collect::<Vec<_>>(),
            vec![(1, 1)]
        );
    }
}
