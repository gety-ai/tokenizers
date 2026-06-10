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
    Legacy(LegacyTrie),
}

impl Clone for Trie {
    fn clone(&self) -> Self {
        let inner = match &self.inner {
            TrieInner::Yada(da) => TrieInner::Yada(da.clone()),
            TrieInner::Legacy(trie) => TrieInner::Legacy(trie.clone()),
        };
        Self { inner }
    }
}

impl Trie {
    pub(crate) fn build<'a>(entries: impl Iterator<Item = (&'a [u8], u32)>) -> Result<Self> {
        let raw = entries
            .map(|(key, value)| (key.to_vec(), value))
            .collect::<Vec<_>>();

        if raw.is_empty()
            || raw.iter().any(|(key, _)| key.contains(&0))
            || raw.iter().any(|(_, value)| *value > YADA_MAX_VALUE)
        {
            return Ok(Self::legacy(raw));
        }

        let mut keyset = raw
            .iter()
            .filter(|(key, _)| !key.is_empty())
            .map(|(key, value)| (key.clone(), *value))
            .collect::<Vec<_>>();

        if keyset.is_empty() {
            return Ok(Self::legacy(raw));
        }

        keyset.sort_by(|(left_key, left_value), (right_key, right_value)| {
            left_key
                .cmp(right_key)
                .then_with(|| left_value.cmp(right_value))
        });

        let mut dedup: Vec<(Vec<u8>, u32)> = Vec::with_capacity(keyset.len());
        for (key, value) in keyset {
            if let Some((last_key, last_value)) = dedup.last_mut() {
                if *last_key == key {
                    *last_value = (*last_value).max(value);
                    continue;
                }
            }
            dedup.push((key, value));
        }

        let bytes = match DoubleArrayBuilder::build(&dedup) {
            Ok(bytes) => bytes,
            Err(_) => return Ok(Self::legacy(raw)),
        };
        let bytes = Arc::<[u8]>::from(bytes);
        let da = match DoubleArray::new(bytes) {
            Ok(da) => da,
            Err(_) => return Ok(Self::legacy(raw)),
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
            TrieInner::Legacy(trie) => SearchIter::Legacy(trie.common_prefix_search(suffix)),
        }
    }

    fn legacy(entries: Vec<(Vec<u8>, u32)>) -> Self {
        let mut trie = LegacyTrie::default();
        for (key, value) in entries {
            trie.push(&key, value);
        }
        Self {
            inner: TrieInner::Legacy(trie),
        }
    }

    #[cfg(test)]
    pub(crate) fn build_legacy_for_tests<'a>(
        entries: impl Iterator<Item = (&'a [u8], u32)>,
    ) -> Self {
        Self::legacy(
            entries
                .map(|(key, value)| (key.to_vec(), value))
                .collect::<Vec<_>>(),
        )
    }

    #[cfg(test)]
    pub(crate) fn is_legacy_for_tests(&self) -> bool {
        matches!(self.inner, TrieInner::Legacy(_))
    }

    #[cfg(test)]
    pub(crate) fn storage_bytes_for_tests(&self) -> Option<usize> {
        match &self.inner {
            TrieInner::Yada(da) => Some(da.0.len()),
            TrieInner::Legacy(_) => None,
        }
    }
}

pub(crate) enum SearchIter<'a> {
    Yada(YadaIter<'a>),
    Legacy(LegacyIter<'a>),
}

impl Iterator for SearchIter<'_> {
    type Item = (u32, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Yada(iter) => iter.next(),
            Self::Legacy(iter) => iter.next(),
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
            let byte = *self.suffix.get(self.pos)?;
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

#[derive(Clone, Default)]
struct LegacyTrie {
    root: LegacyNode,
}

impl LegacyTrie {
    fn push(&mut self, key: &[u8], value: u32) {
        let mut node = &mut self.root;
        for byte in key {
            node = node.children.entry(*byte).or_default();
        }
        node.value = Some(value);
    }

    fn common_prefix_search<'a>(&'a self, suffix: &'a [u8]) -> LegacyIter<'a> {
        LegacyIter {
            node: Some(&self.root),
            suffix,
            pos: 0,
        }
    }
}

#[derive(Clone, Default)]
struct LegacyNode {
    value: Option<u32>,
    children: AHashMap<u8, LegacyNode>,
}

pub(crate) struct LegacyIter<'a> {
    node: Option<&'a LegacyNode>,
    suffix: &'a [u8],
    pos: usize,
}

impl Iterator for LegacyIter<'_> {
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
        let trie = build(&[(b"a", 1), (b"a", 7), (b"ab", 2)]);

        assert_eq!(
            trie.common_prefix_search(b"ab").collect::<Vec<_>>(),
            vec![(7, 1), (2, 2)]
        );
    }

    #[test]
    fn empty_vocab_uses_legacy_and_matches_nothing() {
        let trie = Trie::build(std::iter::empty()).unwrap();

        assert!(trie.is_legacy_for_tests());
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

        assert!(!trie.is_legacy_for_tests());
        assert_eq!(
            trie.common_prefix_search(b"a\0b").collect::<Vec<_>>(),
            vec![]
        );
        assert_eq!(
            trie.common_prefix_search(b"\0ab").collect::<Vec<_>>(),
            vec![]
        );

        let trie = build(&[(b"ab", 1), (b"abc", 2), (b"x", 3), (b"xyz", 4)]);

        assert!(!trie.is_legacy_for_tests());
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
    fn nul_key_uses_legacy_and_matches_byte_exact() {
        let trie = build(&[(b"a", 1), (b"a\0b", 2)]);

        assert!(trie.is_legacy_for_tests());
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
