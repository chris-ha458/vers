macro_rules! gen_iter_impl {
    ($($life:lifetime, )? $name:ident, $type:ty, $item:ty; $({ $($function_defs:tt)* })*) => {
        impl $(<$life>)? $name $(<$life>)? {
            #[must_use]
            fn new(vec: $(&$life)? $type) -> Self {
                if vec.is_empty() {
                    return Self {
                        vec,
                        index: 0,
                        back_index: None,
                    };
                }

                let last = vec.len - 1;
                Self {
                    vec,
                    index: 0,
                    back_index: Some(last),
                }
            }

            /// Advances the iterator by `n` elements. Returns an error if the iterator does not have
            /// enough elements left. Does not call `next` internally.
            /// This method is currently being added to the iterator trait, see
            /// [this issue](https://github.com/rust-lang/rust/issues/77404).
            /// As soon as it is stabilized, this method will be removed and replaced with a custom
            /// implementation in the iterator impl.
            fn advance_by(&mut self, n: usize) -> Result<(), std::num::NonZeroUsize> {
                if n == 0 {
                    return Ok(());
                }

                if Some(self.index + n - 1) > self.back_index {
                    if Some(self.index) > self.back_index {
                        Err(std::num::NonZeroUsize::new(n).unwrap())
                    } else {
                        Err(std::num::NonZeroUsize::new(n - (self.back_index.as_ref().unwrap_or(&usize::MAX).wrapping_sub(self.index).wrapping_add(1))).unwrap())
                    }
                } else {
                    self.index += n;
                    Ok(())
                }
            }

            /// Advances the back iterator by `n` elements. Returns an error if the iterator does not have
            /// enough elements left. Does not call `next` internally.
            /// This method is currently being added to the iterator trait, see
            /// [this issue](https://github.com/rust-lang/rust/issues/77404).
            /// As soon as it is stabilized, this method will be removed and replaced with a custom
            /// implementation in the iterator impl.
            fn advance_back_by(&mut self, n: usize) -> Result<(), std::num::NonZeroUsize> {
                if n == 0 {
                    return Ok(());
                }

                // special case this, because otherwise back_index might be None and we would panic
                if self.is_iter_empty() {
                    return Err(std::num::NonZeroUsize::new(n).unwrap());
                }

                // since the cursors point to unconsumed items, we need to add 1
                let remaining = *self.back_index.as_ref().unwrap() - self.index + 1;
                if remaining < n {
                    return Err(std::num::NonZeroUsize::new(n - remaining).unwrap());
                }
                self.back_index = if self.back_index >= Some(n) { self.back_index.map(|b| b - n) } else { None };
                Ok(())
            }

            fn is_iter_empty(&self) -> bool {
                // this is legal because Ord is behaving as expected on Option
                Some(self.index) > self.back_index
            }
        }

        impl $(<$life>)? Iterator for $name $(<$life>)? {
            type Item = $item;

            fn next(&mut self) -> Option<Self::Item> {
                if self.is_iter_empty() {
                    return None;
                }
                self.vec.get(self.index).map(|v| {
                    self.index += 1;
                    v
                })
            }

            /// Returns the number of elements that this iterator will iterate over. The size is
            /// precise.
            #[must_use]
            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.len(), Some(self.len()))
            }

            /// Returns the exact number of elements that this iterator would iterate over. Does not
            /// call `next` internally.
            #[must_use]
            fn count(self) -> usize
            where
                Self: Sized,
            {
                self.len()
            }

            /// Returns the last element of the iterator. Does not call `next` internally.
            #[must_use]
            fn last(self) -> Option<Self::Item>
            where
                Self: Sized,
            {
                if self.is_iter_empty() {
                    return None;
                }

                Some(self.vec.get_unchecked(*self.back_index.as_ref().unwrap()))
            }

            /// Returns the nth element of the iterator. Does not call `next` internally, but advances
            /// the iterator by `n` elements.
            fn nth(&mut self, n: usize) -> Option<Self::Item> {
                self.advance_by(n).ok()?;
                self.next()
            }

            $($($function_defs)*)*
        }

        impl $(<$life>)? std::iter::ExactSizeIterator for $name $(<$life>)? {
            fn len(&self) -> usize {
                // intentionally overflowing calculations to avoid branches on empty iterator
                (*self.back_index.as_ref().unwrap_or(&usize::MAX)).wrapping_sub(self.index).wrapping_add(1)
            }
        }

        impl $(<$life>)? std::iter::FusedIterator for $name $(<$life>)? {}

        impl $(<$life>)? std::iter::DoubleEndedIterator for $name $(<$life>)? {
            fn next_back(&mut self) -> Option<Self::Item> {
                if Some(self.index) > self.back_index {
                    return None;
                }
                self.vec.get(*self.back_index.as_ref().unwrap()).map(|v| {
                    if *self.back_index.as_ref().unwrap() == 0 {
                        self.back_index = None;
                    } else {
                        self.back_index = Some(self.back_index.unwrap() - 1);
                    }
                    v
                })
            }

            fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
                self.advance_back_by(n).ok()?;
                self.next_back()
            }
        }
    };
}

macro_rules! unroll {
    (1, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { let mut $i: usize = $e; $s };
    (2, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(1, |$i = {$e}| $s, $inc); $inc; $s };
    (3, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(2, |$i = {$e}| $s, $inc); $inc; $s };
    (4, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(3, |$i = {$e}| $s, $inc); $inc; $s };
    (5, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(4, |$i = {$e}| $s, $inc); $inc; $s };
    (6, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(5, |$i = {$e}| $s, $inc); $inc; $s };
    (7, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(6, |$i = {$e}| $s, $inc); $inc; $s };
    (8, |$i:ident = {$e:expr}| $s:stmt, $inc:expr) => { unroll!(7, |$i = {$e}| $s, $inc); $inc; $s };
}

/// Internal macro to implement iterators for the vector types.
/// This macro accepts more patterns than it should, but it isn't exported.
/// The macro accepts the name of the vector type as its first mandatory argument.
/// It then expects two identifiers for the two iterator types.
/// It then optionally accepts an entire unrestricted token tree, which it will paste into the
/// Iterator implementation.
/// This is useful for implementing functions that are only available on certain vector types.
/// Misuse of this token tree will result in compile errors regarding the Iterator trait.
///
/// The macro expects the vector type to implement a function `get_unchecked` that returns a
/// u64, a function `get` that returns an `Option<u64>` and a usize struct member `len`.
///
/// The macro generates the following items:
/// - A struct named `VecTypeIter` that implements `Iterator<Item = u64>` for `VecType`.
/// - A struct named `VecTypeRefIter` that implements `Iterator<Item = u64>` for `&VecType` and `$mut VecType`.
/// - An `impl` block for `VecType` that implements `IntoIterator<Item = u64>` for `VecType`.
/// - An `impl` block for `&VecType` that implements `IntoIterator<Item = u64>` for `&VecType`.
/// - An `impl` block for `&mut VecType` that implements `IntoIterator<Item = u64>` for `&mut VecType`.
/// - An `impl` block for `VecType` that implements `ExactSizeIterator` for `VecTypeIter`.
/// - An `impl` block for `&VecType` that implements `ExactSizeIterator` for `VecTypeRefIter`.
/// - An `impl` block for `&mut VecType` that implements `FuseIterator` for `VecTypeIter`.
/// - An `impl` block for `&mut VecType` that implements `FuseIterator` for `VecTypeRefIter`.
macro_rules! impl_iterator {
    ($type:ty, $own:ident, $bor:ident $(; $($function_defs:tt)*)*) => {
        #[doc = concat!("An owning iterator for `", stringify!($type), "`.")]
        #[doc = concat!("This struct is created by the `into_iter` trait implementation of `", stringify!($type), "`.")]
        #[derive(Clone, Debug)]
        pub struct $own {
            vec: $type,
            index: usize,
            // back index is none, iff it points to element -1 (i.e. element 0 has been consumed by
            // a call to next_back()). It can be Some(..) even if the iterator is empty
            back_index: Option<usize>,
        }

        impl $type {
            #[doc = concat!("Returns an iterator over the elements of `", stringify!($type), "`.")]
            #[must_use]
            pub fn iter(&self) -> $bor<'_> {
                $bor::new(self)
            }
        }

        crate::util::gen_iter_impl!($own, $type, u64; $({ $($function_defs)* })*);

        #[doc = concat!("A borrowing iterator for `", stringify!($type), "`.")]
        #[doc = concat!("This struct is created by the `iter` method of `", stringify!($type), "`.")]
        #[derive(Clone, Debug)]
        pub struct $bor<'a> {
            vec: &'a $type,
            index: usize,
            // back index is none, iff it points to element -1 (i.e. element 0 has been consumed by
            // a call to next_back()). It can be Some(..) even if the iterator is empty
            back_index: Option<usize>,
        }

        crate::util::gen_iter_impl!('a, $bor, $type, u64; $({ $($function_defs)* })*);

        impl IntoIterator for $type {
            type Item = u64;
            type IntoIter = $own;

            #[must_use]
            fn into_iter(self) -> Self::IntoIter {
                $own::new(self)
            }
        }

        impl<'a> IntoIterator for &'a $type {
            type Item = u64;
            type IntoIter = $bor<'a>;

            #[must_use]
            fn into_iter(self) -> Self::IntoIter {
                $bor::new(self)
            }
        }

        impl<'a> IntoIterator for &'a mut $type {
            type Item = u64;
            type IntoIter = $bor<'a>;

            #[must_use]
            fn into_iter(self) -> Self::IntoIter {
                $bor::new(self)
            }
        }
    }
}

pub(crate) use gen_iter_impl;
pub(crate) use impl_iterator;
pub(crate) use unroll;
