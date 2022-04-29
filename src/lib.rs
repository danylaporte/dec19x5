//! Represent a 64-bit decimal type taking 14 digits before the dot and 5 digits after the dot.
//!
//! The calculation uses integer math to ensure accurate financial amounts.
//!
//! # Example
//! ```
//! use dec19x5::Decimal;
//! use std::str::FromStr;
//!
//! // parse from string
//! let d = Decimal::from_str("13.45331").unwrap();
//!
//! // write to string
//! assert_eq!("13.45331", &format!("{}", d));
//!
//! // load an i32
//! let d = Decimal::from(10i32);
//!
//! // multiple ops are supported.
//! assert_eq!(d + Decimal::from(3i32), Decimal::from(13i32));
//! ```
use std::cmp::Ordering;
use std::fmt::{Debug, Display, Error as FmtError, Formatter};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::str::FromStr;

mod atomic;

pub use atomic::AtomicDecimal;

#[cfg(feature = "num-traits")]
use num_traits::{cast, CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, ToPrimitive};

/// Maximum value is 92 233 720 368 547.75807
///
/// # Example
/// ```
/// assert_eq!("92233720368547.75807", &format!("{}", dec19x5::MAX));
/// ```
pub const MAX: Decimal = Decimal(std::i64::MAX);

/// Minimum value is -92 233 720 368 547.75808
///
/// # Example
/// ```
/// assert_eq!("-92233720368547.75808", &format!("{}", dec19x5::MIN));
/// ```
pub const MIN: Decimal = Decimal(std::i64::MIN);

/// A Decimal type for integer calculation of financial amount.
///
/// # Note
///
/// The type is able to take 14 digits before the dot and 5 digits after the dot.
#[derive(Clone, Copy, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Decimal(i64);

impl Add<Decimal> for Decimal {
    type Output = Decimal;

    #[inline]
    fn add(self, other: Decimal) -> Decimal {
        Decimal(self.0 + other.0)
    }
}

impl AddAssign<Decimal> for Decimal {
    #[inline]
    fn add_assign(&mut self, other: Decimal) {
        self.0 += other.0
    }
}

impl Debug for Decimal {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        write!(f, "{}", self)
    }
}

impl Decimal {
    /// Initialize a Decimal value using a value integer and a scale
    /// which represent the number of digit after the dot.
    ///
    /// # Panic
    ///
    /// A scale greater than 18 will cause a panic.
    pub fn new_with_scale(value: i128, scale: u8) -> Self {
        assert!(scale < 19, "Scale {} is greater than 18", scale);

        Decimal(match scale.cmp(&5) {
            Ordering::Less => value * 10i128.pow((5 - scale) as u32),
            Ordering::Greater => value / 10i128.pow((scale - 5) as u32),
            Ordering::Equal => value,
        } as _)
    }

    /// Computes the absolute value of self.
    ///
    /// # Overflow behavior
    /// The absolute value of cannot be represented as an and attempting to calculate it will cause an overflow.
    /// This means that code in debug mode will trigger a panic on this case and optimized code will return without a panic.
    #[inline]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Returns true if self is negative and false if the number is zero or positive.
    #[inline]
    pub fn is_negative(self) -> bool {
        self.0.is_negative()
    }

    /// Returns true if self is positive and false if the number is zero or negative.
    #[inline]
    pub fn is_positive(self) -> bool {
        self.0.is_positive()
    }

    /// Returns true if the decimal is zero.
    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Returns the nearest integer to a number. Round half-way cases away from 0.0.
    ///
    /// # Example
    /// ```
    /// use dec19x5::Decimal;
    ///
    /// let f: Decimal = 3.3.into();
    /// let g: Decimal = 3.6.into();
    /// let h: Decimal = (-3.3).into();
    /// let i: Decimal = (-3.6).into();
    ///
    /// assert_eq!(f.round(), 3);
    /// assert_eq!(g.round(), 4);
    /// assert_eq!(h.round(), -3);
    /// assert_eq!(i.round(), -4);
    /// ```
    pub fn round(self) -> Decimal {
        let v = self.0 as i128;
        let w = (v / 100000) * 100000;

        let w = if v - w >= 50000 {
            w + 100000
        } else if w - v >= 50000 {
            w - 100000
        } else {
            w
        };

        Decimal(w as i64)
    }

    /// round to 2 digits
    ///
    /// # Example
    /// ```
    /// use dec19x5::Decimal;
    ///
    /// assert_eq!("200.50000", &format!("{}", Decimal::from(200.49999).round_2()));
    /// assert_eq!("200.48000", &format!("{}", Decimal::from(200.48499).round_2()));
    /// assert_eq!("201.00000", &format!("{}", Decimal::from(200.99999).round_2()));
    /// ```
    #[inline]
    pub fn round_2(self) -> Decimal {
        let v = self.0 as i128;
        let w = (v / 1000) * 1000;

        let w = if v - w >= 500 {
            w + 1000
        } else if w - v >= 500 {
            w - 1000
        } else {
            w
        };

        Decimal(w as i64)
    }

    pub const fn scale(&self) -> u8 {
        5
    }

    pub fn value(&self) -> i128 {
        self.0 as i128
    }

    /// Returns a Decimal value of zero.
    #[inline]
    pub const fn zero() -> Self {
        Decimal(0)
    }
}

#[cfg(feature = "num-traits")]
impl CheckedAdd for Decimal {
    #[inline]
    fn checked_add(&self, v: &Self) -> Option<Self> {
        Some(Self(self.0.checked_add(v.0)?))
    }
}

#[cfg(feature = "num-traits")]
impl CheckedDiv for Decimal {
    fn checked_div(&self, v: &Self) -> Option<Self> {
        if v.is_zero() {
            None
        } else {
            let v = (self.0 as i128)
                .checked_mul(100000)?
                .checked_div(v.0 as i128)?;

            Some(Self(cast(v)?))
        }
    }
}

#[cfg(feature = "num-traits")]
impl CheckedMul for Decimal {
    fn checked_mul(&self, v: &Self) -> Option<Self> {
        let v = (self.0 as i128).checked_mul(v.0 as i128)? / 100000;
        Some(Self(cast(v)?))
    }
}

#[cfg(feature = "num-traits")]
impl CheckedSub for Decimal {
    #[inline]
    fn checked_sub(&self, v: &Self) -> Option<Self> {
        Some(Self(self.0.checked_sub(v.0)?))
    }
}

#[cfg(feature = "num-traits")]
impl num_traits::One for Decimal {
    #[inline]
    fn one() -> Self {
        Self(100000)
    }

    #[inline]
    fn set_one(&mut self) {
        self.0 = 100000;
    }

    #[inline]
    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        self.0 == 100000
    }
}

#[cfg(feature = "num-traits")]
impl ToPrimitive for Decimal {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        Some(self.0 / 100000)
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        (self.0 / 100000).to_u64()
    }

    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.to_f64()?.to_f32()
    }

    #[inline]
    fn to_f64(&self) -> Option<f64> {
        let v: f64 = cast(self.0)?;
        Some(v / 100000.0)
    }
}

#[cfg(feature = "num-traits")]
impl num_traits::Zero for Decimal {
    #[inline]
    fn zero() -> Self {
        Decimal::zero()
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == 0
    }

    #[inline]
    fn set_zero(&mut self) {
        self.0 = 0;
    }
}

#[cfg(feature = "serde")]
impl<'de> serde_crate::Deserialize<'de> for Decimal {
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde_crate::Deserializer<'de>,
    {
        match serde_crate::Deserialize::<'de>::deserialize(deserializer)? {
            DecimalDe::String(s) => s
                .parse()
                .map_err(|e| serde_crate::de::Error::custom(format!("invalid decimal: {}", e))),
            DecimalDe::Number(n) => n
                .to_string()
                .parse()
                .map_err(|e| serde_crate::de::Error::custom(format!("invalid decimal: {}", e))),
        }
    }
}

#[cfg(feature = "serde")]
#[derive(serde_crate::Deserialize)]
#[serde(untagged, crate = "serde_crate")]
enum DecimalDe {
    String(String),
    Number(serde_json::Number),
}

impl Display for Decimal {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        let v = self.0 / 100000;
        let d = self.0 - v * 100000;
        if self.0 < 0 {
            write!(f, "-{}.{:05}", -v, -d)
        } else {
            write!(f, "{}.{:05}", v, d)
        }
    }
}

impl Div<Decimal> for Decimal {
    type Output = Decimal;

    #[inline]
    fn div(self, other: Decimal) -> Decimal {
        Decimal((self.0 as i128 * 100000 / other.0 as i128) as i64)
    }
}

impl DivAssign<Decimal> for Decimal {
    #[inline]
    fn div_assign(&mut self, other: Decimal) {
        self.0 = (self.0 as i128 * 100000 / other.0 as i128) as i64
    }
}

impl From<Decimal> for i8 {
    #[inline]
    fn from(d: Decimal) -> i8 {
        (d.0 / 100000) as i8
    }
}

impl From<Decimal> for i16 {
    #[inline]
    fn from(d: Decimal) -> i16 {
        (d.0 / 100000) as i16
    }
}

impl From<Decimal> for i32 {
    #[inline]
    fn from(d: Decimal) -> i32 {
        (d.0 / 100000) as i32
    }
}

impl From<Decimal> for i64 {
    #[inline]
    fn from(d: Decimal) -> i64 {
        d.0 / 100000
    }
}

impl From<Decimal> for f32 {
    #[inline]
    fn from(d: Decimal) -> f32 {
        d.0 as f32 / 100000.0
    }
}

impl From<Decimal> for f64 {
    #[inline]
    fn from(d: Decimal) -> f64 {
        d.0 as f64 / 100000.0
    }
}

impl From<Decimal> for u8 {
    #[inline]
    fn from(d: Decimal) -> u8 {
        (d.0 / 100000) as u8
    }
}

impl From<Decimal> for u16 {
    #[inline]
    fn from(d: Decimal) -> u16 {
        (d.0 / 100000) as u16
    }
}

impl From<Decimal> for u32 {
    #[inline]
    fn from(d: Decimal) -> u32 {
        (d.0 / 100000) as u32
    }
}

impl From<Decimal> for u64 {
    #[inline]
    fn from(d: Decimal) -> u64 {
        (d.0 / 100000) as u64
    }
}

impl From<i8> for Decimal {
    #[inline]
    fn from(i: i8) -> Decimal {
        Decimal(i64::from(i) * 100000)
    }
}

impl From<i16> for Decimal {
    #[inline]
    fn from(i: i16) -> Decimal {
        Decimal(i64::from(i) * 100000)
    }
}

impl From<i32> for Decimal {
    #[inline]
    fn from(i: i32) -> Decimal {
        Decimal(i64::from(i) * 100000)
    }
}

impl From<i64> for Decimal {
    #[inline]
    fn from(i: i64) -> Decimal {
        Decimal(i * 100000)
    }
}

impl From<isize> for Decimal {
    #[inline]
    fn from(i: isize) -> Decimal {
        Decimal(i as i64 * 100000)
    }
}

impl From<f32> for Decimal {
    #[inline]
    fn from(i: f32) -> Decimal {
        Decimal::from(i as f64)
    }
}

impl From<f64> for Decimal {
    #[inline]
    fn from(i: f64) -> Decimal {
        Decimal((i * 100000.0) as i64)
    }
}

impl From<u8> for Decimal {
    #[inline]
    fn from(i: u8) -> Decimal {
        Decimal(i64::from(i) * 100000)
    }
}

impl From<u16> for Decimal {
    #[inline]
    fn from(i: u16) -> Decimal {
        Decimal(i64::from(i) * 100000)
    }
}

impl From<u32> for Decimal {
    #[inline]
    fn from(i: u32) -> Decimal {
        Decimal(i64::from(i) * 100000)
    }
}

impl From<u64> for Decimal {
    #[inline]
    fn from(i: u64) -> Decimal {
        Decimal(i as i64 * 100000)
    }
}

impl From<usize> for Decimal {
    #[inline]
    fn from(i: usize) -> Decimal {
        Decimal(i as i64 * 100000)
    }
}

impl FromStr for Decimal {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let nv: Result<i64, _>;
        let dv: Result<i64, _>;
        let s = s.trim();

        if let Some((index, _)) = s.char_indices().find(|(_, c)| c == &'.') {
            let (n, d) = s.split_at(index);

            if d.chars().any(|c| c == '-') {
                return Err(Error::Parse(s.to_owned()));
            }

            let (n, d) = (n.trim_end(), d.trim_start());
            let d = &d[1..];
            let d = if d.len() > 5 { &d[..5] } else { d };

            dv = match d.len() {
                0 => Ok(0),
                1 => d.parse::<i64>().map(|d| d * 10000),
                2 => d.parse::<i64>().map(|d| d * 1000),
                3 => d.parse::<i64>().map(|d| d * 100),
                4 => d.parse::<i64>().map(|d| d * 10),
                _ => d.parse::<i64>(),
            };

            if n.is_empty() && !d.is_empty() {
                nv = Ok(0);
            } else {
                nv = n.parse();
            }
        } else {
            nv = s.parse();
            dv = Ok(0);
        }

        if let (Ok(n), Ok(d)) = (nv, dv) {
            let d1 = d / 100000;
            let d = d - d1 * 100000;

            Ok(Decimal(n * 100000 + if n >= 0 { d } else { -d }))
        } else {
            Err(Error::Parse(s.to_owned()))
        }
    }
}

impl Mul<Decimal> for Decimal {
    type Output = Decimal;

    #[inline]
    fn mul(self, other: Decimal) -> Decimal {
        Decimal((self.0 as i128 * other.0 as i128 / 100000) as i64)
    }
}

impl MulAssign<Decimal> for Decimal {
    #[inline]
    fn mul_assign(&mut self, other: Decimal) {
        self.0 = ((self.0 as i128 * other.0 as i128) / 100000) as i64;
    }
}

impl Neg for Decimal {
    type Output = Decimal;

    #[inline]
    fn neg(self) -> Decimal {
        Decimal(-self.0)
    }
}

#[cfg(feature = "serde")]
impl serde_crate::Serialize for Decimal {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde_crate::Serializer,
    {
        let s = self.to_string();

        match serde_json::from_str::<serde_json::Number>(&s) {
            Ok(n) => n.serialize(serializer),
            Err(_) =>
            // lossy precision here
            {
                serializer.serialize_f64((*self).into())
            }
        }
    }
}

impl Sub<Decimal> for Decimal {
    type Output = Decimal;

    #[inline]
    fn sub(self, other: Decimal) -> Decimal {
        Decimal(self.0 - other.0)
    }
}

impl SubAssign<Decimal> for Decimal {
    #[inline]
    fn sub_assign(&mut self, other: Decimal) {
        self.0 -= other.0
    }
}

impl Sum for Decimal {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Decimal>,
    {
        let mut d = Decimal(0);
        for item in iter {
            d += item;
        }
        d
    }
}

pub enum Error {
    Parse(String),
}

impl Debug for Error {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        match self {
            Error::Parse(s) => {
                f.write_str("Value \"")?;
                f.write_str(s)?;
                f.write_str("\" cannot be parsed as decimal.")?;
            }
        }

        Ok(())
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for Error {}

#[cfg(feature = "tiberius")]
impl<'a> tiberius::FromSql<'a> for Decimal {
    fn from_sql(value: &'a tiberius::ColumnData<'static>) -> tiberius::Result<Option<Self>> {
        fn opt<V: Copy + Into<Decimal>>(v: &Option<V>) -> Option<Decimal> {
            v.as_ref().map(|v| (*v).into())
        }

        Ok(match value {
            tiberius::ColumnData::F32(f) => opt(f),
            tiberius::ColumnData::F64(f) => opt(f),
            tiberius::ColumnData::I16(i) => opt(i),
            tiberius::ColumnData::I32(i) => opt(i),
            tiberius::ColumnData::I64(i) => opt(i),
            tiberius::ColumnData::Numeric(Some(n)) => {
                Some(Decimal::new_with_scale(n.value(), n.scale()))
            }
            tiberius::ColumnData::Numeric(None) => None,
            tiberius::ColumnData::U8(u) => opt(u),
            _ => {
                return Err(tiberius::error::Error::Conversion(
                    "Not convertable to decimal.".into(),
                ))
            }
        })
    }
}

#[cfg(feature = "tiberius")]
impl tiberius::ToSql for Decimal {
    fn to_sql(&self) -> tiberius::ColumnData<'_> {
        tiberius::ColumnData::Numeric(Some(tiberius::numeric::Numeric::new_with_scale(
            self.value(),
            self.scale(),
        )))
    }
}

macro_rules! cmp_ops {
    ($ty:ty) => {
        impl Add<$ty> for Decimal {
            type Output = Decimal;

            #[inline]
            fn add(self, other: $ty) -> Decimal {
                self + Decimal::from(other)
            }
        }

        impl Add<Decimal> for $ty {
            type Output = Decimal;

            #[inline]
            fn add(self, other: Decimal) -> Decimal {
                Decimal::from(self) + other
            }
        }

        impl AddAssign<$ty> for Decimal {
            #[inline]
            fn add_assign(&mut self, other: $ty) {
                *self += Decimal::from(other)
            }
        }

        impl Div<$ty> for Decimal {
            type Output = Decimal;

            #[inline]
            fn div(self, other: $ty) -> Decimal {
                self / Decimal::from(other)
            }
        }

        impl Div<Decimal> for $ty {
            type Output = Decimal;

            #[inline]
            fn div(self, other: Decimal) -> Decimal {
                Decimal::from(self) / other
            }
        }

        impl DivAssign<$ty> for Decimal {
            #[inline]
            fn div_assign(&mut self, other: $ty) {
                *self /= Decimal::from(other)
            }
        }

        impl Mul<$ty> for Decimal {
            type Output = Decimal;

            #[inline]
            fn mul(self, other: $ty) -> Decimal {
                self * Decimal::from(other)
            }
        }

        impl Mul<Decimal> for $ty {
            type Output = Decimal;

            #[inline]
            fn mul(self, other: Decimal) -> Decimal {
                Decimal::from(self) * other
            }
        }

        impl MulAssign<$ty> for Decimal {
            #[inline]
            fn mul_assign(&mut self, other: $ty) {
                *self *= Decimal::from(other)
            }
        }

        impl PartialEq<$ty> for Decimal {
            #[inline]
            fn eq(&self, other: &$ty) -> bool {
                self == &Decimal::from(*other)
            }
        }

        impl PartialEq<Decimal> for $ty {
            #[inline]
            fn eq(&self, other: &Decimal) -> bool {
                &Decimal::from(*self) == other
            }
        }

        impl PartialOrd<$ty> for Decimal {
            #[inline]
            fn partial_cmp(&self, other: &$ty) -> Option<Ordering> {
                self.partial_cmp(&Decimal::from(*other))
            }
        }

        impl PartialOrd<Decimal> for $ty {
            #[inline]
            fn partial_cmp(&self, other: &Decimal) -> Option<Ordering> {
                Decimal::from(*self).partial_cmp(other)
            }
        }

        impl Sub<$ty> for Decimal {
            type Output = Decimal;

            #[inline]
            fn sub(self, other: $ty) -> Decimal {
                self - Decimal::from(other)
            }
        }

        impl Sub<Decimal> for $ty {
            type Output = Decimal;

            #[inline]
            fn sub(self, other: Decimal) -> Decimal {
                Decimal::from(self) - other
            }
        }

        impl SubAssign<$ty> for Decimal {
            #[inline]
            fn sub_assign(&mut self, other: $ty) {
                *self -= Decimal::from(other)
            }
        }
    };
}

cmp_ops!(f32);
cmp_ops!(f64);
cmp_ops!(i8);
cmp_ops!(i16);
cmp_ops!(i32);
cmp_ops!(i64);
cmp_ops!(isize);
cmp_ops!(u8);
cmp_ops!(u16);
cmp_ops!(u32);
cmp_ops!(u64);
cmp_ops!(usize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let x: Decimal = 1.32.into();
        let y: Decimal = 2.54.into();
        let expected: Decimal = 3.86.into();
        assert_eq!(expected, x + y);
    }

    #[test]
    fn add_assign() {
        let mut x: Decimal = 1.32.into();
        let y: Decimal = 2.54.into();
        x += y;

        let expected: Decimal = 3.86.into();
        assert_eq!(expected, x);
    }

    #[test]
    fn abs() {
        let actual = Decimal::from(-100).abs();
        let expected = Decimal::from(100);
        assert_eq!(expected, actual);
    }

    #[cfg(feature = "num-traits")]
    #[test]
    fn checked_add() {
        let x: Decimal = 1.32.into();
        let y: Decimal = 2.54.into();
        let expected: Decimal = 3.86.into();
        assert_eq!(Some(expected), x.checked_add(&y));
    }

    #[cfg(feature = "num-traits")]
    #[test]
    fn checked_div() {
        let x: Decimal = 1.32.into();
        let y: Decimal = 2.54.into();

        let expected = Decimal::from_str("0.51968").unwrap();
        assert_eq!(Some(expected), x.checked_div(&y));

        let x: Decimal = 1_000_000_000.into();
        let expected: Decimal = 1.into();

        assert_eq!(Some(expected), x.checked_div(&x));

        let x: Decimal = 100.into();
        let y = Decimal::zero();

        assert_eq!(None, x.checked_div(&y));
    }

    #[cfg(feature = "num-traits")]
    #[test]
    fn checked_mul() {
        let x: Decimal = 1.32.into();
        let y: Decimal = 2.54.into();

        let expected: Decimal = 3.3528.into();
        assert_eq!(Some(expected), x.checked_mul(&y));

        let x: Decimal = 1_000_000.into();
        let expected: Decimal = 1_000_000_000_000i64.into();

        assert_eq!(Some(expected), x.checked_mul(&x));
    }

    #[cfg(feature = "num-traits")]
    #[test]
    fn checked_sub() {
        let x: Decimal = 1.32.into();
        let y: Decimal = 2.54.into();

        let expected: Decimal = (-1.22).into();
        assert_eq!(Some(expected), x.checked_sub(&y));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialize() {
        assert_eq!(
            Decimal::new_with_scale(254, 2),
            serde_json::from_str::<Decimal>("2.54").unwrap()
        );

        assert_eq!(
            Decimal::new_with_scale(1665, 2),
            serde_json::from_str::<Decimal>("\"16.65\"").unwrap()
        );

        assert_eq!(
            Decimal::new_with_scale(1665, 2),
            serde_json::from_str::<Decimal>("16.65").unwrap()
        );

        assert_eq!(
            Decimal::new_with_scale(-1665, 2),
            serde_json::from_str::<Decimal>("-16.65").unwrap()
        );
    }

    #[test]
    fn display() {
        assert_eq!("0.12000", format!("{}", Decimal::from(0.12)));
        assert_eq!("-3.00000", format!("{}", Decimal::from(-3)));
        assert_eq!("-0.30000", format!("{}", Decimal::from(-0.3)));
    }

    #[test]
    fn div() {
        let x: Decimal = 1.32.into();
        let y: Decimal = 2.54.into();

        let expected = Decimal::from_str("0.51968").unwrap();
        assert_eq!(expected, x / y);

        let x: Decimal = 1_000_000_000.into();
        let expected: Decimal = 1.into();

        assert_eq!(expected, x / x);
    }

    #[test]
    fn div_assign() {
        let mut x: Decimal = 1.32.into();
        let y: Decimal = 2.54.into();
        x /= y;

        let expected = Decimal::from_str("0.51968").unwrap();
        assert_eq!(expected, x);

        let mut x: Decimal = 1_000_000_000.into();
        let expected: Decimal = 1.into();
        x /= x;
        assert_eq!(expected, x);
    }

    #[test]
    fn from_str() {
        let expected: Decimal = 1.2332.into();
        assert_eq!(expected, Decimal::from_str("1.2332").unwrap());

        let expected: Decimal = (-2).into();
        assert_eq!(expected, Decimal::from_str("-2").unwrap());

        assert!(Decimal::from_str("").is_err());
        assert!(Decimal::from_str(".").is_err());
        assert!(Decimal::from_str("2.-2").is_err());

        let expected: Decimal = 2.into();
        assert_eq!(expected, Decimal::from_str("2.").unwrap());

        let expected: Decimal = 0.2.into();
        assert_eq!(expected, Decimal::from_str(".2").unwrap());
    }

    #[test]
    fn mul() {
        let x: Decimal = 1.32.into();
        let y: Decimal = 2.54.into();

        let expected: Decimal = 3.3528.into();
        assert_eq!(expected, x * y);

        let x: Decimal = 1_000_000.into();
        let expected: Decimal = 1_000_000_000_000i64.into();

        assert_eq!(expected, x * x);
    }

    #[test]
    fn mul_assign() {
        let mut x: Decimal = 1.32.into();
        let y: Decimal = 2.54.into();
        x *= y;

        let expected: Decimal = 3.3528.into();
        assert_eq!(expected, x);

        let mut x: Decimal = 1_000_000.into();
        let expected: Decimal = 1_000_000_000_000i64.into();

        x *= x;

        assert_eq!(expected, x);
    }

    #[test]
    fn round() {
        assert_eq!(
            "92233720368547.00000",
            &format!("{}", Decimal(9223372036854749999).round())
        );
        assert_eq!(
            "-92233720368547.00000",
            &format!("{}", Decimal(-9223372036854749999).round())
        );
    }

    #[test]
    fn round2() {
        assert_eq!(
            "92233720368547.50000",
            &format!("{}", Decimal(9223372036854749999).round_2())
        );
        assert_eq!(
            "-92233720368547.50000",
            &format!("{}", Decimal(-9223372036854749999).round_2())
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialize() {
        let x: Decimal = 2.54.into();
        let actual = serde_json::to_string(&x).unwrap();
        assert_eq!("2.54", actual);
    }

    #[test]
    fn sub() {
        let x: Decimal = 1.32.into();
        let y: Decimal = 2.54.into();

        let expected: Decimal = (-1.22).into();
        assert_eq!(expected, x - y);
    }

    #[test]
    fn sub_assign() {
        let mut x: Decimal = 1.32.into();
        let y: Decimal = 2.54.into();
        x -= y;

        let expected: Decimal = (-1.22).into();
        assert_eq!(expected, x);
    }

    #[test]
    fn sum() {
        let actual: Decimal = (0..5).into_iter().map(|_| -> Decimal { 1.into() }).sum();
        assert_eq!(5, actual);
    }

    #[test]
    fn test_limits() {
        let x: Decimal = (MAX - 0.00001 + 0.00001) * 1 / 1;
        assert_eq!(MAX, x);
    }
}
