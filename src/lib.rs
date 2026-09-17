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
//! assert_eq!("13.45331", &format!("{d}"));
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
use num_traits::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, ToPrimitive, cast};

/// Maximum value is 92 233 720 368 547.75807
///
/// # Example
/// ```
/// assert_eq!("92233720368547.75807", &format!("{}", dec19x5::MAX));
/// ```
pub const MAX: Decimal = Decimal(i64::MAX);

/// Minimum value is -92 233 720 368 547.75808
///
/// # Example
/// ```
/// assert_eq!("-92233720368547.75808", &format!("{}", dec19x5::MIN));
/// ```
pub const MIN: Decimal = Decimal(i64::MIN);

const SCALE: i64 = 100_000;

const POW10: [u64; 14] = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
    10_000_000_000,
    100_000_000_000,
    1_000_000_000_000,
    10_000_000_000_000,
];

/// A Decimal type for integer calculation of financial amount.
///
/// # Note
///
/// The type is able to take 14 digits before the dot and 5 digits after the dot.
#[derive(Clone, Copy, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Decimal(i64);

/// `a * b / SCALE`, staying in i64 when the product fits (avoids the i128 division call).
#[inline]
fn mul_raw(a: i64, b: i64) -> i64 {
    match a.checked_mul(b) {
        Some(p) => p / SCALE,
        None => (a as i128 * b as i128 / SCALE as i128) as i64,
    }
}

/// `a * SCALE / b`, staying in i64 when the scaled dividend fits.
#[inline]
fn div_raw(a: i64, b: i64) -> i64 {
    match a.checked_mul(SCALE) {
        Some(p) => p / b,
        None => (a as i128 * SCALE as i128 / b as i128) as i64,
    }
}

/// Round half away from zero to a multiple of `D`; wraps on overflow like the i128->i64 cast did.
#[inline(always)]
fn round_by<const D: i64>(v: i64) -> i64 {
    let w = (v / D) * D;
    let r = v - w;

    if r >= D / 2 {
        w.wrapping_add(D)
    } else if -r >= D / 2 {
        w.wrapping_sub(D)
    } else {
        w
    }
}

/// Formats a raw value with `precision` (0..=5) fractional digits into the tail of `buf`.
fn format_raw(v: i64, precision: usize, buf: &mut [u8; 24]) -> &str {
    let mut int = (v / SCALE).unsigned_abs();
    let mut frac = (v % SCALE).unsigned_abs() / POW10[5 - precision];
    let mut pos = buf.len();

    if precision > 0 {
        for _ in 0..precision {
            pos -= 1;
            buf[pos] = b'0' + (frac % 10) as u8;
            frac /= 10;
        }
        pos -= 1;
        buf[pos] = b'.';
    }

    loop {
        pos -= 1;
        buf[pos] = b'0' + (int % 10) as u8;
        int /= 10;
        if int == 0 {
            break;
        }
    }

    if v < 0 {
        pos -= 1;
        buf[pos] = b'-';
    }

    std::str::from_utf8(&buf[pos..]).expect("ascii")
}

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
        Display::fmt(self, f)
    }
}

impl Decimal {
    /// Initialize a Decimal value using a value integer and a scale
    /// which represent the number of digit after the dot.
    ///
    /// # Panic
    ///
    /// A scale greater than 18 will cause a panic.
    #[inline]
    pub fn new_with_scale(value: i128, scale: u8) -> Self {
        assert!(scale < 19, "Scale {} is greater than 18", scale);

        Decimal(match scale.cmp(&5) {
            Ordering::Less => value * POW10[(5 - scale) as usize] as i128,
            Ordering::Greater => value / POW10[(scale - 5) as usize] as i128,
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
    #[inline]
    pub fn round(self) -> Decimal {
        self.round_n(0)
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
        self.round_n(2)
    }

    #[inline]
    pub fn round_n(self, dec: usize) -> Decimal {
        Decimal(match dec {
            0 => round_by::<100_000>(self.0),
            1 => round_by::<10_000>(self.0),
            2 => round_by::<1_000>(self.0),
            3 => round_by::<100>(self.0),
            4 => round_by::<10>(self.0),
            _ => return self,
        })
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
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde_crate::Deserializer<'de>,
    {
        use serde_crate::Deserialize;
        use serde_crate::de::{Error, MapAccess, Visitor};

        fn invalid<E: Error>() -> E {
            E::custom("invalid decimal")
        }

        struct DecimalVisitor;

        impl<'de> Visitor<'de> for DecimalVisitor {
            type Value = Decimal;

            fn expecting(&self, f: &mut Formatter) -> Result<(), FmtError> {
                f.write_str("a decimal number or string")
            }

            fn visit_str<E: Error>(self, v: &str) -> Result<Decimal, E> {
                Decimal::from_str(v).map_err(|_| invalid())
            }

            fn visit_i64<E: Error>(self, v: i64) -> Result<Decimal, E> {
                v.checked_mul(SCALE).map(Decimal).ok_or_else(invalid)
            }

            fn visit_u64<E: Error>(self, v: u64) -> Result<Decimal, E> {
                i64::try_from(v)
                    .ok()
                    .and_then(|v| v.checked_mul(SCALE))
                    .map(Decimal)
                    .ok_or_else(invalid)
            }

            fn visit_f64<E: Error>(self, v: f64) -> Result<Decimal, E> {
                // Go through the shortest round-trip repr so 16.65 parses as 16.65000, not 16.64999.
                match serde_json::Number::from_f64(v) {
                    Some(n) => number_to_decimal(&n),
                    None => Err(invalid()),
                }
            }

            // serde_json with `arbitrary_precision` delivers numbers as a map.
            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Decimal, A::Error> {
                let n = serde_json::Number::deserialize(
                    serde_crate::de::value::MapAccessDeserializer::new(map),
                )?;
                number_to_decimal(&n)
            }
        }

        deserializer.deserialize_any(DecimalVisitor)
    }
}

#[cfg(feature = "serde")]
fn number_to_decimal<E: serde_crate::de::Error>(n: &serde_json::Number) -> Result<Decimal, E> {
    use std::fmt::Write;

    struct Buf {
        bytes: [u8; 40],
        len: usize,
    }

    impl Write for Buf {
        fn write_str(&mut self, s: &str) -> Result<(), FmtError> {
            let end = self.len + s.len();
            let dst = self.bytes.get_mut(self.len..end).ok_or(FmtError)?;
            dst.copy_from_slice(s.as_bytes());
            self.len = end;
            Ok(())
        }
    }

    // Anything longer than the buffer cannot be a valid decimal anyway.
    let mut buf = Buf {
        bytes: [0; 40],
        len: 0,
    };

    write!(buf, "{n}")
        .ok()
        .and_then(|_| std::str::from_utf8(&buf.bytes[..buf.len]).ok())
        .and_then(|s| Decimal::from_str(s).ok())
        .ok_or_else(|| E::custom("invalid decimal"))
}

/// Format Decimal
///
/// # Example
/// ```
/// use dec19x5::Decimal;
/// use std::str::FromStr;
///
/// assert_eq!("10.01", format!("{:.2}", Decimal::from_str("10.014").unwrap()));
/// assert_eq!("10.02", format!("{:.2}", Decimal::from_str("10.015").unwrap()));
/// assert_eq!("10.01500", format!("{}", Decimal::from_str("10.015").unwrap()));
/// assert_eq!("-10.02", format!("{:.2}", Decimal::from_str("-10.015").unwrap()));
/// ```
impl Display for Decimal {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        let precision = f.precision().unwrap_or(5).min(5);
        let mut buf = [0u8; 24];
        f.write_str(format_raw(self.round_n(precision).0, precision, &mut buf))
    }
}

impl Div<Decimal> for Decimal {
    type Output = Decimal;

    #[inline]
    fn div(self, other: Decimal) -> Decimal {
        Decimal(div_raw(self.0, other.0))
    }
}

impl DivAssign<Decimal> for Decimal {
    #[inline]
    fn div_assign(&mut self, other: Decimal) {
        self.0 = div_raw(self.0, other.0)
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
    type Err = DecParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        fn parse_u64(s: &str) -> Result<u64, DecParseError> {
            u64::from_str(s).map_err(|_| DecParseError)
        }

        let mut s = s.trim();

        let neg = if s.starts_with('-') {
            s = s[1..].trim_start();
            true
        } else if s.ends_with('-') {
            s = s[..s.len() - 1].trim_end();
            true
        } else if s.starts_with('(') && s.ends_with(')') {
            s = s[1..s.len() - 1].trim();
            true
        } else {
            false
        };

        let (n, d) = match s.find('.') {
            Some(index) => {
                let ns = s[..index].trim_end();
                let ds = s[index + 1..].trim_start();

                let n = if ns.is_empty() { 0 } else { parse_u64(ns)? };

                let d = match ds.len() {
                    0 => {
                        if ns.is_empty() {
                            return Err(DecParseError);
                        }
                        0
                    }
                    len @ 1..=5 => parse_u64(ds)? * POW10[5 - len],
                    // up to 16 digits for javascript
                    len @ 6..=16 => parse_u64(ds)? / POW10[len - 5],
                    _ => return Err(DecParseError),
                };

                (n, d)
            }
            None => (parse_u64(s)?, 0),
        };

        if n > 92233720368547 || (n == 92233720368547 && (d > 75808 || (!neg && d > 75807))) {
            return Err(DecParseError);
        }

        // d < 100_000 here, so this fits in u64; the only value >= 2^63 is |MIN|, handled by wrapping_neg.
        let mag = n * SCALE as u64 + d;
        let v = if neg {
            (mag as i64).wrapping_neg()
        } else {
            mag as i64
        };

        Ok(Decimal(v))
    }
}

impl Mul<Decimal> for Decimal {
    type Output = Decimal;

    #[inline]
    fn mul(self, other: Decimal) -> Decimal {
        Decimal(mul_raw(self.0, other.0))
    }
}

impl MulAssign<Decimal> for Decimal {
    #[inline]
    fn mul_assign(&mut self, other: Decimal) {
        self.0 = mul_raw(self.0, other.0);
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
        let mut buf = [0u8; 24];
        let s = format_raw(self.0, 5, &mut buf);

        match serde_json::from_str::<serde_json::Number>(s) {
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

pub struct DecParseError;

impl Debug for DecParseError {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        f.write_str("Unparsable as decimal.")
    }
}

impl Display for DecParseError {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        Debug::fmt(self, f)
    }
}

impl std::error::Error for DecParseError {}

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
                ));
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

        assert_eq!(
            Decimal::from(16),
            serde_json::from_str::<Decimal>("16").unwrap()
        );

        assert_eq!(
            Decimal::from(-16),
            serde_json::from_str::<Decimal>("-16").unwrap()
        );

        assert_eq!(
            MAX,
            serde_json::from_str::<Decimal>("\"92233720368547.75807\"").unwrap()
        );
        assert_eq!(
            MIN,
            serde_json::from_str::<Decimal>("\"-92233720368547.75808\"").unwrap()
        );

        assert!(serde_json::from_str::<Decimal>("92233720368548").is_err());
        assert!(serde_json::from_str::<Decimal>("-92233720368548").is_err());
        assert!(serde_json::from_str::<Decimal>("18446744073709551615").is_err());
        assert!(serde_json::from_str::<Decimal>("1e300").is_err());
        assert!(serde_json::from_str::<Decimal>("true").is_err());
    }

    #[test]
    fn display() {
        assert_eq!("0.12000", format!("{}", Decimal::from(0.12)));
        assert_eq!("-3.00000", format!("{}", Decimal::from(-3)));
        assert_eq!("-0.30000", format!("{}", Decimal::from(-0.3)));
        assert_eq!("0.00000", format!("{}", Decimal::zero()));
        assert_eq!("-0.00001", format!("{}", Decimal(-1)));
        assert_eq!("0", format!("{:.0}", Decimal(-40000)));
        assert_eq!("-1", format!("{:.0}", Decimal(-50000)));
        assert_eq!("-0.4", format!("{:.1}", Decimal(-40000)));
        assert_eq!(
            "12.346",
            format!("{:.3}", Decimal::from_str("12.3456").unwrap())
        );
        assert_eq!(
            "12.34560",
            format!("{:.9}", Decimal::from_str("12.3456").unwrap())
        );
        assert_eq!("92233720368547.75807", format!("{}", MAX));
        assert_eq!("-92233720368547.75808", format!("{}", MIN));
        assert_eq!(format!("{}", MIN), format!("{:?}", MIN));
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

        assert_eq!(
            Decimal::new_with_scale(-1, 2),
            Decimal::from_str("-0.01").unwrap()
        );

        assert_eq!(
            Decimal::new_with_scale(-1, 2),
            Decimal::from_str("- 0.01").unwrap()
        );

        assert_eq!(
            Decimal::new_with_scale(-1, 2),
            Decimal::from_str("0.01-").unwrap()
        );

        assert_eq!(
            Decimal::new_with_scale(-1, 2),
            Decimal::from_str("(0.01)").unwrap()
        );

        assert!(Decimal::from_str("").is_err());
        assert!(Decimal::from_str(".").is_err());
        assert!(Decimal::from_str("2.-2").is_err());
        assert!(Decimal::from_str("2..2").is_err());
        assert!(Decimal::from_str("--2").is_err());

        assert_eq!(Decimal::from_str("92233720368547.75807").unwrap(), MAX);
        assert_eq!(Decimal::from_str("-92233720368547.75808").unwrap(), MIN);

        assert_eq!(
            Decimal::from_str("92233720368547.70007").unwrap(),
            Decimal::new_with_scale(9223372036854770007, 5)
        );

        assert_eq!(
            Decimal::from_str("-92233720368547.00001").unwrap(),
            Decimal::new_with_scale(-9223372036854700001, 5)
        );

        let expected: Decimal = 2.into();
        assert_eq!(expected, Decimal::from_str("2.").unwrap());

        let expected: Decimal = 0.2.into();
        assert_eq!(expected, Decimal::from_str(".2").unwrap());

        assert_eq!(
            Decimal::new_with_scale(123049, 5),
            Decimal::from_str("1.230499123123").unwrap()
        );

        assert_eq!(
            Decimal::new_with_scale(-175808, 5),
            Decimal::from_str("-1.75808").unwrap()
        );

        assert_eq!(
            Decimal::new_with_scale(-9223372036854775807, 5),
            Decimal::from_str("-92233720368547.75807").unwrap()
        );

        assert!(Decimal::from_str("92233720368547.75808").is_err());
        assert!(Decimal::from_str("-92233720368547.75809").is_err());
        assert!(Decimal::from_str("92233720368548").is_err());
    }

    #[test]
    fn mul_div_large() {
        // Products/dividends that overflow i64 must take the i128 path and match exact math.
        let x = Decimal::from(50_000_000_000i64);
        let y = Decimal::from(1_000_000i64);
        assert_eq!(Decimal::from(50_000_000_000i64), Decimal::from(50_000) * y);
        assert_eq!(Decimal::from(50_000i64), x / y);

        let big = Decimal::from(92_233_720_368_547i64);
        assert_eq!(big, big * 1);
        assert_eq!(big, big / 1);
        assert_eq!(Decimal::from(-92_233_720_368_547i64), big / -1);
        assert_eq!(Decimal::from_str("46116860184273.5").unwrap(), big / 2);
        assert_eq!(Decimal::from_str("0.5").unwrap(), Decimal::from(1) / 2);
        assert_eq!(
            Decimal::from_str("-0.33333").unwrap(),
            Decimal::from(-1) / 3
        );
        assert_eq!(Decimal::from_str("1.5").unwrap(), 3 * Decimal::from(0.5));
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

    #[cfg(feature = "serde")]
    #[test]
    fn test_limits_serialize() {
        fn check_serialize(d: Decimal) {
            assert_eq!(
                serde_json::from_str::<Decimal>(&serde_json::to_string(&d).unwrap()).unwrap(),
                d
            );
        }

        check_serialize(Decimal::from(92233720368547i64));
        check_serialize(Decimal::from(-92233720368547i64));
    }
}
