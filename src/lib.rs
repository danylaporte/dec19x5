//! Represent a 64-bit decimal type taking 14 digits before the dot and 5 digits after the dot.
//!
//! The calculation uses integer math to ensure accurate financial amounts.
//!
//! # Example
//! ```
//! use dec19x5::Decimal;
//! use std::str::FromStr;
//!
//! fn main() {
//!     // parse from string
//!     let d = Decimal::from_str("13.45331").unwrap();
//!
//!     // write to string
//!     assert_eq!("13.45331", &format!("{}", d));
//!
//!     // load an i32
//!     let d = Decimal::from(10i32);
//!
//!     // multiple ops are supported.
//!     assert_eq!(d + Decimal::from(3i32), Decimal::from(13i32));
//! }
//! ```
use std::fmt::{Debug, Display, Error as FmtError, Formatter};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::str::FromStr;

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
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        let v = self.0 / 100000;
        let d = self.0 - v * 100000;
        write!(f, "{}.{:05}", v, d)
    }
}

impl Decimal {
    /// Initialize a Decimal value using a value integer and a scale
    /// which represent the number of digit after the dot.
    /// 
    /// # Panic
    /// 
    /// A scale greater than 18 will cause a panic.
    pub fn new_with_scale(mut value: i64, scale: u8) -> Self {
        assert!(scale < 19, "Scale {} is greater than 18", scale);

        if scale < 5 {
            value *= 10i64.pow((5 - scale) as u32);
        } else if scale > 5 {
            value /= 10i64.pow((scale - 5) as u32);
        }

        Decimal(value)
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
    /// assert_eq!(f.round(), 3.into());
    /// assert_eq!(g.round(), 4.into());
    /// assert_eq!(h.round(), (-3).into());
    /// assert_eq!(i.round(), (-4).into());
    /// ```
    pub fn round(mut self) -> Decimal {
        let v = self.0;
        self.0 = (self.0 / 100000) * 100000;

        if v - self.0 >= 50000 {
            self.0 += 100000
        } else if self.0 - v >= 50000 {
            self.0 -= 100000
        }

        self
    }

    /// round to 2 digits
    #[inline]
    pub fn round_2(self) -> Decimal {
        Decimal(Decimal(self.0 * 100).round().0 / 100)
    }

    /// Returns a Decimal value of zero.
    #[inline]
    pub fn zero() -> Self {
        Decimal(0)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Decimal {
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // lossy precision here
        Ok(f64::deserialize(deserializer)?.into())
    }
}

impl Display for Decimal {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        let v = self.0 / 100000;
        let d = self.0 - v * 100000;
        write!(f, "{}.{:05}", v, d)
    }
}

impl Div<Decimal> for Decimal {
    type Output = Decimal;

    #[inline]
    fn div(self, other: Decimal) -> Decimal {
        Decimal(self.0 * 100000 / other.0)
    }
}

impl DivAssign<Decimal> for Decimal {
    #[inline]
    fn div_assign(&mut self, other: Decimal) {
        self.0 = self.0 * 100000 / other.0
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
        Decimal((i * 100000.0) as i64)
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
                return Err(Error::Parse(format!("{} cannot be parsed as decimal.", s)));
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
            Ok(Decimal(n * 100000 + d))
        } else {
            Err(Error::Parse(format!("{} cannot be parsed as decimal.", s)))
        }
    }
}

impl Mul<Decimal> for Decimal {
    type Output = Decimal;

    #[inline]
    fn mul(self, other: Decimal) -> Decimal {
        Decimal(self.0 * other.0 / 100000)
    }
}

impl MulAssign<Decimal> for Decimal {
    #[inline]
    fn mul_assign(&mut self, other: Decimal) {
        self.0 = (self.0 * other.0) / 100000;
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
impl serde::Serialize for Decimal {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // lossy precision here
        serializer.serialize_f64((*self).into())
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

    #[cfg(feature = "serde")]
    #[test]
    fn deserialize() {
        let expected: Decimal = 2.54.into();
        let actual: Decimal = serde_json::from_str("2.54").unwrap();
        assert_eq!(expected, actual);
    }

    #[test]
    fn div() {
        let x: Decimal = 1.32.into();
        let y: Decimal = 2.54.into();

        let expected = Decimal::from_str("0.51968").unwrap();
        assert_eq!(expected, x / y);
    }

    #[test]
    fn div_assign() {
        let mut x: Decimal = 1.32.into();
        let y: Decimal = 2.54.into();
        x /= y;

        let expected = Decimal::from_str("0.51968").unwrap();
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
    }

    #[test]
    fn mul_assign() {
        let mut x: Decimal = 1.32.into();
        let y: Decimal = 2.54.into();
        x *= y;

        let expected: Decimal = 3.3528.into();
        assert_eq!(expected, x);
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
        let actual = (0..5).into_iter().map(|_| -> Decimal { 1.into() }).sum();
        let expected: Decimal = 5.into();
        assert_eq!(expected, actual);
    }
}
