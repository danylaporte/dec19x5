use crate::Decimal;
use std::sync::atomic::{AtomicI64, Ordering};

#[derive(Debug, Default)]
pub struct AtomicDecimal(AtomicI64);

impl AtomicDecimal {
    pub fn new(d: Decimal) -> Self {
        Self(AtomicI64::new(d.0))
    }

    pub fn compare_exchange(
        &self,
        current: Decimal,
        new: Decimal,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Decimal, Decimal> {
        self.0
            .compare_exchange(current.0, new.0, success, failure)
            .map(Decimal)
            .map_err(Decimal)
    }

    pub fn compare_exchange_weak(
        &self,
        current: Decimal,
        new: Decimal,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Decimal, Decimal> {
        self.0
            .compare_exchange_weak(current.0, new.0, success, failure)
            .map(Decimal)
            .map_err(Decimal)
    }

    pub fn fetch_add(&self, val: Decimal, order: Ordering) -> Decimal {
        Decimal(self.0.fetch_add(val.0, order))
    }

    pub fn fetch_sub(&self, val: Decimal, order: Ordering) -> Decimal {
        Decimal(self.0.fetch_sub(val.0, order))
    }

    pub fn fetch_update<F>(
        &self,
        set_order: Ordering,
        fetch_order: Ordering,
        mut f: F,
    ) -> Result<Decimal, Decimal>
    where
        F: FnMut(Decimal) -> Option<Decimal>,
    {
        self.0
            .fetch_update(set_order, fetch_order, |v| f(Decimal(v)).map(|v| v.0))
            .map(Decimal)
            .map_err(Decimal)
    }

    pub fn into_inner(self) -> Decimal {
        Decimal(self.0.into_inner())
    }

    pub fn load(&self, order: Ordering) -> Decimal {
        Decimal(self.0.load(order))
    }

    pub fn store(&self, val: Decimal, order: Ordering) {
        self.0.store(val.0, order)
    }

    pub fn swap(&self, val: Decimal, order: Ordering) -> Decimal {
        Decimal(self.0.swap(val.0, order))
    }
}
