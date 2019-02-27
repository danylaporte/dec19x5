[![Build Status](https://travis-ci.org/danylaporte/dec19x5.svg?branch=master)](https://travis-ci.org/danylaporte/dec19x5)

Represent a 64-bit decimal type taking 14 digits before the dot and 5 digits after the dot.

The calculation uses integer math to ensure accurate financial amounts.

## Documentation
[API Documentation](https://danylaporte.github.io/dec19x5/dec19x5)

## Example

```rust
use dec19x5::Decimal;
use std::str::FromStr;

fn main() {
    // parse from string
    let d = Decimal::from_str("13.45331").unwrap();

    // write to string
    assert_eq!("13.45331", &format!("{}", d));

    // load an i32
    let d = Decimal::from(10i32);

    // multiple ops are supported.
    assert_eq!(d + Decimal::from(3i32), Decimal::from(13i32));
}
```

## License

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0
[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0) or the MIT license
[http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT), at your
option. This file may not be copied, modified, or distributed
except according to those terms.