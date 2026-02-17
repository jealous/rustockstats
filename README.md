# rustockstats

A Rust crate that provides **stockstats-like** technical indicators on top of a Polars `DataFrame`.

Indicators/derived columns are computed **lazily on-demand**: when you call `get("rsi")`, the column is created if missing and then returned.

## Installation

```toml
[dependencies]
rustockstats = { git = "https://github.com/jealous/rustockstats.git" }
# or, once published:
# rustockstats = "0.x"
```

## Usage

```rust
use polars::prelude::*;
use rustockstats::StockDataFrame;

fn main() -> anyhow::Result<()> {
    let df = df!(
        "open" => &[1.0, 2.0, 3.0],
        "high" => &[1.2, 2.2, 3.2],
        "low"  => &[0.8, 1.8, 2.8],
        "close"=> &[1.1, 2.1, 3.1],
        "volume" => &[10.0, 11.0, 12.0],
    )?;

    let mut sdf = StockDataFrame::retype(df)?;

    // Base indicator (uses default windows)
    let _rsi = sdf.get("rsi")?;

    // Parameterized indicator column name: <col>_<window>_<indicator>
    let _ema12 = sdf.get("close_12_ema")?;

    // Access the underlying DataFrame
    let out = sdf.into_df();
    println!("{}", out);

    Ok(())
}
```

## Supported indicators (high level)

This crate supports a broad subset of Python `stockstats`, including:

- Moving averages: `sma`, `ema`, `smma`, `tema`, `vwma`, `linear_wma`
- Momentum/oscillators: `rsi`, `stochrsi`, `wr`, `roc`, `cmo`, `kst`
- Volatility/trend: `atr`, `cci`, `chop`, `supertrend`
- MACD family: `macd`, plus related signal/hist columns
- Volume-based: `mfi`, `vr`, `pvo`, `ppo`
- Others: `boll` (and `boll_ub`/`boll_lb`), `ao`, `aroon`, `ichimoku`, `rvgi`, `qqe`, `cr`, `bop`, `cti`, `pgo`, `psl`, `ftr`, `kama`, `ker`

Many indicators support parameterization by encoding windows into the column name (e.g. `close_12_ema`, `rsi_6`).

## Comparison with Python `stockstats`

- **API style**: similar “computed column” model, but backed by Rust + Polars.
- **Performance**: intended to avoid repeated scans by caching computed columns in the DataFrame.
- **Type system**: errors are explicit via `Result` and `StockStatsError`.

## License

BSD (see [LICENSE](LICENSE)).
