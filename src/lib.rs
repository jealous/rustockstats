use polars::prelude::*;
use regex::Regex;
use std::collections::VecDeque;

mod indicator;
use indicator::Indicator;
use std::fmt::{Display, Formatter};

const KDJ_PARAM_0: f64 = 2.0 / 3.0;
const KDJ_PARAM_1: f64 = 1.0 / 3.0;
const BOLL_STD_TIMES: f64 = 2.0;
const DX_SMMA: usize = 14;
const ADX_EMA: usize = 6;
const ADXR_EMA: usize = 6;
const SUPERTREND_MUL: f64 = 3.0;

/// Errors returned by this crate.
///
/// Most errors wrap a Polars failure or report invalid indicator/column inputs.
#[derive(Debug)]
pub enum StockStatsError {
    Polars(PolarsError),
    InvalidColumn(String),
    InvalidWindow(String),
    MissingBaseColumn(String),
}

impl Display for StockStatsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Polars(e) => write!(f, "polars error: {e}"),
            Self::InvalidColumn(c) => write!(f, "invalid column name: {c}"),
            Self::InvalidWindow(w) => write!(f, "invalid window: {w}"),
            Self::MissingBaseColumn(c) => write!(f, "missing base column: {c}"),
        }
    }
}

impl std::error::Error for StockStatsError {}

impl From<PolarsError> for StockStatsError {
    fn from(value: PolarsError) -> Self {
        Self::Polars(value)
    }
}

type Result<T> = std::result::Result<T, StockStatsError>;

#[derive(Clone, Debug)]
pub(crate) struct Meta {
    name: String,
    column: Option<String>,
    windows: Option<String>,
}

impl Meta {
    pub(crate) fn new(
        name: impl Into<String>,
        column: Option<String>,
        windows: Option<String>,
    ) -> Self {
        Self {
            name: name.into(),
            column,
            windows,
        }
    }

    pub(crate) fn full_name(&self) -> String {
        match (&self.column, &self.windows) {
            (None, None) => self.name.clone(),
            (None, Some(w)) => format!("{}_{}", self.name, w),
            (Some(c), Some(w)) => format!("{}_{}_{}", c, w, self.name),
            (Some(c), None) => format!("{}_{}", c, self.name),
        }
    }

    pub(crate) fn name_ex(&self, ex: &str) -> String {
        let base = format!("{}{}", self.name, ex);
        match &self.windows {
            Some(w) => format!("{}_{}", base, w),
            None => base,
        }
    }

    pub(crate) fn with_name(&self, name: &str) -> Self {
        Self {
            name: name.to_owned(),
            column: self.column.clone(),
            windows: self.windows.clone(),
        }
    }
}

/// A stockstats-like wrapper around a Polars [`DataFrame`].
///
/// Columns are computed lazily on-demand via [`StockDataFrame::get`].
pub struct StockDataFrame {
    df: DataFrame,
}

/// Helper enum exposed for tests to validate column-name parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedColumnName {
    Three(String, String, String),
    Two(String, String),
    NoMatch,
}

impl StockDataFrame {
    /// Normalize common OHLCV column names to lowercase and return a [`StockDataFrame`].
    ///
    /// Recognized names (case-insensitive): Open, High, Low, Close, Volume, Amount.
    pub fn retype(mut df: DataFrame) -> Result<Self> {
        let names: Vec<String> = df
            .get_column_names_str()
            .into_iter()
            .map(std::borrow::ToOwned::to_owned)
            .collect();
        for old in names {
            let low = match old.as_str() {
                "Open" | "open" => Some("open"),
                "High" | "high" => Some("high"),
                "Low" | "low" => Some("low"),
                "Close" | "close" => Some("close"),
                "Volume" | "volume" => Some("volume"),
                "Amount" | "amount" => Some("amount"),
                _ => None,
            };
            if let Some(new_name) = low
                && old != new_name
            {
                df.rename(&old, new_name.into())?;
            }
        }

        Ok(Self { df })
    }

    /// Borrow the underlying Polars [`DataFrame`].
    pub fn df(&self) -> &DataFrame {
        &self.df
    }

    /// Consume this wrapper and return the underlying Polars [`DataFrame`].
    pub fn into_df(self) -> DataFrame {
        self.df
    }

    /// Drop a computed or input column in-place.
    pub fn drop_column(&mut self, name: &str) -> Result<()> {
        self.df.drop_in_place(name)?;
        Ok(())
    }

    pub fn parse_column_name_for_test(name: &str) -> ParsedColumnName {
        match parse_column_name(name) {
            Parsed::Three(c, w, n) => ParsedColumnName::Three(c, w, n),
            Parsed::Two(c, w) => ParsedColumnName::Two(c, w),
            Parsed::NoMatch => ParsedColumnName::NoMatch,
        }
    }

    pub fn is_cross_columns_for_test(name: &str) -> bool {
        is_cross_columns(name)
    }

    pub fn parse_cross_column_for_test(name: &str) -> Option<(String, String, String)> {
        parse_cross_column(name)
    }

    pub fn to_ints_for_test(window_text: &str) -> Result<Vec<i32>> {
        let mut out = Vec::new();
        for seg in window_text.split(',') {
            let seg = seg.trim();
            if seg.is_empty() {
                continue;
            }
            if let Some((a, b)) = seg.split_once('~') {
                let start = parse_int(a.trim())?;
                let end = parse_int(b.trim())?;
                if start <= end {
                    for i in start..=end {
                        out.push(i);
                    }
                } else {
                    for i in end..=start {
                        out.push(i);
                    }
                }
            } else {
                out.push(parse_int(seg)?);
            }
        }
        out.sort_unstable();
        out.dedup();
        Ok(out)
    }

    pub fn get_int_positive_for_test(text: &str) -> Result<i32> {
        let n = parse_int(text)?;
        if n <= 0 {
            return Err(StockStatsError::InvalidWindow(text.to_owned()));
        }
        Ok(n)
    }

    pub fn roc_for_test(x: &[f64], size: i32) -> Vec<f64> {
        roc(x, size)
    }

    pub fn mad_for_test(x: &[f64], window: usize) -> Vec<f64> {
        mad(x, window)
    }

    pub fn linear_wma_for_test(x: &[f64], window: usize) -> Vec<f64> {
        linear_wma(x, window)
    }

    pub fn linear_reg_for_test(x: &[f64], window: usize, correlation: bool) -> Vec<f64> {
        linear_reg(x, window, correlation)
    }

    pub fn sym_wma4_for_test(x: &[f64]) -> Vec<f64> {
        sym_wma4(x)
    }

    pub fn s_shift_for_test(x: &[f64], shift: i32) -> Vec<f64> {
        shift_arr(x, shift)
    }

    pub fn get(&mut self, key: &str) -> Result<&Column> {
        if !self.has_column(key) {
            self.init_column(key)?;
        }
        self.df
            .column(key)
            .map_err(|_| StockStatsError::InvalidColumn(key.to_owned()))
    }

    fn has_column(&self, name: &str) -> bool {
        self.df.get_column_index(name).is_some()
    }

    fn init_column(&mut self, key: &str) -> Result<()> {
        if self.df.height() == 0 {
            self.add_column_f64(key, vec![])?;
            return Ok(());
        }

        if let Some(res) = self.init_alias_column(key) {
            return res;
        }
        if let Some(res) = self.init_compound_indicator_column(key) {
            return res;
        }

        self.init_generic_column(key)
    }

    fn init_alias_column(&mut self, key: &str) -> Option<Result<()>> {
        match key {
            "rate" => Some(self.get_rate()),
            "middle" | "tp" | "typical_price" => Some(self.get_tp(Meta::new(
                if key == "middle" { "middle" } else { "tp" },
                None,
                None,
            ))),
            "boll" | "boll_ub" | "boll_lb" => Some(self.get_boll(Meta::new("boll", None, None))),
            _ if key.starts_with("boll_ub_") || key.starts_with("boll_lb_") => key
                .rsplit_once("_")
                .map(|(_, w)| self.get_boll(Meta::new("boll", None, Some(w.to_owned())))),
            _ => None,
        }
    }

    fn init_compound_indicator_column(&mut self, key: &str) -> Option<Result<()>> {
        match key {
            "macd" | "macds" | "macdh" => Some(self.get_macd(Meta::new("macd", None, None))),
            "pvo" | "pvos" | "pvoh" => Some(self.get_pvo(Meta::new("pvo", None, None))),
            "ppo" | "ppos" | "ppoh" => Some(self.get_ppo(Meta::new("ppo", None, None))),
            "qqe" | "qqel" | "qqes" => Some(self.get_qqe(Meta::new("qqe", None, None))),
            "cr" | "cr-ma1" | "cr-ma2" | "cr-ma3" => Some(self.get_cr(Meta::new("cr", None, None))),
            "tr" => Some(self.get_tr(Meta::new("tr", None, None))),
            "dx" | "adx" | "adxr" => Some(self.get_dmi_defaults()),
            "log-ret" => Some(self.get_log_ret()),
            "wt1" | "wt2" => Some(self.get_wt(Meta::new("wt", None, None))),
            "supertrend" | "supertrend_ub" | "supertrend_lb" => {
                Some(self.get_supertrend(Meta::new("supertrend", None, None)))
            }
            "eribull" | "eribear" => Some(self.get_eri(Meta::new("eri", None, None))),
            "rvgi" | "rvgis" => Some(self.get_rvgi(Meta::new("rvgi", None, None))),
            "kst" => Some(self.get_kst(Meta::new("kst", None, None))),
            "bop" => Some(self.get_bop(Meta::new("bop", None, None))),
            _ => None,
        }
    }

    fn init_generic_column(&mut self, key: &str) -> Result<()> {
        if key.ends_with("_delta") {
            return self.get_delta_suffix(key);
        }
        if is_cross_columns(key) {
            return self.get_cross(key);
        }
        if is_compare_columns(key) {
            return self.get_compare(key);
        }

        if Indicator::from_name(key).is_some() {
            return self.dispatch(Meta::new(key, None, None));
        }

        match parse_column_name(key) {
            Parsed::Three(col, windows, name)
                if col == "boll" && (name == "ub" || name == "lb") =>
            {
                self.get_boll(Meta::new("boll", None, Some(windows)))
            }
            Parsed::Three(col, windows, name) => {
                self.dispatch(Meta::new(name, Some(col), Some(windows)))
            }
            Parsed::Two(name, windows) => self.dispatch(Meta::new(name, None, Some(windows))),
            Parsed::NoMatch => Err(StockStatsError::InvalidColumn(key.to_owned())),
        }
    }

    fn dispatch(&mut self, meta: Meta) -> Result<()> {
        let ind = Indicator::from_name(meta.name.as_str())
            .ok_or_else(|| StockStatsError::InvalidColumn(meta.full_name()))?;
        ind.compute(self, meta)
    }

    fn get_default_windows(&self, name: &str) -> Result<&'static str> {
        Indicator::from_name(name)
            .and_then(|i| i.default_windows())
            .ok_or_else(|| StockStatsError::InvalidColumn(name.to_owned()))
    }

    fn parse_ints(&self, meta: &Meta) -> Result<Vec<i32>> {
        let window_text = match &meta.windows {
            Some(w) => w.as_str(),
            None => self.get_default_windows(&meta.name)?,
        };

        let mut out = Vec::new();
        for seg in window_text.split(',') {
            let seg = seg.trim();
            if seg.is_empty() {
                continue;
            }
            if let Some((a, b)) = seg.split_once('~') {
                let start = parse_int(a.trim())?;
                let end = parse_int(b.trim())?;
                if start <= end {
                    for i in start..=end {
                        out.push(i);
                    }
                }
            } else {
                out.push(parse_int(seg)?);
            }
        }
        Ok(out)
    }

    fn get_int(&self, meta: &Meta, idx: usize) -> Result<i32> {
        let ints = self.parse_ints(meta)?;
        if let Some(v) = ints.get(idx) {
            return Ok(*v);
        }

        let dft = self
            .get_default_windows(&meta.name)?
            .split(',')
            .map(parse_int)
            .collect::<Result<Vec<_>>>()?;

        dft.get(idx)
            .copied()
            .ok_or_else(|| StockStatsError::InvalidWindow(meta.name.clone()))
    }

    fn get_column_name_for_meta(&self, meta: &Meta) -> Result<String> {
        meta.column
            .clone()
            .or_else(|| {
                Indicator::from_name(meta.name.as_str())
                    .and_then(|i| i.default_column())
                    .map(ToOwned::to_owned)
            })
            .ok_or_else(|| StockStatsError::MissingBaseColumn(meta.full_name()))
    }

    fn ensure_column_vec(&mut self, name: &str) -> Result<Vec<f64>> {
        if !self.has_column(name) {
            self.get(name)?;
        }
        self.column_to_f64(name)
    }

    fn column_to_f64(&self, name: &str) -> Result<Vec<f64>> {
        let col = self.df.column(name)?;
        let s = col.as_materialized_series();
        let casted = if s.dtype() == &DataType::Float64 {
            s.clone()
        } else {
            s.cast(&DataType::Float64)?
        };
        Ok(casted
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(0.0))
            .collect())
    }

    fn add_column_f64(&mut self, name: &str, vals: Vec<f64>) -> Result<()> {
        let s = Series::new(name.into(), vals);
        self.df.with_column(s)?;
        Ok(())
    }

    fn add_column_bool(&mut self, name: &str, vals: Vec<bool>) -> Result<()> {
        let s = Series::new(name.into(), vals);
        self.df.with_column(s)?;
        Ok(())
    }

    fn get_rate(&mut self) -> Result<()> {
        let close = self.ensure_column_vec("close")?;
        let mut out = vec![f64::NAN; close.len()];
        for i in 1..close.len() {
            out[i] = if close[i - 1] != 0.0 {
                (close[i] / close[i - 1] - 1.0) * 100.0
            } else {
                0.0
            };
        }
        self.add_column_f64("rate", out)
    }

    fn get_change(&mut self, meta: Meta) -> Result<()> {
        let close = self.ensure_column_vec("close")?;
        let window = self.get_int(&meta, 0)?;
        self.add_column_f64(&meta.full_name(), roc(&close, window))
    }

    fn get_r(&mut self, meta: Meta) -> Result<()> {
        let col_name = self.get_column_name_for_meta(&meta)?;
        let src = self.ensure_column_vec(&col_name)?;
        let shift = -self.get_int(&meta, 0)?;
        self.add_column_f64(&meta.full_name(), roc(&src, shift))
    }

    fn get_s(&mut self, meta: Meta) -> Result<()> {
        let col_name = self.get_column_name_for_meta(&meta)?;
        let src = self.ensure_column_vec(&col_name)?;
        let shift = self.get_int(&meta, 0)?;
        self.add_column_f64(&meta.full_name(), shift_arr(&src, shift))
    }

    fn get_d(&mut self, meta: Meta) -> Result<()> {
        let col_name = self.get_column_name_for_meta(&meta)?;
        let src = self.ensure_column_vec(&col_name)?;
        let window = self.get_int(&meta, 0)?;
        self.add_column_f64(&meta.full_name(), delta(&src, window))
    }

    fn get_delta_suffix(&mut self, key: &str) -> Result<()> {
        let src_col = key.trim_end_matches("_delta");
        let src = self.ensure_column_vec(src_col)?;
        self.add_column_f64(key, col_diff(&src))
    }

    fn get_c(&mut self, meta: Meta) -> Result<()> {
        let col_name = self.get_column_name_for_meta(&meta)?;
        let window = self.get_int(&meta, 0)? as usize;
        let s = self.ensure_column_vec(&col_name)?;
        let arr: Vec<f64> = s
            .iter()
            .map(|v| if *v != 0.0 { 1.0 } else { 0.0 })
            .collect();
        self.add_column_f64(&meta.full_name(), rolling_sum_min1(&arr, window))
    }

    fn get_fc(&mut self, meta: Meta) -> Result<()> {
        let col_name = self.get_column_name_for_meta(&meta)?;
        let window = self.get_int(&meta, 0)? as usize;
        let mut arr = self.ensure_column_vec(&col_name)?;
        for v in &mut arr {
            *v = if *v != 0.0 { 1.0 } else { 0.0 };
        }
        arr.reverse();
        let mut out = rolling_sum_min1(&arr, window);
        out.reverse();
        self.add_column_f64(&meta.full_name(), out)
    }

    fn shifted_columns(&mut self, column: &str, shifts: &[i32]) -> Result<Vec<Vec<f64>>> {
        let src = self.ensure_column_vec(column)?;
        Ok(shifts.iter().map(|s| shift_arr(&src, *s)).collect())
    }

    fn get_max(&mut self, meta: Meta) -> Result<()> {
        let col_name = self.get_column_name_for_meta(&meta)?;
        let shifts = self.parse_ints(&meta)?;
        let cols = self.shifted_columns(&col_name, &shifts)?;
        if cols.is_empty() {
            return Err(StockStatsError::InvalidWindow(meta.full_name()));
        }
        let n = cols[0].len();
        let mut out = vec![0.0; n];
        for i in 0..n {
            let mut m = f64::NEG_INFINITY;
            for c in &cols {
                m = m.max(c[i]);
            }
            out[i] = m;
        }
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_min(&mut self, meta: Meta) -> Result<()> {
        let col_name = self.get_column_name_for_meta(&meta)?;
        let shifts = self.parse_ints(&meta)?;
        let cols = self.shifted_columns(&col_name, &shifts)?;
        if cols.is_empty() {
            return Err(StockStatsError::InvalidWindow(meta.full_name()));
        }
        let n = cols[0].len();
        let mut out = vec![0.0; n];
        for i in 0..n {
            let mut m = f64::INFINITY;
            for c in &cols {
                m = m.min(c[i]);
            }
            out[i] = m;
        }
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_p(&mut self, meta: Meta) -> Result<()> {
        let col_name = self.get_column_name_for_meta(&meta)?;
        let mut shifts = self.parse_ints(&meta)?;
        shifts.reverse();
        let src = self.ensure_column_vec(&col_name)?;
        let n = src.len();
        let mut out = vec![0.0; n];
        for (count, shift) in shifts.iter().enumerate() {
            let shifted = shift_arr(&src, *shift);
            let weight = (1_i64 << count) as f64;
            for i in 0..n {
                if shifted[i] > 0.0 {
                    out[i] += weight;
                }
            }
        }
        set_nan_by_shifts(&mut out, &shifts);
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_sma(&mut self, meta: Meta) -> Result<()> {
        let col_name = self.get_column_name_for_meta(&meta)?;
        let src = self.ensure_column_vec(&col_name)?;
        let window = self.get_int(&meta, 0)? as usize;
        self.add_column_f64(&meta.full_name(), rolling_mean_min1(&src, window))
    }

    fn get_smma(&mut self, meta: Meta) -> Result<()> {
        let col_name = self.get_column_name_for_meta(&meta)?;
        let src = self.ensure_column_vec(&col_name)?;
        let window = self.get_int(&meta, 0)? as f64;
        self.add_column_f64(&meta.full_name(), ewm_adjust(&src, 1.0 / window))
    }

    fn get_ema(&mut self, meta: Meta) -> Result<()> {
        let col_name = self.get_column_name_for_meta(&meta)?;
        let src = self.ensure_column_vec(&col_name)?;
        let window = self.get_int(&meta, 0)? as f64;
        let alpha = 2.0 / (window + 1.0);
        self.add_column_f64(&meta.full_name(), ewm_adjust(&src, alpha))
    }

    fn get_macd(&mut self, meta: Meta) -> Result<()> {
        let close = self.ensure_column_vec("close")?;
        let short_w = self.get_int(&meta, 0)? as f64;
        let long_w = self.get_int(&meta, 1)? as f64;
        let signal_w = self.get_int(&meta, 2)? as f64;

        let ema_short = ewm_adjust(&close, 2.0 / (short_w + 1.0));
        let ema_long = ewm_adjust(&close, 2.0 / (long_w + 1.0));
        let macd: Vec<f64> = ema_short
            .iter()
            .zip(ema_long.iter())
            .map(|(a, b)| a - b)
            .collect();
        self.add_column_f64(&meta.full_name(), macd.clone())?;

        let macds = ewm_adjust(&macd, 2.0 / (signal_w + 1.0));
        self.add_column_f64(&meta.name_ex("s"), macds.clone())?;

        let macdh: Vec<f64> = macd.iter().zip(macds.iter()).map(|(a, b)| a - b).collect();
        self.add_column_f64(&meta.name_ex("h"), macdh)
    }

    fn get_ppo(&mut self, meta: Meta) -> Result<()> {
        let close = self.ensure_column_vec("close")?;
        self.ppo_and_pvo("ppo", &close, &meta)
    }

    fn get_pvo(&mut self, meta: Meta) -> Result<()> {
        let volume = self.ensure_column_vec("volume")?;
        self.ppo_and_pvo("pvo", &volume, &meta)
    }

    fn ppo_and_pvo(&mut self, name: &str, src: &[f64], meta: &Meta) -> Result<()> {
        let short_w = self.get_int(meta, 0)? as f64;
        let long_w = self.get_int(meta, 1)? as f64;
        let signal_w = self.get_int(meta, 2)? as f64;

        let p_short = ewm_adjust(src, 2.0 / (short_w + 1.0));
        let p_long = ewm_adjust(src, 2.0 / (long_w + 1.0));
        let p: Vec<f64> = p_short
            .iter()
            .zip(p_long.iter())
            .map(|(a, b)| if *b != 0.0 { (a - b) / b * 100.0 } else { 0.0 })
            .collect();

        let name_col = meta.with_name(name).full_name();
        let s_col = format!("{}s", name_col);
        let h_col = format!("{}h", name_col);
        self.add_column_f64(&name_col, p.clone())?;
        let ps = ewm_adjust(&p, 2.0 / (signal_w + 1.0));
        self.add_column_f64(&s_col, ps.clone())?;
        let ph: Vec<f64> = p.iter().zip(ps.iter()).map(|(a, b)| a - b).collect();
        self.add_column_f64(&h_col, ph)
    }

    fn rsv(&mut self, window: usize) -> Result<Vec<f64>> {
        let low = self.ensure_column_vec("low")?;
        let high = self.ensure_column_vec("high")?;
        let close = self.ensure_column_vec("close")?;
        let low_min = rolling_min_min1(&low, window);
        let high_max = rolling_max_min1(&high, window);
        Ok(divide_non_zero(
            &close
                .iter()
                .zip(low_min.iter())
                .map(|(c, l)| c - l)
                .collect::<Vec<_>>(),
            &high_max
                .iter()
                .zip(low_min.iter())
                .map(|(h, l)| h - l)
                .collect::<Vec<_>>(),
        )
        .into_iter()
        .map(|v| v * 100.0)
        .collect())
    }

    fn get_rsv(&mut self, meta: Meta) -> Result<()> {
        let out = self.rsv(self.get_int(&meta, 0)? as usize)?;
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_rsi(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let close = self.ensure_column_vec("close")?;
        let diff = np_diff(&close);

        let up: Vec<f64> = diff
            .iter()
            .map(|d| if *d > 0.0 { *d } else { 0.0 })
            .collect();
        let down: Vec<f64> = diff
            .iter()
            .map(|d| if *d < 0.0 { -*d } else { 0.0 })
            .collect();

        let up_smma = ewm_adjust(&up, 1.0 / window as f64);
        let down_smma = ewm_adjust(&down, 1.0 / window as f64);

        let mut out = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            let total = up_smma[i] + down_smma[i];
            if total != 0.0 {
                out.push(100.0 * (up_smma[i] / total));
            } else {
                out.push(50.0);
            }
        }
        if let Some(first) = out.first_mut() {
            *first = 50.0;
        }
        self.add_column_f64(&meta.full_name(), out)
    }

    fn rsi_vec(&mut self, window: usize) -> Result<Vec<f64>> {
        let name = format!("rsi_{}", window);
        if !self.has_column(&name) {
            self.get_rsi(Meta::new("rsi", None, Some(window.to_string())))?;
        }
        self.ensure_column_vec(&name)
    }

    fn get_stochrsi(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let rsi = self.rsi_vec(window)?;
        let rsi_min = rolling_min_min1(&rsi, window);
        let rsi_max = rolling_max_min1(&rsi, window);
        let out: Vec<f64> = (0..rsi.len())
            .map(|i| {
                let range = rsi_max[i] - rsi_min[i];
                if range != 0.0 {
                    (rsi[i] - rsi_min[i]) / range * 100.0
                } else {
                    0.0
                }
            })
            .collect();
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_wr(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let low = self.ensure_column_vec("low")?;
        let high = self.ensure_column_vec("high")?;
        let close = self.ensure_column_vec("close")?;
        let ln = rolling_min_min1(&low, window);
        let hn = rolling_max_min1(&high, window);
        let out: Vec<f64> = (0..close.len())
            .map(|i| {
                let den = hn[i] - ln[i];
                if den != 0.0 {
                    (hn[i] - close[i]) / den * -100.0
                } else {
                    0.0
                }
            })
            .collect();
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_kdjk(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let rsv = self.rsv(window)?;
        let mut k = 50.0;
        let mut out = Vec::with_capacity(rsv.len());
        for v in rsv {
            k = KDJ_PARAM_0 * k + KDJ_PARAM_1 * v;
            out.push(k);
        }
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_kdjd(&mut self, meta: Meta) -> Result<()> {
        let kdjk_name = meta.full_name().replace("kdjd", "kdjk");
        let src = self.ensure_column_vec(&kdjk_name)?;
        let mut k = 50.0;
        let mut out = Vec::with_capacity(src.len());
        for v in src {
            k = KDJ_PARAM_0 * k + KDJ_PARAM_1 * v;
            out.push(k);
        }
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_kdjj(&mut self, meta: Meta) -> Result<()> {
        let kdjk_name = meta.full_name().replace("kdjj", "kdjk");
        let kdjd_name = meta.full_name().replace("kdjj", "kdjd");
        let k = self.ensure_column_vec(&kdjk_name)?;
        let d = self.ensure_column_vec(&kdjd_name)?;
        let out: Vec<f64> = k
            .iter()
            .zip(d.iter())
            .map(|(kv, dv)| 3.0 * kv - 2.0 * dv)
            .collect();
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_boll(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let close = self.ensure_column_vec("close")?;
        let moving_avg = rolling_mean_min1(&close, window);
        let moving_std = rolling_std_sample_min1(&close, window);

        self.add_column_f64(&meta.full_name(), moving_avg.clone())?;

        let width: Vec<f64> = moving_std.iter().map(|s| BOLL_STD_TIMES * s).collect();
        let ub: Vec<f64> = moving_avg
            .iter()
            .zip(width.iter())
            .map(|(a, w)| a + w)
            .collect();
        let lb: Vec<f64> = moving_avg
            .iter()
            .zip(width.iter())
            .map(|(a, w)| a - w)
            .collect();

        self.add_column_f64(&meta.name_ex("_ub"), ub)?;
        self.add_column_f64(&meta.name_ex("_lb"), lb)
    }

    fn tr(&mut self) -> Result<Vec<f64>> {
        let close = self.ensure_column_vec("close")?;
        let high = self.ensure_column_vec("high")?;
        let low = self.ensure_column_vec("low")?;
        let prev_close = shift_arr(&close, -1);

        let mut out = vec![0.0; close.len()];
        for i in 0..close.len() {
            let c1 = high[i] - low[i];
            let c2 = (high[i] - prev_close[i]).abs();
            let c3 = (low[i] - prev_close[i]).abs();
            out[i] = c1.max(c2).max(c3);
        }
        Ok(out)
    }

    fn get_tr(&mut self, meta: Meta) -> Result<()> {
        let out = self.tr()?;
        self.add_column_f64(&meta.full_name(), out)
    }

    fn atr(&mut self, window: usize) -> Result<Vec<f64>> {
        Ok(ewm_adjust(&self.tr()?, 1.0 / window as f64))
    }

    fn get_atr(&mut self, meta: Meta) -> Result<()> {
        let out = self.atr(self.get_int(&meta, 0)? as usize)?;
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_dma(&mut self, meta: Meta) -> Result<()> {
        let fast = self.get_int(&meta, 0)? as usize;
        let slow = self.get_int(&meta, 1)? as usize;
        let col_name = self.get_column_name_for_meta(&meta)?;
        let col = self.ensure_column_vec(&col_name)?;
        let fast_ma = rolling_mean_min1(&col, fast);
        let slow_ma = rolling_mean_min1(&col, slow);
        let out: Vec<f64> = fast_ma
            .iter()
            .zip(slow_ma.iter())
            .map(|(a, b)| (a - b) + 1e-12)
            .collect();
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_pdm_ndm(&mut self, window: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        let hd = np_diff(&self.ensure_column_vec("high")?);
        let ld = np_diff_neg(&self.ensure_column_vec("low")?);

        let mut p = vec![0.0; hd.len()];
        let mut n = vec![0.0; hd.len()];
        for i in 0..hd.len() {
            p[i] = if hd[i] > 0.0 && hd[i] > ld[i] {
                hd[i]
            } else {
                0.0
            };
            n[i] = if ld[i] > 0.0 && ld[i] > hd[i] {
                ld[i]
            } else {
                0.0
            };
        }

        if window > 1 {
            Ok((
                ewm_adjust(&p, 1.0 / window as f64),
                ewm_adjust(&n, 1.0 / window as f64),
            ))
        } else {
            Ok((p, n))
        }
    }

    fn get_pdm(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let (p, _) = self.get_pdm_ndm(window)?;
        self.add_column_f64(&meta.full_name(), p)
    }

    fn get_ndm(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let (_, n) = self.get_pdm_ndm(window)?;
        self.add_column_f64(&meta.full_name(), n)
    }

    fn get_pdi_ndi(&mut self, window: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        let (pdm, ndm) = self.get_pdm_ndm(window)?;
        let atr = self.atr(window)?;
        let pdi: Vec<f64> = pdm
            .iter()
            .zip(atr.iter())
            .map(|(a, b)| if *b != 0.0 { a / b * 100.0 } else { 0.0 })
            .collect();
        let ndi: Vec<f64> = ndm
            .iter()
            .zip(atr.iter())
            .map(|(a, b)| if *b != 0.0 { a / b * 100.0 } else { 0.0 })
            .collect();
        Ok((pdi, ndi))
    }

    fn dx_vals(&mut self, window: usize) -> Result<Vec<f64>> {
        let (pdi, ndi) = self.get_pdi_ndi(window)?;
        Ok((0..pdi.len())
            .map(|i| {
                let den = pdi[i] + ndi[i];
                if den != 0.0 {
                    ((pdi[i] - ndi[i]).abs() / den) * 100.0
                } else {
                    0.0
                }
            })
            .collect())
    }

    fn get_pdi(&mut self, meta: Meta) -> Result<()> {
        let (pdi, _) = self.get_pdi_ndi(self.get_int(&meta, 0)? as usize)?;
        self.add_column_f64(&meta.full_name(), pdi)
    }

    fn get_ndi(&mut self, meta: Meta) -> Result<()> {
        let (_, ndi) = self.get_pdi_ndi(self.get_int(&meta, 0)? as usize)?;
        self.add_column_f64(&meta.full_name(), ndi)
    }

    fn get_dx(&mut self, meta: Meta) -> Result<()> {
        let out = self.dx_vals(self.get_int(&meta, 0)? as usize)?;
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_adx(&mut self, meta: Meta) -> Result<()> {
        let w = self.get_int(&meta, 0)? as usize;
        let dx = self.dx_vals(w)?;
        self.add_column_f64(
            &meta.full_name(),
            ewm_adjust(&dx, 2.0 / (ADX_EMA as f64 + 1.0)),
        )
    }

    fn get_adxr(&mut self, meta: Meta) -> Result<()> {
        let w = self.get_int(&meta, 0)? as usize;
        let dx = self.dx_vals(w)?;
        let adx = ewm_adjust(&dx, 2.0 / (ADX_EMA as f64 + 1.0));
        self.add_column_f64(
            &meta.full_name(),
            ewm_adjust(&adx, 2.0 / (ADXR_EMA as f64 + 1.0)),
        )
    }

    fn get_dmi_defaults(&mut self) -> Result<()> {
        let dx = self.dx_vals(DX_SMMA)?;
        self.add_column_f64("dx", dx.clone())?;
        let adx = ewm_adjust(&dx, 2.0 / (ADX_EMA as f64 + 1.0));
        self.add_column_f64("adx", adx.clone())?;
        self.add_column_f64("adxr", ewm_adjust(&adx, 2.0 / (ADXR_EMA as f64 + 1.0)))
    }

    fn tp(&mut self) -> Result<Vec<f64>> {
        if self.has_column("amount") {
            let amount = self.ensure_column_vec("amount")?;
            let volume = self.ensure_column_vec("volume")?;
            Ok(amount
                .iter()
                .zip(volume.iter())
                .map(|(a, v)| if *v != 0.0 { a / v } else { 0.0 })
                .collect())
        } else {
            let close = self.ensure_column_vec("close")?;
            let high = self.ensure_column_vec("high")?;
            let low = self.ensure_column_vec("low")?;
            Ok(close
                .iter()
                .zip(high.iter())
                .zip(low.iter())
                .map(|((c, h), l)| (c + h + l) / 3.0)
                .collect())
        }
    }

    fn get_tp(&mut self, meta: Meta) -> Result<()> {
        let out = self.tp()?;
        if meta.name == "middle" {
            self.add_column_f64("middle", out)
        } else if meta.name == "typical_price" {
            self.add_column_f64("typical_price", out)
        } else {
            self.add_column_f64(&meta.full_name(), out)
        }
    }

    fn get_cci(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let tp = self.tp()?;
        let tp_sma = rolling_mean_min1(&tp, window);
        let mad = mad(&tp, window);
        let divisor: Vec<f64> = mad.iter().map(|v| 0.015 * v).collect();

        let out: Vec<f64> = tp
            .iter()
            .zip(tp_sma.iter())
            .zip(divisor.iter())
            .map(|((t, m), d)| if *d != 0.0 { (t - m) / d } else { 0.0 })
            .collect();
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_vr(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let change = self.ensure_column_vec("change")?;
        let volume = self.ensure_column_vec("volume")?;

        let gt_zero: Vec<f64> = change
            .iter()
            .zip(volume.iter())
            .map(|(c, v)| if *c > 0.0 { *v } else { 0.0 })
            .collect();
        let lt_zero: Vec<f64> = change
            .iter()
            .zip(volume.iter())
            .map(|(c, v)| if *c < 0.0 { *v } else { 0.0 })
            .collect();
        let eq_zero: Vec<f64> = change
            .iter()
            .zip(volume.iter())
            .map(|(c, v)| if *c == 0.0 { *v } else { 0.0 })
            .collect();

        let avs = rolling_sum_min1(&gt_zero, window);
        let bvs = rolling_sum_min1(&lt_zero, window);
        let cvs = rolling_sum_min1(&eq_zero, window);

        let out: Vec<f64> = (0..change.len())
            .map(|i| {
                let half = cvs[i] * 0.5;
                let divisor = bvs[i] + half;
                if divisor != 0.0 {
                    (avs[i] + half) / divisor * 100.0
                } else {
                    0.0
                }
            })
            .collect();
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_cr(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let middle = self.tp()?;
        let ym = shift_arr(&middle, -1);
        let high = self.ensure_column_vec("high")?;
        let low = self.ensure_column_vec("low")?;

        let p1_m: Vec<f64> = ym.iter().zip(high.iter()).map(|(a, b)| a.min(*b)).collect();
        let p2_m: Vec<f64> = ym.iter().zip(low.iter()).map(|(a, b)| a.min(*b)).collect();
        let p1: Vec<f64> = high.iter().zip(p1_m.iter()).map(|(a, b)| a - b).collect();
        let p2: Vec<f64> = ym.iter().zip(p2_m.iter()).map(|(a, b)| a - b).collect();
        let p1 = rolling_sum_min1(&p1, window);
        let p2 = rolling_sum_min1(&p2, window);

        let cr: Vec<f64> = p1
            .iter()
            .zip(p2.iter())
            .map(|(a, b)| if *b != 0.0 { a / b * 100.0 } else { 0.0 })
            .collect();
        let name = meta.full_name();
        self.add_column_f64(&name, cr.clone())?;

        let ma1 = self.shifted_cr_sma(&cr, 5);
        let ma2 = self.shifted_cr_sma(&cr, 10);
        let ma3 = self.shifted_cr_sma(&cr, 20);
        self.add_column_f64(&format!("{}-ma1", name), ma1)?;
        self.add_column_f64(&format!("{}-ma2", name), ma2)?;
        self.add_column_f64(&format!("{}-ma3", name), ma3)
    }

    fn shifted_cr_sma(&self, cr: &[f64], window: usize) -> Vec<f64> {
        let cr_sma = rolling_mean_min1(cr, window);
        shift_arr(&cr_sma, -((window as f64 / 2.5 + 1.0) as i32))
    }

    fn get_mad(&mut self, meta: Meta) -> Result<()> {
        let c = self.get_column_name_for_meta(&meta)?;
        let x = self.ensure_column_vec(&c)?;
        self.add_column_f64(&meta.full_name(), mad(&x, self.get_int(&meta, 0)? as usize))
    }

    fn get_mstd(&mut self, meta: Meta) -> Result<()> {
        let c = self.get_column_name_for_meta(&meta)?;
        let x = self.ensure_column_vec(&c)?;
        self.add_column_f64(
            &meta.full_name(),
            rolling_std_sample_min1(&x, self.get_int(&meta, 0)? as usize),
        )
    }

    fn get_mvar(&mut self, meta: Meta) -> Result<()> {
        let c = self.get_column_name_for_meta(&meta)?;
        let x = self.ensure_column_vec(&c)?;
        self.add_column_f64(
            &meta.full_name(),
            rolling_var_sample_min1(&x, self.get_int(&meta, 0)? as usize),
        )
    }

    fn get_roc(&mut self, meta: Meta) -> Result<()> {
        let c = self.get_column_name_for_meta(&meta)?;
        let x = self.ensure_column_vec(&c)?;
        self.add_column_f64(&meta.full_name(), roc(&x, self.get_int(&meta, 0)?))
    }

    fn get_lrma(&mut self, meta: Meta) -> Result<()> {
        let c = self.get_column_name_for_meta(&meta)?;
        let x = self.ensure_column_vec(&c)?;
        self.add_column_f64(
            &meta.full_name(),
            linear_reg(&x, self.get_int(&meta, 0)? as usize, false),
        )
    }

    fn get_linear_wma(&mut self, meta: Meta) -> Result<()> {
        let c = self.get_column_name_for_meta(&meta)?;
        let x = self.ensure_column_vec(&c)?;
        self.add_column_f64(
            &meta.full_name(),
            linear_wma(&x, self.get_int(&meta, 0)? as usize),
        )
    }

    fn get_cti(&mut self, meta: Meta) -> Result<()> {
        let c = self.get_column_name_for_meta(&meta)?;
        let x = self.ensure_column_vec(&c)?;
        self.add_column_f64(
            &meta.full_name(),
            linear_reg(&x, self.get_int(&meta, 0)? as usize, true),
        )
    }

    fn get_trix(&mut self, meta: Meta) -> Result<()> {
        let c = self.get_column_name_for_meta(&meta)?;
        let x = self.ensure_column_vec(&c)?;
        let w = self.get_int(&meta, 0)? as f64;
        let a = 2.0 / (w + 1.0);
        let s = ewm_adjust(&x, a);
        let d = ewm_adjust(&s, a);
        let t = ewm_adjust(&d, a);
        let mut out = vec![0.0; t.len()];
        for i in 1..t.len() {
            if t[i - 1] != 0.0 {
                out[i] = (t[i] / t[i - 1] - 1.0) * 100.0;
            }
        }
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_tema(&mut self, meta: Meta) -> Result<()> {
        let c = self.get_column_name_for_meta(&meta)?;
        let x = self.ensure_column_vec(&c)?;
        let w = self.get_int(&meta, 0)? as f64;
        let a = 2.0 / (w + 1.0);
        let s = ewm_adjust(&x, a);
        let d = ewm_adjust(&s, a);
        let t = ewm_adjust(&d, a);
        let out: Vec<f64> = (0..x.len())
            .map(|i| 3.0 * s[i] - 3.0 * d[i] + t[i])
            .collect();
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_mfi(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let tp = self.tp()?;
        let volume = self.ensure_column_vec("volume")?;
        let raw: Vec<f64> = tp.iter().zip(volume.iter()).map(|(a, b)| a * b).collect();

        let mut tp_diff = vec![0.0; tp.len()];
        for i in 1..tp.len() {
            tp_diff[i] = tp[i] - tp[i - 1];
        }
        let pos: Vec<f64> = tp_diff
            .iter()
            .zip(raw.iter())
            .map(|(d, r)| if *d > 0.0 { *r } else { 0.0 })
            .collect();
        let neg: Vec<f64> = tp_diff
            .iter()
            .zip(raw.iter())
            .map(|(d, r)| if *d < 0.0 { *r } else { 0.0 })
            .collect();
        let pos_sum = rolling_sum_min1(&pos, window);
        let neg_sum = rolling_sum_min1(&neg, window);

        let mut out: Vec<f64> = pos_sum
            .iter()
            .zip(neg_sum.iter())
            .map(|(a, b)| {
                let t = a + b;
                if t > 0.0 { a / t } else { 0.5 }
            })
            .collect();
        let lim = usize::min(window, out.len());
        out[..lim].fill(0.5);
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_ker(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let c = self.get_column_name_for_meta(&meta)?;
        let v = self.ensure_column_vec(&c)?;
        self.add_column_f64(&meta.full_name(), ker(&v, window))
    }

    fn get_kama(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let fast = self.get_int(&meta, 1)? as f64;
        let slow = self.get_int(&meta, 2)? as f64;
        let c = self.get_column_name_for_meta(&meta)?;
        let col = self.ensure_column_vec(&c)?;

        let er = ker(&col, window);
        let fast_s = 2.0 / (fast + 1.0);
        let slow_s = 2.0 / (slow + 1.0);
        let smoothing: Vec<f64> = er
            .iter()
            .map(|e| 2.0 * (e * (fast_s - slow_s) + slow_s))
            .collect();

        let mut kama = rolling_mean_min1(&col, window);
        let mut last = if col.len() >= window {
            kama[window - 1]
        } else {
            0.0
        };
        for i in window..col.len() {
            let cur = smoothing[i] * (col[i] - last) + last;
            kama[i] = cur;
            last = cur;
        }
        self.add_column_f64(&meta.full_name(), kama)
    }

    fn get_vwma(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let volume = self.ensure_column_vec("volume")?;
        let tp = self.tp()?;
        let tpv: Vec<f64> = volume.iter().zip(tp.iter()).map(|(v, t)| v * t).collect();
        let r_tpv = rolling_sum_min1(&tpv, window);
        let r_vol = rolling_sum_min1(&volume, window);
        let out: Vec<f64> = r_tpv
            .iter()
            .zip(r_vol.iter())
            .map(|(a, b)| if *b != 0.0 { a / b } else { 0.0 })
            .collect();
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_chop(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let atr1 = self.atr(1)?;
        let atr_sum = rolling_sum_min1(&atr1, window);
        let high = rolling_max_min1(&self.ensure_column_vec("high")?, window);
        let low = rolling_min_min1(&self.ensure_column_vec("low")?, window);
        let den = (window as f64).log10();
        let out: Vec<f64> = (0..atr_sum.len())
            .map(|i| {
                let hl = high[i] - low[i];
                if hl > 0.0 {
                    let x = atr_sum[i] / hl;
                    if x > 0.0 {
                        x.log10() * 100.0 / den
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            })
            .collect();
        self.add_column_f64(&meta.full_name(), out)
    }

    fn wt1(&mut self, n1: usize, n2: usize) -> Result<Vec<f64>> {
        let tp = self.tp()?;
        let esa = ewm_adjust(&tp, 2.0 / (n1 as f64 + 1.0));
        let abs: Vec<f64> = tp
            .iter()
            .zip(esa.iter())
            .map(|(a, b)| (a - b).abs())
            .collect();
        let d = ewm_adjust(&abs, 2.0 / (n1 as f64 + 1.0));
        let ci: Vec<f64> = (0..tp.len())
            .map(|i| {
                let den = 0.015 * d[i];
                if den != 0.0 {
                    (tp[i] - esa[i]) / den
                } else {
                    0.0
                }
            })
            .collect();
        let mut out = ewm_adjust(&ci, 2.0 / (n2 as f64 + 1.0));
        if !out.is_empty() {
            out[0] = 0.0;
        }
        Ok(out)
    }

    fn get_wt1(&mut self, meta: Meta) -> Result<()> {
        let out = self.wt1(
            self.get_int(&meta, 0)? as usize,
            self.get_int(&meta, 1)? as usize,
        )?;
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_wt2(&mut self, meta: Meta) -> Result<()> {
        let wt1 = self.wt1(
            self.get_int(&meta, 0)? as usize,
            self.get_int(&meta, 1)? as usize,
        )?;
        self.add_column_f64(&meta.full_name(), rolling_mean_min1(&wt1, 4))
    }

    fn get_wt(&mut self, meta: Meta) -> Result<()> {
        let wt1 = self.wt1(
            self.get_int(&meta, 0)? as usize,
            self.get_int(&meta, 1)? as usize,
        )?;
        self.add_column_f64(&meta.name_ex("1"), wt1.clone())?;
        self.add_column_f64(&meta.name_ex("2"), rolling_mean_min1(&wt1, 4))
    }

    fn get_supertrend(&mut self, meta: Meta) -> Result<()> {
        let high = self.ensure_column_vec("high")?;
        let low = self.ensure_column_vec("low")?;
        let close = self.ensure_column_vec("close")?;
        let window = self.get_int(&meta, 0)? as usize;
        let atr = self.atr(window)?;

        let n = close.len();
        let mut final_ub = vec![0.0; n];
        let mut final_lb = vec![0.0; n];
        let mut st = vec![0.0; n];
        let mut direction = vec![0.0; n];

        for i in 1..n {
            let hl2 = (high[i] + low[i]) * 0.5;
            let basic_ub = hl2 + SUPERTREND_MUL * atr[i];
            let basic_lb = hl2 - SUPERTREND_MUL * atr[i];

            final_ub[i] = if basic_ub < final_ub[i - 1] || close[i - 1] > final_ub[i - 1] {
                basic_ub
            } else {
                final_ub[i - 1]
            };
            final_lb[i] = if basic_lb > final_lb[i - 1] || close[i - 1] < final_lb[i - 1] {
                basic_lb
            } else {
                final_lb[i - 1]
            };

            direction[i] = if close[i] > final_ub[i] {
                1.0
            } else if close[i] < final_lb[i] {
                -1.0
            } else {
                direction[i - 1]
            };
            st[i] = if direction[i] == 1.0 {
                final_lb[i]
            } else {
                final_ub[i]
            };
        }

        let base = meta.with_name("supertrend").full_name();
        self.add_column_f64(&base, st)?;
        self.add_column_f64(&format!("{}_ub", base), final_ub)?;
        self.add_column_f64(&format!("{}_lb", base), final_lb)
    }

    fn get_ao(&mut self, meta: Meta) -> Result<()> {
        let fast = self.get_int(&meta, 0)? as usize;
        let slow = self.get_int(&meta, 1)? as usize;
        let high = self.ensure_column_vec("high")?;
        let low = self.ensure_column_vec("low")?;
        let med: Vec<f64> = high
            .iter()
            .zip(low.iter())
            .map(|(h, l)| (h + l) * 0.5)
            .collect();
        let out: Vec<f64> = rolling_mean_min1(&med, fast)
            .iter()
            .zip(rolling_mean_min1(&med, slow).iter())
            .map(|(a, b)| a - b)
            .collect();
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_bop(&mut self, meta: Meta) -> Result<()> {
        let close = self.ensure_column_vec("close")?;
        let open = self.ensure_column_vec("open")?;
        let high = self.ensure_column_vec("high")?;
        let low = self.ensure_column_vec("low")?;
        let out: Vec<f64> = (0..close.len())
            .map(|i| {
                let den = high[i] - low[i];
                if den != 0.0 {
                    (close[i] - open[i]) / den
                } else {
                    0.0
                }
            })
            .collect();
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_cmo(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let close_diff = col_diff(&self.ensure_column_vec("close")?);
        let up: Vec<f64> = close_diff.iter().map(|d| d.max(0.0)).collect();
        let down: Vec<f64> = close_diff.iter().map(|d| (-d).max(0.0)).collect();
        let sum_up = rolling_sum_min1(&up, window);
        let sum_down = rolling_sum_min1(&down, window);
        let mut out: Vec<f64> = (0..sum_up.len())
            .map(|i| {
                let den = sum_up[i] + sum_down[i];
                if den != 0.0 {
                    100.0 * (sum_up[i] - sum_down[i]) / den
                } else {
                    0.0
                }
            })
            .collect();
        if !out.is_empty() {
            out[0] = 0.0;
        }
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_aroon(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let high = self.ensure_column_vec("high")?;
        let low = self.ensure_column_vec("low")?;
        let high_idx = rolling_arg_index(&high, window, true);
        let low_idx = rolling_arg_index(&low, window, false);

        let out: Vec<f64> = (0..high.len())
            .map(|i| {
                if i + 1 >= window {
                    let up = (window as f64 - high_idx[i]) / window as f64 * 100.0;
                    let down = (window as f64 - low_idx[i]) / window as f64 * 100.0;
                    up - down
                } else {
                    0.0
                }
            })
            .collect();
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_z(&mut self, meta: Meta) -> Result<()> {
        let w = self.get_int(&meta, 0)? as usize;
        let c = self.get_column_name_for_meta(&meta)?;
        let x = self.ensure_column_vec(&c)?;
        let mean = rolling_mean_min1(&x, w);
        let std = rolling_std_sample_min1(&x, w);
        let mut out: Vec<f64> = (0..x.len())
            .map(|i| {
                if std[i].is_finite() && std[i] != 0.0 {
                    (x[i] - mean[i]) / std[i]
                } else {
                    0.0
                }
            })
            .collect();
        if out.len() > 1 {
            out[0] = out[1];
        }
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_ichimoku(&mut self, meta: Meta) -> Result<()> {
        let conv = self.get_int(&meta, 0)? as usize;
        let base = self.get_int(&meta, 1)? as usize;
        let lead = self.get_int(&meta, 2)? as usize;
        let conv_line = self.hl_mid(conv)?;
        let base_line = self.hl_mid(base)?;
        let lead_a: Vec<f64> = conv_line
            .iter()
            .zip(base_line.iter())
            .map(|(a, b)| (a + b) * 0.5)
            .collect();
        let lead_b = self.hl_mid(lead)?;
        let la = shift_arr(&lead_a, -(base as i32));
        let lb = shift_arr(&lead_b, -(base as i32));
        let out: Vec<f64> = la.iter().zip(lb.iter()).map(|(a, b)| a - b).collect();
        self.add_column_f64(&meta.full_name(), out)
    }

    fn hl_mid(&mut self, period: usize) -> Result<Vec<f64>> {
        let high = rolling_max_min1(&self.ensure_column_vec("high")?, period);
        let low = rolling_min_min1(&self.ensure_column_vec("low")?, period);
        Ok(high
            .iter()
            .zip(low.iter())
            .map(|(a, b)| (a + b) * 0.5)
            .collect())
    }

    fn get_coppock(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let fast = self.get_int(&meta, 1)?;
        let slow = self.get_int(&meta, 2)?;
        let close = self.ensure_column_vec("close")?;
        let fast_roc = roc(&close, fast);
        let slow_roc = roc(&close, slow);
        let summed: Vec<f64> = fast_roc
            .iter()
            .zip(slow_roc.iter())
            .map(|(a, b)| a + b)
            .collect();
        self.add_column_f64(&meta.full_name(), linear_wma(&summed, window))
    }

    fn get_eribull(&mut self, meta: Meta) -> Result<()> {
        let (bull, _) = self.eri(self.get_int(&meta, 0)? as usize)?;
        self.add_column_f64(&meta.full_name(), bull)
    }

    fn get_eribear(&mut self, meta: Meta) -> Result<()> {
        let (_, bear) = self.eri(self.get_int(&meta, 0)? as usize)?;
        self.add_column_f64(&meta.full_name(), bear)
    }

    fn get_eri(&mut self, meta: Meta) -> Result<()> {
        let (bull, bear) = self.eri(self.get_int(&meta, 0)? as usize)?;
        self.add_column_f64(&meta.name_ex("bull"), bull)?;
        self.add_column_f64(&meta.name_ex("bear"), bear)
    }

    fn eri(&mut self, window: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        let close = self.ensure_column_vec("close")?;
        let high = self.ensure_column_vec("high")?;
        let low = self.ensure_column_vec("low")?;
        let ema = ewm_adjust_false(&close, 2.0 / (window as f64 + 1.0));
        let bull: Vec<f64> = high.iter().zip(ema.iter()).map(|(h, e)| h - e).collect();
        let bear: Vec<f64> = low.iter().zip(ema.iter()).map(|(l, e)| l - e).collect();
        Ok((bull, bear))
    }

    fn get_ftr(&mut self, meta: Meta) -> Result<()> {
        let out = self.ftr(self.get_int(&meta, 0)? as usize)?;
        self.add_column_f64(&meta.full_name(), out)
    }

    fn ftr(&mut self, window: usize) -> Result<Vec<f64>> {
        let high = self.ensure_column_vec("high")?;
        let low = self.ensure_column_vec("low")?;
        let mp: Vec<f64> = high
            .iter()
            .zip(low.iter())
            .map(|(h, l)| (h + l) * 0.5)
            .collect();
        let highest = rolling_max_min1(&mp, window);
        let lowest = rolling_min_min1(&mp, window);

        let mut result = vec![0.0; mp.len()];
        let mut v = 0.0;
        for i in window..mp.len() {
            let width = (highest[i] - lowest[i]).max(0.001);
            let pos = (mp[i] - lowest[i]) / width - 0.5;
            v = 0.66 * pos + 0.67 * v;
            v = v.clamp(-0.999, 0.999);
            result[i] = 0.5 * (((1.0 + v) / (1.0 - v)).ln() + result[i - 1]);
        }
        Ok(result)
    }

    fn get_rvgis(&mut self, meta: Meta) -> Result<()> {
        self.get_rvgi(Meta::new("rvgi", None, meta.windows.clone()))
    }

    fn get_rvgi(&mut self, meta: Meta) -> Result<()> {
        let window = self.get_int(&meta, 0)? as usize;
        let mut rvgi = self.rvgi(window)?;
        let lim1 = usize::min(3, rvgi.len());
        rvgi[..lim1].fill(0.0);
        let mut signal = sym_wma4(&rvgi);
        let lim2 = usize::min(6, signal.len());
        signal[..lim2].fill(0.0);
        self.add_column_f64(&meta.full_name(), rvgi)?;
        self.add_column_f64(&meta.name_ex("s"), signal)
    }

    fn rvgi(&mut self, window: usize) -> Result<Vec<f64>> {
        let close = self.ensure_column_vec("close")?;
        let open = self.ensure_column_vec("open")?;
        let high = self.ensure_column_vec("high")?;
        let low = self.ensure_column_vec("low")?;

        let co: Vec<f64> = close.iter().zip(open.iter()).map(|(c, o)| c - o).collect();
        let hl: Vec<f64> = high.iter().zip(low.iter()).map(|(h, l)| h - l).collect();

        let nu = sym_wma4(&co);
        let de = sym_wma4(&hl);
        let num = rolling_mean_min1(&nu, window);
        let den = rolling_mean_min1(&de, window);
        Ok(num
            .iter()
            .zip(den.iter())
            .map(|(a, b)| if *b != 0.0 { a / b } else { 0.0 })
            .collect())
    }

    fn inertia(&mut self, window: usize, rvgi_window: usize) -> Result<Vec<f64>> {
        if self.df.height() < window + rvgi_window {
            return Ok(vec![0.0; self.df.height()]);
        }
        let rvgi = self.rvgi(rvgi_window)?;
        let mut v = linear_reg(&rvgi, window, false);
        let limit = usize::min(v.len(), usize::max(window, rvgi_window) + 2);
        v[..limit].fill(0.0);
        Ok(v)
    }

    fn get_inertia(&mut self, meta: Meta) -> Result<()> {
        let w = self.get_int(&meta, 0)? as usize;
        let rw = self.get_int(&meta, 1)? as usize;
        let out = self.inertia(w, rw)?;
        self.add_column_f64(&meta.full_name(), out)
    }

    fn kst(&mut self) -> Result<Vec<f64>> {
        let close = self.ensure_column_vec("close")?;
        let ma1 = rolling_mean_min1(&roc(&close, 10), 10);
        let ma2 = rolling_mean_min1(&roc(&close, 15), 10);
        let ma3 = rolling_mean_min1(&roc(&close, 20), 10);
        let ma4 = rolling_mean_min1(&roc(&close, 30), 15);
        Ok((0..close.len())
            .map(|i| ma1[i] + ma2[i] * 2.0 + ma3[i] * 3.0 + ma4[i] * 4.0)
            .collect())
    }

    fn get_kst(&mut self, meta: Meta) -> Result<()> {
        let out = self.kst()?;
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_pgo(&mut self, meta: Meta) -> Result<()> {
        let w = self.get_int(&meta, 0)? as usize;
        let close = self.ensure_column_vec("close")?;
        let up: Vec<f64> = close
            .iter()
            .zip(rolling_mean_min1(&close, w).iter())
            .map(|(a, b)| a - b)
            .collect();
        let tr = self.tr()?;
        let down = ewm_adjust(&tr, 2.0 / (w as f64 + 1.0));
        let out: Vec<f64> = up
            .iter()
            .zip(down.iter())
            .map(|(a, b)| if *b != 0.0 { a / b } else { 0.0 })
            .collect();
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_psl(&mut self, meta: Meta) -> Result<()> {
        let w = self.get_int(&meta, 0)? as usize;
        let c = self.get_column_name_for_meta(&meta)?;
        let diff = col_diff(&self.ensure_column_vec(&c)?);
        let up: Vec<f64> = diff
            .iter()
            .map(|d| if *d > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let out: Vec<f64> = rolling_sum_min1(&up, w)
            .iter()
            .map(|v| v / w as f64 * 100.0)
            .collect();
        self.add_column_f64(&meta.full_name(), out)
    }

    fn get_qqe(&mut self, meta: Meta) -> Result<()> {
        let rsi_window = self.get_int(&meta, 0)? as usize;
        let rsi_ma_window = self.get_int(&meta, 1)? as usize;
        let factor = 4.236;
        let wilder_window = rsi_window * 2 - 1;

        let mut rsi = self.rsi_vec(rsi_window)?;
        let n_nan = usize::min(rsi_window, rsi.len());
        for v in rsi.iter_mut().take(n_nan) {
            *v = f64::NAN;
        }

        let rsi_ma = ewm_adjust_false_nan(&rsi, 2.0 / (rsi_ma_window as f64 + 1.0));
        let tr: Vec<f64> = diff_abs(&rsi_ma);
        let tr_ma = ewm_adjust_false_nan(&tr, 2.0 / (wilder_window as f64 + 1.0));
        let tr_ma_ma: Vec<f64> = ewm_adjust_false_nan(&tr_ma, 2.0 / (wilder_window as f64 + 1.0))
            .iter()
            .map(|v| v * factor)
            .collect();

        let size = rsi_ma.len();
        let upper: Vec<f64> = (0..size).map(|i| rsi_ma[i] + tr_ma_ma[i]).collect();
        let lower: Vec<f64> = (0..size).map(|i| rsi_ma[i] - tr_ma_ma[i]).collect();

        let mut out_long = vec![0.0; size];
        let mut out_short = vec![0.0; size];
        let mut out_trend = vec![1_i32; size];
        let mut out_qqe = vec![if size > 0 { rsi_ma[0] } else { 0.0 }; size];
        let mut out_qqe_long = vec![f64::NAN; size];
        let mut out_qqe_short = vec![f64::NAN; size];

        for i in 1..size {
            let c_rsi = rsi_ma[i];
            let p_rsi = rsi_ma[i - 1];

            let p_long = out_long[i - 1];
            out_long[i] = if p_rsi > p_long && c_rsi > p_long {
                p_long.max(lower[i])
            } else {
                lower[i]
            };

            let p_short = out_short[i - 1];
            out_short[i] = if p_rsi < p_short && c_rsi < p_short {
                p_short.min(upper[i])
            } else {
                upper[i]
            };

            if c_rsi > p_short && p_rsi <= out_short[i.saturating_sub(2)] {
                out_trend[i] = 1;
            } else if c_rsi < p_long && p_rsi >= out_long[i.saturating_sub(2)] {
                out_trend[i] = -1;
            } else {
                out_trend[i] = out_trend[i - 1];
            }

            if out_trend[i] == 1 {
                out_qqe[i] = out_long[i];
                out_qqe_long[i] = out_long[i];
            } else {
                out_trend[i] = -1;
                out_qqe[i] = out_short[i];
                out_qqe_short[i] = out_short[i];
            }
        }

        self.add_column_f64(&meta.full_name(), nan_to_zero(&out_qqe))?;
        self.add_column_f64(&meta.name_ex("l"), nan_to_zero(&out_qqe_long))?;
        self.add_column_f64(&meta.name_ex("s"), nan_to_zero(&out_qqe_short))
    }

    fn get_log_ret(&mut self) -> Result<()> {
        let close = self.ensure_column_vec("close")?;
        let prev = shift_arr(&close, -1);
        let out: Vec<f64> = close
            .iter()
            .zip(prev.iter())
            .map(|(c, p)| {
                if *p > 0.0 && *c > 0.0 {
                    (c / p).ln()
                } else {
                    0.0
                }
            })
            .collect();
        self.add_column_f64("log-ret", out)
    }

    fn get_compare(&mut self, key: &str) -> Result<()> {
        let (left, op, right) = parse_compare_column(key)
            .ok_or_else(|| StockStatsError::InvalidColumn(key.to_owned()))?;
        let l = self.ensure_column_vec(&left)?;
        let r = if self.has_column(&right)
            || parse_column_name(&right) != Parsed::NoMatch
            || right.contains('_')
        {
            self.ensure_column_vec(&right)?
        } else if let Ok(v) = right.parse::<f64>() {
            vec![v; l.len()]
        } else {
            self.ensure_column_vec(&right)?
        };
        let out: Vec<bool> = l
            .iter()
            .zip(r.iter())
            .map(|(a, b)| match op.as_str() {
                "le" => a <= b,
                "ge" => a >= b,
                "lt" => a < b,
                "gt" => a > b,
                "eq" => a == b,
                "ne" => a != b,
                _ => false,
            })
            .collect();
        self.add_column_bool(key, out)
    }

    fn get_cross(&mut self, key: &str) -> Result<()> {
        let (left, op, right) = parse_cross_column(key)
            .ok_or_else(|| StockStatsError::InvalidColumn(key.to_owned()))?;
        let l = self.ensure_column_vec(&left)?;
        let r = self.ensure_column_vec(&right)?;
        let lt: Vec<bool> = l.iter().zip(r.iter()).map(|(a, b)| a > b).collect();
        let mut different = vec![false; lt.len()];
        for i in 1..lt.len() {
            different[i] = lt[i] != lt[i - 1];
        }
        let out: Vec<bool> = (0..lt.len())
            .map(|i| match op.as_str() {
                "x" => different[i],
                "xu" => different[i] && lt[i],
                "xd" => different[i] && !lt[i],
                _ => false,
            })
            .collect();
        self.add_column_bool(key, out)
    }
}

fn parse_int(text: &str) -> Result<i32> {
    text.trim()
        .parse::<i32>()
        .map_err(|_| StockStatsError::InvalidWindow(text.to_owned()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Parsed {
    Three(String, String, String),
    Two(String, String),
    NoMatch,
}

fn parse_column_name(name: &str) -> Parsed {
    let re_three = Regex::new(r"(.*)_([\d\-+~,.]+)_(\w+)").expect("valid regex");
    if let Some(m) = re_three.captures(name) {
        return Parsed::Three(
            m.get(1).map_or("", |m| m.as_str()).to_owned(),
            m.get(2).map_or("", |m| m.as_str()).to_owned(),
            m.get(3).map_or("", |m| m.as_str()).to_owned(),
        );
    }

    let re_two = Regex::new(r"(.*)_([\d\-+~,]+)").expect("valid regex");
    if let Some(m) = re_two.captures(name) {
        return Parsed::Two(
            m.get(1).map_or("", |m| m.as_str()).to_owned(),
            m.get(2).map_or("", |m| m.as_str()).to_owned(),
        );
    }
    Parsed::NoMatch
}

fn is_cross_columns(name: &str) -> bool {
    Regex::new(r"(.+)_(x|xu|xd)_(.+)")
        .expect("regex")
        .is_match(name)
}

fn parse_cross_column(name: &str) -> Option<(String, String, String)> {
    let re = Regex::new(r"(.+)_(x|xu|xd)_(.+)").expect("regex");
    re.captures(name).map(|m| {
        (
            m.get(1).map_or("", |x| x.as_str()).to_owned(),
            m.get(2).map_or("", |x| x.as_str()).to_owned(),
            m.get(3).map_or("", |x| x.as_str()).to_owned(),
        )
    })
}

fn is_compare_columns(name: &str) -> bool {
    Regex::new(r"(.+)_(le|ge|lt|gt|eq|ne)_(.+)")
        .expect("regex")
        .is_match(name)
}

fn parse_compare_column(name: &str) -> Option<(String, String, String)> {
    let re = Regex::new(r"(.+)_(le|ge|lt|gt|eq|ne)_(.+)").expect("regex");
    re.captures(name).map(|m| {
        (
            m.get(1).map_or("", |x| x.as_str()).to_owned(),
            m.get(2).map_or("", |x| x.as_str()).to_owned(),
            m.get(3).map_or("", |x| x.as_str()).to_owned(),
        )
    })
}

fn roc(x: &[f64], size: i32) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0; n];
    if size == 0 {
        return out;
    }

    if size > 0 {
        let k = size as usize;
        for i in k..n {
            if x[i - k] != 0.0 {
                out[i] = (x[i] - x[i - k]) / x[i - k] * 100.0;
            }
        }
    } else {
        let k = (-size) as usize;
        for i in 0..n.saturating_sub(k) {
            if x[i + k] != 0.0 {
                out[i] = (x[i] - x[i + k]) / x[i + k] * 100.0;
            }
        }
    }
    out
}

fn np_diff(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let mut out = vec![0.0; x.len()];
    for i in 1..x.len() {
        out[i] = x[i] - x[i - 1];
    }
    out
}

fn np_diff_neg(x: &[f64]) -> Vec<f64> {
    let d = np_diff(x);
    d.into_iter().map(|v| -v).collect()
}

fn col_diff(x: &[f64]) -> Vec<f64> {
    np_diff(x)
}

fn delta(x: &[f64], window: i32) -> Vec<f64> {
    if x.is_empty() || window == 0 {
        return vec![0.0; x.len()];
    }
    let mut out = vec![0.0; x.len()];
    if window > 0 {
        let k = window as usize;
        for i in 0..x.len().saturating_sub(k) {
            out[i] = x[i] - x[i + k];
        }
    } else {
        let k = (-window) as usize;
        for i in k..x.len() {
            out[i] = x[i] - x[i - k];
        }
    }
    out
}

fn shift_arr(x: &[f64], window: i32) -> Vec<f64> {
    if x.is_empty() || window == 0 {
        return x.to_vec();
    }

    let n = x.len();
    let mut out = vec![0.0; n];
    if window < 0 {
        let k = (-window) as usize;
        if k >= n {
            out.fill(x[0]);
            return out;
        }
        out[..k].fill(x[0]);
        out[k..n].copy_from_slice(&x[..n - k]);
    } else {
        let k = window as usize;
        if k >= n {
            out.fill(x[n - 1]);
            return out;
        }
        out[..n - k].copy_from_slice(&x[k..n]);
        out[n - k..].fill(x[n - 1]);
    }
    out
}

fn divide_non_zero(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| if *y != 0.0 { x / y } else { 0.0 })
        .collect()
}

fn rolling_sum_min1(x: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let mut out = vec![0.0; n];
    let mut cumsum = vec![0.0; n];
    cumsum[0] = x[0];
    for i in 1..n {
        cumsum[i] = cumsum[i - 1] + x[i];
    }
    for i in 0..n {
        if i >= window {
            out[i] = cumsum[i] - cumsum[i - window];
        } else {
            out[i] = cumsum[i];
        }
    }
    out
}

fn rolling_mean_min1(x: &[f64], window: usize) -> Vec<f64> {
    let sum = rolling_sum_min1(x, window);
    sum.iter()
        .enumerate()
        .map(|(i, s)| {
            let cnt = usize::min(window, i + 1);
            *s / cnt as f64
        })
        .collect()
}

fn rolling_min_min1(x: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0; n];
    for i in 0..n {
        let start = (i + 1).saturating_sub(window);
        let mut v = f64::INFINITY;
        for &item in &x[start..=i] {
            v = v.min(item);
        }
        out[i] = v;
    }
    out
}

fn rolling_max_min1(x: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0; n];
    for i in 0..n {
        let start = (i + 1).saturating_sub(window);
        let mut v = f64::NEG_INFINITY;
        for &item in &x[start..=i] {
            v = v.max(item);
        }
        out[i] = v;
    }
    out
}

fn rolling_std_sample_min1(x: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        let start = (i + 1).saturating_sub(window);
        let slice = &x[start..=i];
        let len = slice.len();
        if len < 2 {
            out[i] = f64::NAN;
            continue;
        }
        let mean = slice.iter().sum::<f64>() / len as f64;
        let var = slice
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f64>()
            / (len as f64 - 1.0);
        out[i] = var.sqrt();
    }
    out
}

fn rolling_var_sample_min1(x: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        let start = (i + 1).saturating_sub(window);
        let slice = &x[start..=i];
        let len = slice.len();
        if len < 2 {
            out[i] = f64::NAN;
            continue;
        }
        let mean = slice.iter().sum::<f64>() / len as f64;
        out[i] = slice
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f64>()
            / (len as f64 - 1.0);
    }
    out
}

fn ewm_adjust(x: &[f64], alpha: f64) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    let mut num = 0.0;
    let mut den = 0.0;
    let w = 1.0 - alpha;
    for (i, v) in x.iter().enumerate() {
        num = *v + w * num;
        den = 1.0 + w * den;
        out[i] = num / den;
    }
    out
}

fn ewm_adjust_false(x: &[f64], alpha: f64) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let mut out = vec![0.0; x.len()];
    out[0] = x[0];
    for i in 1..x.len() {
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1];
    }
    out
}

fn ewm_adjust_false_nan(x: &[f64], alpha: f64) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let mut out = vec![f64::NAN; x.len()];
    let mut started = false;
    for i in 0..x.len() {
        if !started {
            if x[i].is_finite() {
                out[i] = x[i];
                started = true;
            }
            continue;
        }
        if x[i].is_finite() {
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1];
        } else {
            out[i] = out[i - 1];
        }
    }
    out
}

fn mad(x: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    if window > n {
        return vec![f64::NAN; n];
    }

    let mut out = vec![0.0; n];
    for i in (window - 1)..n {
        let start = i + 1 - window;
        let slice = &x[start..=i];
        let mean = slice.iter().sum::<f64>() / window as f64;
        out[i] = slice.iter().map(|v| (v - mean).abs()).sum::<f64>() / window as f64;
    }
    out
}

fn ker(x: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    let mut net = vec![0.0; n];
    for i in window..n {
        net[i] = (x[i] - x[i - window]).abs();
    }
    let mut abs_diff = vec![0.0; n];
    for i in 1..n {
        abs_diff[i] = (x[i] - x[i - 1]).abs();
    }
    let vol = rolling_sum_min1(&abs_diff, window);
    let mut out: Vec<f64> = (0..n)
        .map(|i| if vol[i] > 0.0 { net[i] / vol[i] } else { 0.0 })
        .collect();
    let lim = usize::min(window, out.len());
    out[..lim].fill(0.0);
    out
}

fn linear_wma(x: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    if window > n {
        return vec![0.0; n];
    }
    let total_weight = 0.5 * window as f64 * (window as f64 + 1.0);
    let mut out = vec![0.0; n];
    for i in (window - 1)..n {
        let mut v = 0.0;
        for k in 0..window {
            v += x[i + 1 - window + k] * (k + 1) as f64;
        }
        out[i] = v / total_weight;
    }
    out
}

fn linear_reg(x: &[f64], window: usize, correlation: bool) -> Vec<f64> {
    let n = x.len();
    if window > n {
        return vec![0.0; n];
    }
    let mut out = vec![0.0; n];

    let mut x_sum = 0.0;
    let mut x2_sum = 0.0;
    for i in 1..=window {
        let f = i as f64;
        x_sum += f;
        x2_sum += f * f;
    }
    let divisor = window as f64 * x2_sum - x_sum * x_sum;

    for i in (window - 1)..n {
        let start = i + 1 - window;
        let slice = &x[start..=i];
        let y_sum: f64 = slice.iter().sum();
        let xy_sum: f64 = slice
            .iter()
            .enumerate()
            .map(|(idx, y)| (idx as f64 + 1.0) * y)
            .sum();

        out[i] = if correlation {
            let y2_sum: f64 = slice.iter().map(|v| v * v).sum();
            let rn = window as f64 * xy_sum - x_sum * y_sum;
            let rd = (divisor * (window as f64 * y2_sum - y_sum * y_sum)).sqrt();
            if rd != 0.0 { rn / rd } else { 0.0 }
        } else {
            let m = (window as f64 * xy_sum - x_sum * y_sum) / divisor;
            let b = (y_sum * x2_sum - x_sum * xy_sum) / divisor;
            m * (window as f64 - 1.0) + b
        };
    }
    out
}

fn rolling_arg_index(arr: &[f64], window: usize, max_mode: bool) -> Vec<f64> {
    let n = arr.len();
    let mut out = vec![0.0; n];
    let mut dq: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        while let Some(&f) = dq.front() {
            if f <= i.saturating_sub(window) {
                dq.pop_front();
            } else {
                break;
            }
        }

        while let Some(&b) = dq.back() {
            let cond = if max_mode {
                arr[b] <= arr[i]
            } else {
                arr[b] >= arr[i]
            };
            if cond {
                dq.pop_back();
            } else {
                break;
            }
        }
        dq.push_back(i);

        if i + 1 >= window
            && let Some(&f) = dq.front()
        {
            out[i] = (i - f) as f64;
        }
    }
    out
}

fn diff_abs(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let mut out = vec![f64::NAN; x.len()];
    for i in 1..x.len() {
        if x[i].is_finite() && x[i - 1].is_finite() {
            out[i] = (x[i] - x[i - 1]).abs();
        }
    }
    out
}

fn sym_wma4(arr: &[f64]) -> Vec<f64> {
    let n = arr.len();
    let mut out = vec![0.0; n];
    if n < 4 {
        return out;
    }
    for i in 3..n {
        out[i] = (arr[i - 3] + 2.0 * arr[i - 2] + 2.0 * arr[i - 1] + arr[i]) / 6.0;
    }
    out
}

fn nan_to_zero(x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|v| if v.is_finite() { *v } else { 0.0 })
        .collect()
}

fn set_nan_by_shifts(x: &mut [f64], shifts: &[i32]) {
    if shifts.is_empty() {
        return;
    }
    let max_shift = *shifts.iter().max().unwrap_or(&0);
    let min_shift = *shifts.iter().min().unwrap_or(&0);
    set_nan_single_shift(x, max_shift);
    set_nan_single_shift(x, min_shift);
}

fn set_nan_single_shift(x: &mut [f64], shift: i32) {
    if shift > 0 {
        let k = shift as usize;
        let n = x.len();
        if k <= n {
            x[n - k..].fill(f64::NAN);
        }
    } else if shift < 0 {
        let k = (-shift) as usize;
        let n = usize::min(k, x.len());
        x[..n].fill(f64::NAN);
    }
}
