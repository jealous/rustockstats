use approx::assert_abs_diff_eq;
use polars::prelude::*;
use rustockstats::{ParsedColumnName, StockDataFrame, StockStatsError};

fn load_csv_df(path: &str) -> DataFrame {
    let file = std::fs::File::open(path).expect("open csv");
    CsvReader::new(file).finish().expect("read csv")
}

fn load_csv(path: &str) -> StockDataFrame {
    StockDataFrame::retype(load_csv_df(path)).expect("retype")
}

fn filter_within(df: DataFrame, start: i64, end: i64) -> DataFrame {
    let date = df
        .column("date")
        .expect("date")
        .as_materialized_series()
        .cast(&DataType::Int64)
        .expect("cast date")
        .i64()
        .expect("date i64")
        .clone();
    let mask = &date.gt_eq(start) & &date.lt_eq(end);
    df.filter(&mask).expect("filter within")
}

fn stock_20days() -> StockDataFrame {
    StockDataFrame::retype(filter_within(
        load_csv_df("tests/data/987654.csv"),
        20110101,
        20110120,
    ))
    .expect("retype")
}

fn stock_30days() -> StockDataFrame {
    StockDataFrame::retype(filter_within(
        load_csv_df("tests/data/987654.csv"),
        20110101,
        20110130,
    ))
    .expect("retype")
}

fn stock_90days() -> StockDataFrame {
    StockDataFrame::retype(filter_within(
        load_csv_df("tests/data/987654.csv"),
        20110101,
        20110331,
    ))
    .expect("retype")
}

fn supor() -> StockDataFrame {
    load_csv("tests/data/002032.csv")
}

fn supor_50days() -> StockDataFrame {
    StockDataFrame::retype(filter_within(
        load_csv_df("tests/data/002032.csv"),
        20040817,
        20041031,
    ))
    .expect("retype")
}

fn supor_100days() -> StockDataFrame {
    StockDataFrame::retype(filter_within(
        load_csv_df("tests/data/002032.csv"),
        20040817,
        20050101,
    ))
    .expect("retype")
}

fn get_idx(sdf: &StockDataFrame, date: i64) -> usize {
    let date_col = sdf
        .df()
        .column("date")
        .expect("date")
        .as_materialized_series()
        .cast(&DataType::Int64)
        .expect("cast date");
    date_col
        .i64()
        .expect("date i64")
        .into_no_null_iter()
        .position(|d| d == date)
        .expect("date row")
}

fn get_val(sdf: &mut StockDataFrame, col: &str, date: i64) -> f64 {
    let _ = sdf.get(col).expect("compute col");
    let idx = get_idx(sdf, date);
    let col_vals = sdf
        .df()
        .column(col)
        .expect("col")
        .as_materialized_series()
        .cast(&DataType::Float64)
        .expect("cast col");
    col_vals.f64().expect("f64").get(idx).expect("value exists")
}

fn get_val_at(sdf: &mut StockDataFrame, col: &str, idx: usize) -> f64 {
    let _ = sdf.get(col).expect("compute col");
    let col_vals = sdf
        .df()
        .column(col)
        .expect("col")
        .as_materialized_series()
        .cast(&DataType::Float64)
        .expect("cast col");
    col_vals.f64().expect("f64").get(idx).unwrap_or(f64::NAN)
}

fn get_bool(sdf: &mut StockDataFrame, col: &str, date: i64) -> bool {
    let _ = sdf.get(col).expect("compute col");
    let idx = get_idx(sdf, date);
    sdf.df()
        .column(col)
        .expect("col")
        .as_materialized_series()
        .bool()
        .expect("bool")
        .get(idx)
        .expect("bool value")
}

fn get_i64(sdf: &mut StockDataFrame, col: &str, date: i64) -> i64 {
    let _ = sdf.get(col).expect("compute col");
    let idx = get_idx(sdf, date);
    let s = sdf.df().column(col).expect("col").as_materialized_series();
    if let Ok(i64s) = s.i64() {
        i64s.get(idx).expect("i64")
    } else {
        s.cast(&DataType::Int64)
            .expect("cast i64")
            .i64()
            .expect("i64")
            .get(idx)
            .expect("i64")
    }
}

fn colnames(sdf: &StockDataFrame) -> Vec<String> {
    sdf.df()
        .get_column_names_str()
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
}

fn len_df(sdf: &StockDataFrame) -> usize {
    sdf.df().height()
}

fn first_existing_col(df: &DataFrame, preferred: &str, fallback: &str) -> String {
    if df.get_column_index(preferred).is_some() {
        preferred.to_string()
    } else {
        fallback.to_string()
    }
}

#[test]
fn test_wrap_yfinance() {
    let mut stock = StockDataFrame::retype(filter_within(
        load_csv_df("tests/data/002032_yf_2020_2021.csv"),
        20200101,
        20211231,
    ))
    .expect("retype");
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_20_dma", 20210104),
        0.7459,
        epsilon = 1e-3
    );
}

#[test]
fn test_kdj_of_yfinance() {
    let mut stock = StockDataFrame::retype(filter_within(
        load_csv_df("tests/data/002032_yf_2020_2021.csv"),
        20200101,
        20211231,
    ))
    .expect("retype");
    assert_abs_diff_eq!(
        get_val(&mut stock, "kdjk", 20210104),
        69.54346,
        epsilon = 1e-3
    );
}

#[test]
fn yfinance_test_get_wr() {
    let mut stock = supor();
    assert_abs_diff_eq!(
        get_val(&mut stock, "wr", 20160817),
        -49.1621,
        epsilon = 1e-3
    );
}

#[test]
fn yfinance_test_get_adx() {
    let mut stock = supor();
    assert_abs_diff_eq!(
        get_val(&mut stock, "adx", 20160817),
        15.5378,
        epsilon = 1e-2
    );
}

#[test]
fn test_delta() {
    let mut stock = load_csv("tests/data/987654.csv");
    let _ = stock.get("volume_delta").expect("volume_delta");
    assert!(stock.df().column("volume_delta").expect("col").len() > 1);
    assert_abs_diff_eq!(
        get_val(&mut stock, "volume_delta", 20141219),
        -63383600.0,
        epsilon = 1e-3
    );
}

#[test]
fn test_must_have_positive_int() {
    let err = StockDataFrame::get_int_positive_for_test("-54").expect_err("must error");
    assert!(matches!(err, StockStatsError::InvalidWindow(_)));
}

#[test]
fn test_new_works_like_retype() {
    let df = df![
        "date" => vec![1_i64, 2],
        "Open" => vec![10.0, 11.0],
        "High" => vec![12.0, 13.0],
        "Low" => vec![9.0, 10.0],
        "Close" => vec![11.0, 12.0],
        "Volume" => vec![100.0, 120.0],
        "Amount" => vec![1000.0, 1200.0],
        "custom" => vec![1.0, 2.0],
    ]
    .expect("df");

    let stock_new = StockDataFrame::new(df.clone()).expect("new");
    let stock_retype = StockDataFrame::retype(df).expect("retype");

    let new_cols = colnames(&stock_new);
    let retype_cols = colnames(&stock_retype);
    assert_eq!(new_cols, retype_cols);
    assert!(new_cols.contains(&"open".to_string()));
    assert!(new_cols.contains(&"high".to_string()));
    assert!(new_cols.contains(&"low".to_string()));
    assert!(new_cols.contains(&"close".to_string()));
    assert!(new_cols.contains(&"volume".to_string()));
    assert!(new_cols.contains(&"amount".to_string()));
    assert!(!new_cols.contains(&"Open".to_string()));
}

#[test]
fn test_multiple_columns() {
    let stock = load_csv("tests/data/987654.csv");
    let sel = stock.df().select(["open", "close"]).expect("select");
    assert_eq!(sel.get_column_names_str(), vec!["open", "close"]);
}

#[test]
fn test_column_le_count() {
    let mut df = filter_within(load_csv_df("tests/data/987654.csv"), 20110101, 20110120);
    let close_name = first_existing_col(&df, "close", "Close");
    let close = df
        .column(&close_name)
        .expect("close")
        .as_materialized_series()
        .f64()
        .expect("f64");
    let res: Vec<bool> = close.into_no_null_iter().map(|v| v <= 13.01).collect();
    df.with_column(Series::new("res".into(), res)).expect("res");
    let mut stock = StockDataFrame::retype(df).expect("retype");
    assert_eq!(get_i64(&mut stock, "res_5_c", 20110117), 1);
    assert_eq!(get_i64(&mut stock, "res_5_c", 20110119), 3);
}

#[test]
fn test_column_ge_future_count() {
    let mut df = filter_within(load_csv_df("tests/data/987654.csv"), 20110101, 20110120);
    let close_name = first_existing_col(&df, "close", "Close");
    let close = df
        .column(&close_name)
        .expect("close")
        .as_materialized_series()
        .f64()
        .expect("f64");
    let res: Vec<bool> = close.into_no_null_iter().map(|v| v >= 12.8).collect();
    df.with_column(Series::new("res".into(), res)).expect("res");
    let mut stock = StockDataFrame::retype(df).expect("retype");
    assert_eq!(get_i64(&mut stock, "res_5_fc", 20110119), 1);
    assert_eq!(get_i64(&mut stock, "res_5_fc", 20110117), 1);
    assert_eq!(get_i64(&mut stock, "res_5_fc", 20110113), 3);
    assert_eq!(get_i64(&mut stock, "res_5_fc", 20110111), 4);
}

#[test]
fn test_column_delta() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "open_-1_d", 20110104),
        0.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "open_-1_d", 20110120),
        0.07,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_delta_p2() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "open_2_d", 20110104),
        -0.31,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "open_2_d", 20110119),
        0.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "open_2_d", 20110118),
        -0.2,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_rate_minus_2() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "open_-2_r", 20110105),
        0.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "open_-2_r", 20110106),
        2.495,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_rate_prev() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "rate", 20110107),
        4.4198,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_rate_plus2() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "open_2_r", 20110118),
        -1.566,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "open_2_r", 20110119),
        0.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "open_2_r", 20110120),
        0.0,
        epsilon = 1e-8
    );
}

#[test]
fn test_change() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(get_val(&mut stock, "change", 20110104), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(
        get_val(&mut stock, "change", 20110105),
        0.793,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "change", 20110107),
        4.4198,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "change_2", 20110104),
        0.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "change_2", 20110105),
        0.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "change_2", 20110106),
        0.476,
        epsilon = 1e-3
    );
}

#[test]
fn test_middle() {
    let mut stock = stock_20days();
    let idx = 20110104;
    let middle = get_val(&mut stock, "middle", idx);
    let tp = get_val(&mut stock, "tp", idx);
    assert_abs_diff_eq!(middle, 12.53, epsilon = 1e-3);
    assert_abs_diff_eq!(tp, middle, epsilon = 1e-8);
}

#[test]
fn test_typical_price_with_amount() {
    let mut stock = StockDataFrame::retype(filter_within(
        load_csv_df("tests/data/002032.csv"),
        20040817,
        20040913,
    ))
    .expect("retype");
    assert_abs_diff_eq!(get_val(&mut stock, "tp", 20040817), 11.541, epsilon = 1e-3);
    assert_abs_diff_eq!(
        get_val(&mut stock, "middle", 20040817),
        11.541,
        epsilon = 1e-3
    );
}

#[test]
fn test_cr() {
    let mut stock = stock_90days();
    let _ = stock.get("cr").expect("cr");
    assert_abs_diff_eq!(
        get_val(&mut stock, "cr", 20110331),
        178.1714,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "cr-ma1", 20110331),
        120.0364,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "cr-ma2", 20110331),
        117.1488,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "cr-ma3", 20110331),
        111.5195,
        epsilon = 1e-3
    );
    let _ = stock.get("cr_26").expect("cr_26");
    assert_abs_diff_eq!(
        get_val(&mut stock, "cr_26", 20110331),
        178.1714,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "cr_26-ma1", 20110331),
        120.0364,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "cr_26-ma2", 20110331),
        117.1488,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "cr_26-ma3", 20110331),
        111.5195,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_permutation() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "volume_-1_d_-3,-2,-1_p", 20110107),
        2.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "volume_-1_d_-3,-2,-1_p", 20110110),
        5.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "volume_-1_d_-3,-2,-1_p", 20110111),
        2.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "volume_-1_d_-3,-2,-1_p", 20110112),
        4.0,
        epsilon = 1e-8
    );
    assert!(get_val(&mut stock, "volume_-1_d_-3,-2,-1_p", 20110104).is_nan());
    assert!(get_val(&mut stock, "volume_-1_d_-3,-2,-1_p", 20110105).is_nan());
    assert!(get_val(&mut stock, "volume_-1_d_-3,-2,-1_p", 20110106).is_nan());
}

#[test]
fn test_column_max() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "volume_-3,2,-1_max", 20110106),
        166409700.0,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "volume_-3,2,-1_max", 20110120),
        110664100.0,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "volume_-3,2,-1_max", 20110112),
        362436800.0,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_min() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "volume_-3~1_min", 20110106),
        83140300.0,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "volume_-3~1_min", 20110120),
        50888500.0,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "volume_-3~1_min", 20110112),
        72035800.0,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_shift_positive() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_2_s", 20110118),
        12.48,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_2_s", 20110119),
        12.48,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_2_s", 20110120),
        12.48,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_shift_zero() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_0_s", 20110118),
        12.69,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_0_s", 20110119),
        12.82,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_0_s", 20110120),
        12.48,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_shift_negative() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_-2_s", 20110104),
        12.61,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_-2_s", 20110105),
        12.61,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_-2_s", 20110106),
        12.61,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_-2_s", 20110107),
        12.71,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_rsv() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "rsv_3", 20110106),
        60.6557,
        epsilon = 1e-3
    );
}

#[test]
fn test_change_single_default_window() {
    let mut stock = stock_20days();
    let idx = 20110114;
    let rsv = get_val(&mut stock, "rsv", idx);
    let rsv_9 = get_val(&mut stock, "rsv_9", idx);
    let rsv_5 = get_val(&mut stock, "rsv_5", idx);
    assert_abs_diff_eq!(rsv, rsv_9, epsilon = 1e-8);
    assert!((rsv - rsv_5).abs() > 1e-6);
}

#[test]
fn test_column_kdj_default() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "kdjk", 20110104),
        60.5263,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "kdjd", 20110104),
        53.5087,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "kdjj", 20110104),
        74.5614,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_kdjk() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "kdjk_3", 20110104),
        60.5263,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "kdjk_3", 20110120),
        31.2133,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_kdjd() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "kdjd_3", 20110104),
        53.5087,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "kdjd_3", 20110120),
        43.1347,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_kdjj() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "kdjj_3", 20110104),
        74.5614,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "kdjj_3", 20110120),
        7.37,
        epsilon = 1e-2
    );
}

#[test]
fn test_z_kdj() {
    let mut stock = stock_90days();
    let mut df = stock.df().clone();
    for col in ["open", "high", "low", "close", "volume"] {
        let zcol = format!("{}_20_z", col);
        let _ = stock.get(&zcol).expect("zcol");
        let vals = stock
            .df()
            .column(&zcol)
            .expect("z")
            .as_materialized_series()
            .cast(&DataType::Float64)
            .expect("cast")
            .f64()
            .expect("f64")
            .into_iter()
            .map(|v| v.unwrap_or(0.0))
            .collect::<Vec<_>>();
        df.with_column(Series::new(col.into(), vals))
            .expect("replace");
    }
    stock = StockDataFrame::retype(df).expect("retype");
    assert_abs_diff_eq!(
        get_val(&mut stock, "kdjk", 20110104),
        66.666,
        epsilon = 1e-2
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "kdjd", 20110104),
        55.555,
        epsilon = 1e-2
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "kdjj", 20110104),
        88.888,
        epsilon = 1e-2
    );
}

#[test]
fn test_column_cross() {
    let mut stock = stock_30days();
    let _ = stock.get("kdjk_3_x_kdjd_3").expect("cross");
    let cross = stock
        .df()
        .column("kdjk_3_x_kdjd_3")
        .expect("col")
        .as_materialized_series()
        .bool()
        .expect("bool")
        .into_no_null_iter()
        .collect::<Vec<_>>();
    assert_eq!(cross.iter().filter(|v| **v).count(), 2);
    assert!(get_bool(&mut stock, "kdjk_3_x_kdjd_3", 20110114));
    assert!(get_bool(&mut stock, "kdjk_3_x_kdjd_3", 20110125));
}

#[test]
fn test_column_cross_up() {
    let mut stock = stock_30days();
    let _ = stock.get("kdjk_3_xu_kdjd_3").expect("cross up");
    let cross = stock
        .df()
        .column("kdjk_3_xu_kdjd_3")
        .expect("col")
        .as_materialized_series()
        .bool()
        .expect("bool")
        .into_no_null_iter()
        .collect::<Vec<_>>();
    assert_eq!(cross.iter().filter(|v| **v).count(), 1);
    assert!(get_bool(&mut stock, "kdjk_3_xu_kdjd_3", 20110125));
}

#[test]
fn test_column_cross_down() {
    let mut stock = stock_30days();
    let _ = stock.get("kdjk_3_xd_kdjd_3").expect("cross down");
    let cross = stock
        .df()
        .column("kdjk_3_xd_kdjd_3")
        .expect("col")
        .as_materialized_series()
        .bool()
        .expect("bool")
        .into_no_null_iter()
        .collect::<Vec<_>>();
    assert_eq!(cross.iter().filter(|v| **v).count(), 1);
    assert!(get_bool(&mut stock, "kdjk_3_xd_kdjd_3", 20110114));
}

#[test]
fn test_column_sma() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "open_2_sma", 20110104),
        12.42,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "open_2_sma", 20110105),
        12.56,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_smma() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "high_5_smma", 20110120),
        13.0394,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_ema() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_5_ema", 20110107),
        12.9026,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_5_ema", 20110110),
        12.9668,
        epsilon = 1e-3
    );
}

#[test]
fn test_ema_of_empty_df() {
    let mut s = StockDataFrame::retype(DataFrame::empty()).expect("retype");
    let ema = s.get("close_10_ema").expect("ema");
    assert_eq!(ema.len(), 0);
}

#[test]
fn test_column_macd() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "macd", 20110225),
        -0.0382,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "macds", 20110225),
        -0.0101,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "macdh", 20110225),
        -0.02805,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_macds() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "macds", 20110225),
        -0.0101,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_macdh() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "macdh", 20110225),
        -0.02805,
        epsilon = 1e-3
    );
}

#[test]
fn test_ppo() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(get_val(&mut stock, "ppo", 20110331), 1.1190, epsilon = 1e-3);
    assert_abs_diff_eq!(
        get_val(&mut stock, "ppos", 20110331),
        0.6840,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "ppoh", 20110331),
        0.4349,
        epsilon = 1e-3
    );
}

#[test]
fn test_eri() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "eribull", 20110104),
        0.070,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "eribear", 20110104),
        -0.309,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "eribull", 20110222),
        0.099,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "eribear", 20110222),
        -0.290,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "eribull_10", 20110222),
        0.092,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "eribear_10", 20110222),
        -0.297,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_mstd() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_3_mstd", 20110106),
        0.05033,
        epsilon = 1e-4
    );
}

#[test]
fn test_bollinger() {
    let mut stock = StockDataFrame::retype(filter_within(
        load_csv_df("tests/data/987654.csv"),
        20140930,
        20141211,
    ))
    .expect("retype");
    assert_abs_diff_eq!(
        get_val(&mut stock, "boll", 20141103),
        9.8035,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "boll_ub", 20141103),
        10.1310,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "boll_lb", 20141103),
        9.4759,
        epsilon = 1e-3
    );
}

#[test]
fn test_bollinger_with_window() {
    let mut stock = StockDataFrame::retype(filter_within(
        load_csv_df("tests/data/987654.csv"),
        20140930,
        20141211,
    ))
    .expect("retype");
    assert_abs_diff_eq!(
        get_val(&mut stock, "boll_20", 20141103),
        9.8035,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "boll_ub_20", 20141103),
        10.1310,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "boll_lb_20", 20141103),
        9.4759,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "boll_ub_5", 20141103),
        10.44107,
        epsilon = 1e-3
    );
}

#[test]
fn test_bollinger_empty() {
    let mut stock = StockDataFrame::retype(filter_within(
        load_csv_df("tests/data/987654.csv"),
        18800101,
        18900101,
    ))
    .expect("retype");
    let s = stock.get("boll_ub").expect("boll_ub");
    assert_eq!(s.len(), 0);
}

#[test]
fn test_column_mvar() {
    let mut stock = stock_20days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "open_3_mvar", 20110106),
        0.0292,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_parse_error() {
    let mut stock = stock_90days();
    let err1 = stock.get("foobarbaz").expect_err("should error");
    assert!(matches!(err1, StockStatsError::InvalidColumn(_)));
    let err2 = stock.get("close_1_foo_3_4").expect_err("should error");
    assert!(matches!(err2, StockStatsError::InvalidColumn(_)));
}

#[test]
fn test_parse_column_name_1() {
    assert_eq!(
        StockDataFrame::parse_column_name_for_test("amount_-5~-1_p"),
        ParsedColumnName::Three("amount".to_string(), "-5~-1".to_string(), "p".to_string())
    );
}

#[test]
fn test_parse_column_name_2() {
    assert_eq!(
        StockDataFrame::parse_column_name_for_test("open_+2~4_d"),
        ParsedColumnName::Three("open".to_string(), "+2~4".to_string(), "d".to_string())
    );
}

#[test]
fn test_parse_column_name_stacked() {
    assert_eq!(
        StockDataFrame::parse_column_name_for_test("open_-1_d_-1~-3_p"),
        ParsedColumnName::Three(
            "open_-1_d".to_string(),
            "-1~-3".to_string(),
            "p".to_string()
        )
    );
}

#[test]
fn test_parse_column_name_3() {
    assert_eq!(
        StockDataFrame::parse_column_name_for_test("close_-3,-1,+2_p"),
        ParsedColumnName::Three("close".to_string(), "-3,-1,+2".to_string(), "p".to_string())
    );
}

#[test]
fn test_parse_column_name_max() {
    assert_eq!(
        StockDataFrame::parse_column_name_for_test("close_-3,-1,+2_max"),
        ParsedColumnName::Three(
            "close".to_string(),
            "-3,-1,+2".to_string(),
            "max".to_string()
        )
    );
}

#[test]
fn test_parse_column_name_float() {
    assert_eq!(
        StockDataFrame::parse_column_name_for_test("close_12.32_le"),
        ParsedColumnName::Three("close".to_string(), "12.32".to_string(), "le".to_string())
    );
}

#[test]
fn test_parse_column_name_stacked_xu() {
    assert_eq!(
        StockDataFrame::parse_column_name_for_test("cr-ma2_xu_cr-ma1_20_c"),
        ParsedColumnName::Three(
            "cr-ma2_xu_cr-ma1".to_string(),
            "20".to_string(),
            "c".to_string()
        )
    );
}

#[test]
fn test_parse_column_name_rsv() {
    assert_eq!(
        StockDataFrame::parse_column_name_for_test("rsv_9"),
        ParsedColumnName::Two("rsv".to_string(), "9".to_string())
    );
}

#[test]
fn test_parse_column_name_no_match() {
    assert_eq!(
        StockDataFrame::parse_column_name_for_test("no match"),
        ParsedColumnName::NoMatch
    );
}

#[test]
fn test_to_int_split() {
    let shifts = StockDataFrame::to_ints_for_test("5,1,3, -2").expect("to ints");
    assert_eq!(shifts, vec![-2, 1, 3, 5]);
}

#[test]
fn test_to_int_continue() {
    let shifts = StockDataFrame::to_ints_for_test("3, -3~-1, 5").expect("to ints");
    assert_eq!(shifts, vec![-3, -2, -1, 3, 5]);
}

#[test]
fn test_to_int_dedup() {
    let shifts = StockDataFrame::to_ints_for_test("3, -3~-1, 5, -2~-1").expect("to ints");
    assert_eq!(shifts, vec![-3, -2, -1, 3, 5]);
}

#[test]
fn test_is_cross_columns() {
    assert!(StockDataFrame::is_cross_columns_for_test("a_x_b"));
    assert!(StockDataFrame::is_cross_columns_for_test("a_xu_b"));
    assert!(StockDataFrame::is_cross_columns_for_test("a_xd_b"));
    assert!(!StockDataFrame::is_cross_columns_for_test("a_xx_b"));
    assert!(!StockDataFrame::is_cross_columns_for_test("a_xa_b"));
    assert!(!StockDataFrame::is_cross_columns_for_test("a_x_"));
    assert!(!StockDataFrame::is_cross_columns_for_test("_xu_b"));
    assert!(!StockDataFrame::is_cross_columns_for_test("_xd_"));
}

#[test]
fn test_parse_cross_column() {
    let result = StockDataFrame::parse_cross_column_for_test("a_x_b").expect("parse");
    assert_eq!(result, ("a".to_string(), "x".to_string(), "b".to_string()));
}

#[test]
fn test_parse_cross_column_xu() {
    let result = StockDataFrame::parse_cross_column_for_test("a_xu_b").expect("parse");
    assert_eq!(result, ("a".to_string(), "xu".to_string(), "b".to_string()));
}

#[test]
fn test_get_log_ret() {
    let mut stock = stock_30days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "log-ret", 20110128),
        -0.010972,
        epsilon = 1e-4
    );
}

#[test]
fn test_rsv_nan_value() {
    let df = df![
        "date" => vec![1_i64, 2, 3, 4],
        "open" => vec![10.0, 10.0, 10.0, 10.0],
        "high" => vec![10.0, 10.0, 10.0, 10.0],
        "low" => vec![10.0, 10.0, 10.0, 10.0],
        "close" => vec![10.0, 10.0, 10.0, 10.0],
        "volume" => vec![100.0, 100.0, 100.0, 100.0],
    ]
    .expect("df");
    let mut stock = StockDataFrame::retype(df).expect("retype");
    assert_abs_diff_eq!(get_val_at(&mut stock, "rsv_9", 0), 0.0, epsilon = 1e-8);
}

#[test]
fn test_unwrap() {
    let mut stock = supor();
    let _ = stock.get("boll").expect("boll");
    assert_abs_diff_eq!(
        get_val(&mut stock, "boll", 20160817),
        39.6120,
        epsilon = 1e-3
    );
}

#[test]
fn test_get_rsi() {
    let mut s = supor();
    assert_abs_diff_eq!(get_val_at(&mut s, "rsi", 0), 50.0, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val_at(&mut s, "rsi_6", 0), 50.0, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut s, "rsi_6", 20160817), 71.3114, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "rsi_12", 20160817), 63.1125, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "rsi_24", 20160817), 61.3064, epsilon = 1e-3);
    assert_abs_diff_eq!(
        get_val(&mut s, "rsi", 20160817),
        get_val(&mut s, "rsi_14", 20160817),
        epsilon = 1e-8
    );
}

#[test]
fn test_get_stoch_rsi() {
    let mut stock = stock_90days();
    let idx = 20110331;
    assert_abs_diff_eq!(
        get_val(&mut stock, "stochrsi", idx),
        67.0955,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "stochrsi_6", idx),
        27.5693,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "stochrsi_14", idx),
        get_val(&mut stock, "stochrsi", idx),
        epsilon = 1e-8
    );
}

#[test]
fn test_get_wr() {
    let mut s = supor();
    let idx = 20160817;
    assert_abs_diff_eq!(get_val(&mut s, "wr_14", idx), -49.1620, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "wr_6", idx), -16.5322, epsilon = 1e-3);
    assert_abs_diff_eq!(
        get_val(&mut s, "wr", idx),
        get_val(&mut s, "wr_14", idx),
        epsilon = 1e-8
    );
}

#[test]
fn test_get_cci() {
    let mut stock = StockDataFrame::retype(filter_within(
        load_csv_df("tests/data/002032.csv"),
        20160701,
        20160831,
    ))
    .expect("retype");
    stock.drop_column("amount").expect("drop amount");
    assert_abs_diff_eq!(get_val(&mut stock, "cci", 20160817), 50.0, epsilon = 1e-2);
    assert_abs_diff_eq!(
        get_val(&mut stock, "cci_14", 20160817),
        50.0,
        epsilon = 1e-2
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "cci_14", 20160816),
        24.7987,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "cci_14", 20160815),
        -26.46,
        epsilon = 1e-2
    );
}

#[test]
fn test_get_atr() {
    let mut s = supor();
    assert_abs_diff_eq!(get_val(&mut s, "atr_14", 20160817), 1.3334, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "atr", 20160817), 1.3334, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "atr", 20160816), 1.3229, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "atr", 20160815), 1.2815, epsilon = 1e-3);
}

#[test]
fn test_get_sma_tr() {
    let mut s = supor();
    assert_abs_diff_eq!(
        get_val(&mut s, "tr_14_sma", 20160817),
        1.3321,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(get_val(&mut s, "tr_14_sma", 20160816), 1.37, epsilon = 1e-2);
    assert_abs_diff_eq!(get_val(&mut s, "tr_14_sma", 20160815), 1.47, epsilon = 1e-2);
}

#[test]
fn test_get_dma() {
    let mut s = supor();
    assert_abs_diff_eq!(get_val(&mut s, "dma", 20160817), 2.078, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "dma", 20160816), 2.15, epsilon = 2e-3);
    assert_abs_diff_eq!(get_val(&mut s, "dma", 20160815), 2.2743, epsilon = 1e-3);
    assert_abs_diff_eq!(
        get_val(&mut s, "close_10,50_dma", 20160817),
        2.078,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut s, "high_5,10_dma", 20160817),
        0.174,
        epsilon = 1e-3
    );
}

#[test]
fn test_pdm_ndm() {
    let mut c = stock_90days();
    assert_abs_diff_eq!(get_val(&mut c, "pdm_14", 20110104), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut c, "pdm_14", 20110331), 0.0842, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut c, "ndm_14", 20110104), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut c, "ndm_14", 20110331), 0.0432, epsilon = 1e-3);
}

#[test]
fn test_get_pdi() {
    let mut s = supor();
    assert_abs_diff_eq!(get_val(&mut s, "pdi", 20160817), 25.747, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "pdi", 20160816), 27.948, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "pdi", 20160815), 24.646, epsilon = 1e-3);
}

#[test]
fn test_get_mdi() {
    let mut s = supor();
    assert_abs_diff_eq!(get_val(&mut s, "ndi", 20160817), 16.195, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "ndi", 20160816), 17.579, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "ndi", 20160815), 19.542, epsilon = 1e-3);
}

#[test]
fn test_dx() {
    let mut s = supor();
    assert_abs_diff_eq!(get_val(&mut s, "dx", 20160817), 22.774, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "dx", 20160815), 11.550, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "dx", 20160812), 4.828, epsilon = 1e-3);
}

#[test]
fn test_adx() {
    let mut s = supor();
    assert_abs_diff_eq!(get_val(&mut s, "adx", 20160817), 15.535, epsilon = 1e-2);
    assert_abs_diff_eq!(get_val(&mut s, "adx", 20160816), 12.640, epsilon = 1e-2);
    assert_abs_diff_eq!(get_val(&mut s, "adx", 20160815), 8.586, epsilon = 1e-2);
}

#[test]
fn test_adxr() {
    let mut s = supor();
    assert_abs_diff_eq!(get_val(&mut s, "adxr", 20160817), 13.208, epsilon = 1e-2);
    assert_abs_diff_eq!(get_val(&mut s, "adxr", 20160816), 12.278, epsilon = 1e-2);
    assert_abs_diff_eq!(get_val(&mut s, "adxr", 20160815), 12.133, epsilon = 1e-2);
}

#[test]
fn test_trix_default() {
    let mut s = supor();
    assert_abs_diff_eq!(get_val(&mut s, "trix", 20160817), 0.1999, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "trix", 20160816), 0.2135, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "trix", 20160815), 0.24, epsilon = 1e-2);
    assert_abs_diff_eq!(
        get_val(&mut s, "close_12_trix", 20160815),
        0.24,
        epsilon = 1e-2
    );
    assert_abs_diff_eq!(
        get_val(&mut s, "high_12_trix", 20160815),
        0.235,
        epsilon = 1e-3
    );
}

#[test]
fn test_tema_default() {
    let mut s = supor();
    assert_abs_diff_eq!(
        get_val(&mut s, "tema", 20160817),
        get_val(&mut s, "close_5_tema", 20160817),
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(get_val(&mut s, "tema", 20160817), 40.2883, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "tema", 20160816), 39.6371, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "tema", 20160815), 39.3778, epsilon = 1e-3);
    assert_abs_diff_eq!(
        get_val(&mut s, "high_3_tema", 20160815),
        39.7315,
        epsilon = 1e-3
    );
}

#[test]
fn test_trix_ma() {
    let mut s = supor();
    assert_abs_diff_eq!(
        get_val(&mut s, "trix_9_sma", 20160817),
        0.34,
        epsilon = 1e-2
    );
    assert_abs_diff_eq!(
        get_val(&mut s, "trix_9_sma", 20160816),
        0.38,
        epsilon = 1e-2
    );
    assert_abs_diff_eq!(
        get_val(&mut s, "trix_9_sma", 20160815),
        0.4238,
        epsilon = 1e-3
    );
}

#[test]
fn test_vr_default() {
    let mut s = supor();
    assert_abs_diff_eq!(get_val(&mut s, "vr", 20160817), 153.1961, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "vr", 20160816), 171.6939, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "vr", 20160815), 178.7854, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "vr_26", 20160817), 153.1961, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "vr_26", 20160816), 171.6939, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut s, "vr_26", 20160815), 178.7854, epsilon = 1e-3);
}

#[test]
fn test_vr_ma() {
    let mut s = supor();
    assert_abs_diff_eq!(
        get_val(&mut s, "vr_6_sma", 20160817),
        182.7714,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut s, "vr_6_sma", 20160816),
        190.0970,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut s, "vr_6_sma", 20160815),
        197.5225,
        epsilon = 1e-3
    );
}

#[test]
fn test_mfi() {
    let mut stock = stock_90days();
    let first = 20110104;
    let last = 20110331;
    assert_abs_diff_eq!(get_val(&mut stock, "mfi", first), 0.5, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut stock, "mfi", last), 0.7144, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut stock, "mfi_3", first), 0.5, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut stock, "mfi_3", last), 0.7874, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut stock, "mfi_15", first), 0.5, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut stock, "mfi_15", last), 0.6733, epsilon = 1e-3);
}

#[test]
fn test_mfi_with_amount() {
    let mut s = supor();
    assert_abs_diff_eq!(get_val(&mut s, "mfi", 20160817), 0.48265, epsilon = 1e-3);
}

#[test]
fn test_ker() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(get_val(&mut stock, "ker", 20110104), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut stock, "ker", 20110105), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut stock, "ker", 20110117), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut stock, "ker", 20110118), 0.0357, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut stock, "ker", 20110210), 0.305, epsilon = 1e-3);
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_10_ker", 20110118),
        0.0357,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "high_5_ker", 20110210),
        0.399,
        epsilon = 1e-3
    );
}

#[test]
fn test_column_kama() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_10,2,30_kama", 20110331),
        13.6648,
        epsilon = 1e-3
    );
}

#[test]
fn test_kama_with_default_fast_slow() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_2_kama", 20110331),
        13.7326,
        epsilon = 1e-3
    );
}

#[test]
fn test_vwma() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "vwma", 20110330),
        13.312679,
        epsilon = 1e-3
    );
    let idx = 20110331;
    assert_abs_diff_eq!(get_val(&mut stock, "vwma", idx), 13.350941, epsilon = 1e-3);
    assert_abs_diff_eq!(
        get_val(&mut stock, "vwma_14", idx),
        get_val(&mut stock, "vwma", idx),
        epsilon = 1e-8
    );
    assert!((get_val(&mut stock, "vwma_7", idx) - get_val(&mut stock, "vwma", idx)).abs() > 1e-4);
}

#[test]
fn test_chop() {
    let mut stock = stock_90days();
    let idx = 20110330;
    assert_abs_diff_eq!(get_val(&mut stock, "chop", idx), 44.8926, epsilon = 1e-3);
    assert_abs_diff_eq!(
        get_val(&mut stock, "chop_14", idx),
        get_val(&mut stock, "chop", idx),
        epsilon = 1e-8
    );
    assert!((get_val(&mut stock, "chop_7", idx) - get_val(&mut stock, "chop", idx)).abs() > 1e-4);
}

#[test]
fn test_chop_flat_bars() {
    let df = df![
        "open" => vec![10.0_f64; 20],
        "high" => vec![10.0_f64; 20],
        "low" => vec![10.0_f64; 20],
        "close" => vec![10.0_f64; 20],
        "volume" => vec![100.0_f64; 20],
        "date" => (0_i64..20_i64).collect::<Vec<_>>(),
    ]
    .expect("df");
    let mut stock = StockDataFrame::retype(df).expect("retype");
    let _ = stock.get("chop").expect("chop");
    let col = stock
        .df()
        .column("chop")
        .expect("col")
        .as_materialized_series()
        .f64()
        .expect("f64");
    assert_eq!(col.len(), 20);
    for v in col.into_no_null_iter() {
        assert_eq!(v, 0.0);
        assert!(v.is_finite());
    }
}

#[test]
fn test_column_conflict() {
    let mut stock = stock_90days();
    let idx = 20110331;
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_26_ema", idx),
        13.2488,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(get_val(&mut stock, "macd", idx), 0.1482, epsilon = 1e-3);
}

#[test]
fn test_wave_trend() {
    let mut stock = stock_90days();
    let idx = 20110331;
    assert_abs_diff_eq!(get_val_at(&mut stock, "wt1", 0), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut stock, "wt1", idx), 38.9610, epsilon = 2e-2);
    assert_abs_diff_eq!(get_val_at(&mut stock, "wt2", 0), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut stock, "wt2", idx), 31.6997, epsilon = 2e-2);
    assert_abs_diff_eq!(
        get_val(&mut stock, "wt1_10,21", idx),
        38.9610,
        epsilon = 2e-2
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "wt2_10,21", idx),
        31.6997,
        epsilon = 2e-2
    );
}

#[test]
fn test_init_all() {
    let mut stock = stock_90days();
    for col in [
        "macd",
        "kdjj",
        "mfi",
        "boll",
        "adx",
        "cr-ma2",
        "supertrend_lb",
        "boll_lb",
        "ao",
        "cti",
        "ftr",
        "psl",
    ] {
        let _ = stock.get(col).expect("compute");
    }
    let columns = colnames(&stock);
    for name in [
        "macd",
        "kdjj",
        "mfi",
        "boll",
        "adx",
        "cr-ma2",
        "supertrend_lb",
        "boll_lb",
        "ao",
        "cti",
        "ftr",
        "psl",
    ] {
        assert!(columns.contains(&name.to_string()));
    }
}

#[test]
fn test_supertrend() {
    let mut stock = stock_90days();
    let idx = 20110302;
    assert_abs_diff_eq!(
        get_val(&mut stock, "supertrend", idx),
        13.3430,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "supertrend_ub", idx),
        13.3430,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "supertrend_lb", idx),
        12.2541,
        epsilon = 1e-3
    );
    let idx = 20110331;
    assert_abs_diff_eq!(
        get_val(&mut stock, "supertrend", idx),
        12.9021,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "supertrend_ub", idx),
        14.6457,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "supertrend_lb", idx),
        12.9021,
        epsilon = 1e-3
    );
}

#[test]
fn test_ao() {
    let mut stock = stock_90days();
    let idx = 20110302;
    assert_abs_diff_eq!(get_val(&mut stock, "ao", idx), -0.112, epsilon = 1e-3);
    assert_abs_diff_eq!(
        get_val(&mut stock, "ao_5,34", idx),
        get_val(&mut stock, "ao", idx),
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(get_val(&mut stock, "ao_5,10", idx), -0.071, epsilon = 1e-3);
}

#[test]
fn test_bop() {
    let mut stock = stock_30days();
    assert_abs_diff_eq!(get_val(&mut stock, "bop", 20110104), 0.5, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut stock, "bop", 20110106), -0.207, epsilon = 1e-3);
}

#[test]
fn test_cmo() {
    let mut stock = stock_30days();
    assert_abs_diff_eq!(get_val(&mut stock, "cmo", 20110104), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut stock, "cmo", 20110126), 7.023, epsilon = 1e-3);
    assert_abs_diff_eq!(
        get_val(&mut stock, "cmo", 20110127),
        -16.129,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "cmo_14", 20110126),
        7.023,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "cmo_5", 20110126),
        7.895,
        epsilon = 1e-3
    );
}

#[test]
fn test_drop_column_inplace() {
    let df = filter_within(load_csv_df("tests/data/002032.csv"), 20040817, 20040913);
    let mut stock = StockDataFrame::retype(df).expect("retype");
    stock.drop_column("open").expect("drop open");
    stock.drop_column("close").expect("drop close");
    let cols = colnames(&stock);
    assert!(cols.contains(&"high".to_string()));
    assert!(cols.contains(&"low".to_string()));
    assert!(!cols.contains(&"open".to_string()));
    assert!(!cols.contains(&"close".to_string()));
}

#[test]
fn test_drop_column() {
    let mut stock = StockDataFrame::retype(filter_within(
        load_csv_df("tests/data/002032.csv"),
        20040817,
        20040913,
    ))
    .expect("retype");
    let before = colnames(&stock);
    stock.drop_column("open").expect("drop open");
    stock.drop_column("close").expect("drop close");
    let after = colnames(&stock);
    assert!(after.contains(&"high".to_string()));
    assert!(after.contains(&"low".to_string()));
    assert!(!after.contains(&"open".to_string()));
    assert!(!after.contains(&"close".to_string()));
    assert!(before.contains(&"open".to_string()));
    assert!(before.contains(&"close".to_string()));
}

#[test]
fn test_drop_head_inplace() {
    let df = filter_within(load_csv_df("tests/data/002032.csv"), 20040817, 20040913);
    let stock = StockDataFrame::retype(df).expect("retype");
    let mut stock =
        StockDataFrame::retype(stock.df().slice(9, len_df(&stock) - 9)).expect("retype");
    assert_eq!(len_df(&stock), 11);
    assert_eq!(get_i64(&mut stock, "date", 20040830), 20040830);
}

#[test]
fn test_drop_head() {
    let df = filter_within(load_csv_df("tests/data/002032.csv"), 20040817, 20040913);
    let stock = StockDataFrame::retype(df).expect("retype");
    let ret = StockDataFrame::retype(stock.df().slice(9, len_df(&stock) - 9)).expect("retype");
    assert_eq!(len_df(&ret), 11);
    assert_eq!(len_df(&stock), 20);
}

#[test]
fn test_drop_tail_inplace() {
    let df = filter_within(load_csv_df("tests/data/002032.csv"), 20040817, 20040913);
    let stock = StockDataFrame::retype(df).expect("retype");
    let stock = StockDataFrame::retype(stock.df().slice(0, 11)).expect("retype");
    assert_eq!(len_df(&stock), 11);
    let idx = len_df(&stock) - 1;
    let last = stock
        .df()
        .column("date")
        .expect("date")
        .as_materialized_series()
        .cast(&DataType::Int64)
        .expect("cast")
        .i64()
        .expect("i64")
        .get(idx)
        .expect("last");
    assert_eq!(last, 20040831);
}

#[test]
fn test_drop_tail() {
    let df = filter_within(load_csv_df("tests/data/002032.csv"), 20040817, 20040913);
    let stock = StockDataFrame::retype(df).expect("retype");
    let ret = StockDataFrame::retype(stock.df().slice(0, 11)).expect("retype");
    assert_eq!(len_df(&ret), 11);
    assert_eq!(len_df(&stock), 20);
}

#[test]
fn test_aroon() {
    let mut stock = supor_50days();
    assert_abs_diff_eq!(get_val(&mut stock, "aroon", 20040924), 28.0, epsilon = 1e-8);
    assert_abs_diff_eq!(
        get_val(&mut stock, "aroon_25", 20040924),
        28.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "aroon_5", 20040924),
        40.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "aroon_5", 20041020),
        -80.0,
        epsilon = 1e-8
    );
}

#[test]
fn test_close_z() {
    let mut stock = supor_100days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_14_z", 20040817),
        -std::f64::consts::FRAC_1_SQRT_2,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_14_z", 20040915),
        2.005,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_14_z", 20041014),
        -2.014,
        epsilon = 1e-3
    );
}

#[test]
fn test_roc() {
    let mut stock = supor_100days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "high_5_roc", 20040817),
        0.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "high_5_roc", 20040915),
        5.912,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "high_5_roc", 20041014),
        5.009,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "high_5_roc", 20041220),
        -4.776,
        epsilon = 1e-3
    );
    let high = stock
        .df()
        .column("high")
        .expect("high")
        .as_materialized_series()
        .cast(&DataType::Float64)
        .expect("cast")
        .f64()
        .expect("f64")
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let roc = StockDataFrame::roc_for_test(&high, 5);
    let idx = get_idx(&stock, 20040915);
    assert_abs_diff_eq!(roc[idx], 5.912, epsilon = 1e-3);
}

#[test]
fn test_mad() {
    let mut stock = stock_30days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_5_mad", 20110104),
        0.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_5_mad", 20110114),
        0.146,
        epsilon = 1e-3
    );
}

#[test]
fn test_mad_raw() {
    let series = vec![10.0, 15.0, 15.0, 17.0, 18.0, 21.0];
    let res = StockDataFrame::mad_for_test(&series, 6);
    assert_abs_diff_eq!(res[5], 2.667, epsilon = 1e-3);
}

#[test]
fn test_sym_wma4() {
    let series = vec![4.0, 2.0, 2.0, 4.0, 8.0];
    let res = StockDataFrame::sym_wma4_for_test(&series);
    assert_abs_diff_eq!(res[0], 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(res[2], 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(res[3], 2.666, epsilon = 1e-3);
    assert_abs_diff_eq!(res[4], 3.666, epsilon = 1e-3);
}

#[test]
fn test_ichimoku() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "ichimoku", 20110228),
        0.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "ichimoku", 20110308),
        0.0275,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "ichimoku", 20110318),
        -0.0975,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "ichimoku_9,26,52", 20110228),
        0.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "ichimoku_9,26,52", 20110308),
        0.0275,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "ichimoku_9,26,52", 20110318),
        -0.0975,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "ichimoku_5,10,20", 20110228),
        -0.11,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "ichimoku_5,10,20", 20110308),
        0.0575,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "ichimoku_5,10,20", 20110318),
        0.0175,
        epsilon = 1e-3
    );
}

#[test]
fn test_linear_wma() {
    let series = vec![10.0, 15.0, 15.0, 17.0, 18.0, 21.0];
    let res = StockDataFrame::linear_wma_for_test(&series, 6);
    assert_abs_diff_eq!(res[0], 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(res[5], 17.571, epsilon = 1e-3);
}

#[test]
fn test_coppock() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "coppock", 20110117),
        0.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "coppock", 20110221),
        3.293,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "coppock", 20110324),
        -2.274,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "coppock_10,11,14", 20110221),
        3.293,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "coppock_5,10,15", 20110221),
        4.649,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "coppock_5,10,15", 20110324),
        -2.177,
        epsilon = 1e-3
    );
}

#[test]
fn test_linear_regression_raw() {
    let arr = vec![1.0, 5.0, 7.0, 2.0, 4.0, 3.0, 7.0, 9.0, 2.0];
    let lg = StockDataFrame::linear_reg_for_test(&arr, 5, false);
    assert_abs_diff_eq!(lg[3], 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(lg[8], 5.2, epsilon = 1e-8);
    let cr = StockDataFrame::linear_reg_for_test(&arr, 5, true);
    assert_abs_diff_eq!(cr[3], 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(cr[8], 0.108, epsilon = 1e-3);
}

#[test]
fn test_linear_regression() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_10_lrma", 20110114),
        0.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_10_lrma", 20110127),
        12.782,
        epsilon = 1e-3
    );
}

#[test]
fn test_cti() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(get_val(&mut stock, "cti", 20110118), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut stock, "cti", 20110131), -0.113, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut stock, "cti", 20110215), 0.369, epsilon = 1e-3);
    assert_abs_diff_eq!(
        get_val(&mut stock, "close_12_cti", 20110215),
        0.369,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "high_10_cti", 20110118),
        -0.006,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "high_10_cti", 20110131),
        -0.043,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "high_10_cti", 20110215),
        0.5006,
        epsilon = 1e-3
    );
}

#[test]
fn test_ftr() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(get_val(&mut stock, "ftr", 20110114), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut stock, "ftr", 20110128), -1.135, epsilon = 1e-3);
    assert_abs_diff_eq!(
        get_val(&mut stock, "ftr_9", 20110128),
        get_val(&mut stock, "ftr", 20110128),
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "ftr_15", 20110128),
        -1.005,
        epsilon = 1e-3
    );
}

#[test]
fn test_rvgi() {
    let mut stock = stock_30days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "rvgi", 20110128),
        get_val(&mut stock, "rvgi_14", 20110128),
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "rvgis", 20110128),
        get_val(&mut stock, "rvgis_14", 20110128),
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(get_val(&mut stock, "rvgi", 20110106), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut stock, "rvgi", 20110107), 0.257, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut stock, "rvgis", 20110111), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(
        get_val(&mut stock, "rvgis", 20110112),
        0.303,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "rvgi_10", 20110128),
        -0.056,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "rvgis_10", 20110128),
        -0.043,
        epsilon = 1e-3
    );
}

#[test]
fn test_change_group_window_defaults() {
    let mut stock = stock_90days();
    let i = 20110225;
    assert_abs_diff_eq!(
        get_val(&mut stock, "macd", i),
        get_val(&mut stock, "macd_12,26,9", i),
        epsilon = 1e-8
    );
}

#[test]
fn test_inertia() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "inertia", 20110209),
        0.0,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "inertia", 20110210),
        -0.024856,
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "inertia", 20110304),
        0.155576,
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "inertia_20,14", 20110304),
        0.155576,
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "inertia_20", 20110304),
        0.155576,
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "inertia_10", 20110209),
        0.011085,
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "inertia_10", 20110210),
        -0.014669,
        epsilon = 1e-4
    );
}

#[test]
fn test_kst() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(get_val(&mut stock, "kst", 20110117), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(
        get_val(&mut stock, "kst", 20110118),
        0.063442,
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "kst", 20110131),
        -2.519985,
        epsilon = 1e-4
    );
}

#[test]
fn test_pgo() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(
        get_val(&mut stock, "pgo", 20110117),
        -0.968845,
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "pgo", 20110214),
        1.292029,
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "pgo_14", 20110214),
        1.292029,
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "pgo_10", 20110117),
        -0.959768,
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "pgo_10", 20110214),
        1.214206,
        epsilon = 1e-4
    );
}

#[test]
fn test_psl() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(get_val(&mut stock, "psl", 20110118), 41.666, epsilon = 1e-3);
    assert_abs_diff_eq!(get_val(&mut stock, "psl", 20110127), 50.0, epsilon = 1e-3);
    assert_abs_diff_eq!(
        get_val(&mut stock, "psl_12", 20110118),
        41.666,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "psl_10", 20110118),
        50.0,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "psl_10", 20110131),
        60.0,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "high_12_psl", 20110118),
        41.666,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "high_12_psl", 20110127),
        41.666,
        epsilon = 1e-3
    );
}

#[test]
fn test_s_shift() {
    let stock = stock_90days();
    let close = stock
        .df()
        .column("close")
        .expect("close")
        .as_materialized_series()
        .cast(&DataType::Float64)
        .expect("cast")
        .f64()
        .expect("f64")
        .into_no_null_iter()
        .collect::<Vec<_>>();
    let close_n1 = StockDataFrame::s_shift_for_test(&close, -1);
    let close_0 = StockDataFrame::s_shift_for_test(&close, 0);
    let close_p1 = StockDataFrame::s_shift_for_test(&close, 1);
    let idx_20110104 = get_idx(&stock, 20110104);
    let idx_20110105 = get_idx(&stock, 20110105);
    let idx_20110106 = get_idx(&stock, 20110106);
    assert_abs_diff_eq!(close_n1[idx_20110104], 12.61, epsilon = 1e-3);
    assert_abs_diff_eq!(close_n1[idx_20110105], 12.61, epsilon = 1e-3);
    assert_abs_diff_eq!(close_n1[idx_20110106], 12.71, epsilon = 1e-3);
    assert_abs_diff_eq!(close_0[idx_20110106], 12.67, epsilon = 1e-3);
    let idx_20110330 = get_idx(&stock, 20110330);
    let idx_20110329 = get_idx(&stock, 20110329);
    let idx_20110331 = get_idx(&stock, 20110331);
    assert_abs_diff_eq!(close_0[idx_20110330], 13.85, epsilon = 1e-3);
    assert_abs_diff_eq!(close_p1[idx_20110329], 13.85, epsilon = 1e-3);
    assert_abs_diff_eq!(close_p1[idx_20110330], 13.62, epsilon = 1e-3);
    assert_abs_diff_eq!(close_p1[idx_20110331], 13.62, epsilon = 1e-3);
}

#[test]
fn test_s_shift_when_df_is_empty() {
    let close: Vec<f64> = Vec::new();
    let close_n1 = StockDataFrame::s_shift_for_test(&close, -1);
    let close_p1 = StockDataFrame::s_shift_for_test(&close, 1);
    assert_eq!(close_n1.len(), 0);
    assert_eq!(close_p1.len(), 0);
}

#[test]
fn test_pvo() {
    let mut stock = stock_90days();
    assert_abs_diff_eq!(get_val(&mut stock, "pvo", 20110331), 3.4708, epsilon = 1e-3);
    assert_abs_diff_eq!(
        get_val(&mut stock, "pvos", 20110331),
        -3.7464,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "pvoh", 20110331),
        7.2173,
        epsilon = 1e-3
    );
}

#[test]
fn test_qqe() {
    let mut stock = stock_90days();
    let _ = stock.get("qqe").expect("qqe");
    let _ = stock.get("qqe_14,5").expect("qqe dft");
    let _ = stock.get("qqe_10,4").expect("qqe custom");
    assert_abs_diff_eq!(get_val(&mut stock, "qqe", 20110125), 44.603, epsilon = 1e-3);
    assert_abs_diff_eq!(
        get_val(&mut stock, "qqel", 20110125),
        44.603,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(get_val(&mut stock, "qqes", 20110125), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut stock, "qqe", 20110223), 53.26, epsilon = 1e-2);
    assert_abs_diff_eq!(get_val(&mut stock, "qqel", 20110223), 0.0, epsilon = 1e-8);
    assert_abs_diff_eq!(get_val(&mut stock, "qqes", 20110223), 53.26, epsilon = 1e-2);
    assert_abs_diff_eq!(
        get_val(&mut stock, "qqe_14,5", 20110125),
        44.603,
        epsilon = 1e-3
    );
    assert_abs_diff_eq!(
        get_val(&mut stock, "qqe_10,4", 20110125),
        39.431,
        epsilon = 1e-3
    );
}
