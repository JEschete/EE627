from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


BASE_DIR = Path(__file__).resolve().parent


def read_series(csv_name: str) -> pd.Series:
	csv_path = BASE_DIR / csv_name
	values = pd.read_csv(csv_path, header=None).iloc[:, 0]
	series = pd.Series(values, name=csv_name.replace(".csv", ""))
	return pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)


def adf_test(series: pd.Series) -> dict:
	result = adfuller(series, autolag="AIC")
	stat = float(result[0])
	pvalue = float(result[1])
	critical_values = result[4] if len(result) > 4 else result[2]
	return {
		"stat": stat,
		"pvalue": pvalue,
		"critical_values": critical_values,
		"is_stationary": pvalue < 0.05,
	}


def choose_d(series: pd.Series, max_d: int = 2) -> tuple[int, dict]:
	for d in range(max_d + 1):
		test_series = series if d == 0 else series.diff(d).dropna()
		result = adf_test(test_series)
		if result["is_stationary"]:
			return d, result
	final_series = series.diff(max_d).dropna()
	return max_d, adf_test(final_series)


def fit_best_arima(series: pd.Series, p_max: int = 4, d_max: int = 2, q_max: int = 4) -> dict:
	best = None
	best_aic = np.inf

	chosen_d, adf_result = choose_d(series, max_d=d_max)

	for p in range(p_max + 1):
		for q in range(q_max + 1):
			if p == 0 and q == 0:
				continue
			order = (p, chosen_d, q)
			try:
				model = ARIMA(series, order=order)
				fit = model.fit()
				if np.isfinite(fit.aic) and fit.aic < best_aic:
					best_aic = fit.aic
					best = {
						"order": order,
						"aic": fit.aic,
						"bic": fit.bic,
						"fit": fit,
					}
			except Exception:
				continue

	if best is None:
		raise RuntimeError("No ARIMA model converged. Try reducing model order bounds.")

	best["adf"] = adf_result
	return best


def summarize_model(problem_label: str, csv_name: str, result: dict) -> str:
	order = result["order"]
	adf = result["adf"]

	lines = [
		f"{problem_label} ({csv_name})",
		"=" * 60,
		f"Selected model (lowest AIC): ARIMA{order}",
		f"AIC: {result['aic']:.4f}",
		f"BIC: {result['bic']:.4f}",
		"",
		"Stationarity check used to choose differencing order d:",
		f"ADF statistic: {adf['stat']:.6f}",
		f"ADF p-value: {adf['pvalue']:.6f}",
		f"Stationary at 5%: {adf['is_stationary']}",
		"Critical values:",
	]

	for key, value in adf["critical_values"].items():
		lines.append(f"  {key}: {value:.6f}")

	lines.extend(
		[
			"",
			"Model coefficients:",
			result["fit"].params.to_string(),
		]
	)

	return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
	path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
	q2_series = read_series("EE627A_HW3_Q2.csv")
	q3_series = read_series("EE627A_HW3_Q3.csv")

	q2_result = fit_best_arima(q2_series)
	q3_result = fit_best_arima(q3_series)

	q2_text = summarize_model("Problem 2", "EE627A_HW3_Q2.csv", q2_result)
	q3_text = summarize_model("Problem 3", "EE627A_HW3_Q3.csv", q3_result)

	write_text(BASE_DIR / "HW3_Q2_output.txt", q2_text)
	write_text(BASE_DIR / "HW3_Q3_output.txt", q3_text)

	summary_lines = [
		"Homework 3 outputs generated:",
		"- HW3_Q2_output.txt",
		"- HW3_Q3_output.txt",
		"",
		f"Q2 best model: ARIMA{q2_result['order']} (AIC={q2_result['aic']:.4f})",
		f"Q3 best model: ARIMA{q3_result['order']} (AIC={q3_result['aic']:.4f})",
	]
	print("\n".join(summary_lines))


if __name__ == "__main__":
	main()
