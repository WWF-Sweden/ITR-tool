"""Synthetic private values must not appear in automatic diagnostics."""
import contextlib
import io
from types import SimpleNamespace
import traceback
import unittest
from unittest.mock import patch

import pandas as pd
import requests

from ITR.configs import ColumnsConfig
from ITR.data.csv import CSVProvider
from ITR.data.excel import ExcelProvider
from ITR.data.inmemory import InMemoryProvider
from ITR.data.sbti import SBTi
from ITR.interfaces import EScope, ETimeFrames
from ITR.portfolio_aggregation import PortfolioAggregation
from ITR.temperature_score import TemperatureScore
from ITR.utils import dataframe_to_portfolio


PRIVATE_NAME = "PRIVATE_FUND_82f73"
PRIVATE_ID = "PRIVATE_COMPANY_4ab92"
PRIVATE_TARGET = "PRIVATE_TARGET_2da41"
PRIVATE_VALUE = "PRIVATE_VALUE_93e21"


class SafeDiagnosticsTests(unittest.TestCase):
    def assert_private_values_absent(self, text):
        for value in (PRIVATE_NAME, PRIVATE_ID, PRIVATE_TARGET, PRIVATE_VALUE):
            self.assertNotIn(value, text)

    def target_frame(self, **changes):
        row = dict(company_id=PRIVATE_ID, company_name=PRIVATE_NAME,
                   target_ids=[PRIVATE_TARGET], target_type="absolute", scope="S1",
                   base_year=2020, end_year=2035, start_year=2020,
                   statement_date=2025, achieved_reduction=0,
                   coverage_s1=1.0, reduction_ambition=0.5, base_year_ghg_s1=100.0)
        row.update(changes)
        return pd.DataFrame([row])

    def provider(self, cls):
        provider = cls.__new__(cls)
        provider.c = ColumnsConfig
        provider.data = {"fundamental_data": pd.DataFrame({
            "company_id": [PRIVATE_ID], "company_name": [PRIVATE_NAME],
        })}
        provider.invalid_targets = []
        return provider

    def test_invalid_target_logs_only_field_names_for_all_providers(self):
        for cls in (ExcelProvider, InMemoryProvider, CSVProvider):
            with self.subTest(provider=cls.__name__):
                provider = self.provider(cls)
                frame = self.target_frame(scope=PRIVATE_VALUE)
                with self.assertLogs(cls.__module__, level="WARNING") as logs:
                    result = provider._target_df_to_model(frame)
                self.assertEqual(result, [])
                self.assertIn("scope", " ".join(logs.output))
                self.assert_private_values_absent(" ".join(logs.output))
                if cls is ExcelProvider:
                    self.assertEqual(provider.invalid_targets[0]["company_id"], PRIVATE_ID)
                    self.assertEqual(provider.invalid_targets[0]["company_name"], PRIVATE_NAME)
                    self.assertEqual(provider.invalid_targets[0]["target_ids"], PRIVATE_TARGET)

    def test_valid_target_rows_still_load_without_warnings(self):
        for cls in (ExcelProvider, InMemoryProvider, CSVProvider):
            with self.subTest(provider=cls.__name__):
                with self.assertNoLogs(cls.__module__, level="WARNING"):
                    result = self.provider(cls)._target_df_to_model(
                        self.target_frame(statement_date="2025-01-01")
                    )
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].company_id, PRIVATE_ID)
                self.assertEqual(result[0].target_ids, [PRIVATE_TARGET])

    def test_year_conversion_error_and_traceback_do_not_echo_cell_value(self):
        for cls in (ExcelProvider, InMemoryProvider):
            with self.subTest(provider=cls.__name__):
                try:
                    self.provider(cls)._target_df_to_model(self.target_frame(end_year=PRIVATE_VALUE))
                except ValueError as error:
                    text = "".join(traceback.format_exception(error))
                    self.assertIn("Target conversion failed", text)
                    self.assert_private_values_absent(text)
                else:
                    self.fail("Invalid year must still fail conversion")

    def test_company_validation_error_is_safe_for_all_providers(self):
        frame = pd.DataFrame({"company_id": [PRIVATE_ID], "company_name": [PRIVATE_NAME],
                              "isic": ["A"], "ghg_s1": [PRIVATE_VALUE]})
        for cls in (ExcelProvider, InMemoryProvider, CSVProvider):
            with self.subTest(provider=cls.__name__):
                provider = self.provider(cls)
                if cls is ExcelProvider:
                    provider.data["fundamental_data"] = frame.copy()
                elif cls is CSVProvider:
                    provider.data = frame.copy()
                else:
                    provider.data_fundamental = frame.copy()
                try:
                    provider.get_company_data([PRIVATE_ID])
                except ValueError as error:
                    text = "".join(traceback.format_exception(error))
                    self.assertIn("ghg_s1", text)
                    self.assert_private_values_absent(text)
                else:
                    self.fail("Invalid company input must fail validation")

    def test_portfolio_validation_error_does_not_echo_input(self):
        frame = pd.DataFrame({"company_id": [PRIVATE_ID], "company_name": [PRIVATE_NAME],
                              "investment_value": [PRIVATE_VALUE], "engagement_target": [False]})
        try:
            dataframe_to_portfolio(frame)
        except ValueError as error:
            text = "".join(traceback.format_exception(error))
            self.assertIn("investment_value", text)
            self.assert_private_values_absent(text)
        else:
            self.fail("Invalid investment value must fail validation")

    def test_missing_aggregation_values_report_counts_not_company_names(self):
        aggregation = PortfolioAggregation()
        for field in ("ghg_s1", "company_revenue"):
            frame = pd.DataFrame({"company_name": [PRIVATE_NAME], field: [None]})
            with self.assertRaises(ValueError) as caught:
                aggregation._check_column(frame, field)
            text = str(caught.exception)
            self.assertIn(field, text)
            self.assertIn("1", text)
            self.assert_private_values_absent(text)

    def test_duplicate_company_warning_does_not_list_company_names(self):
        score = TemperatureScore(time_frames=[ETimeFrames.MID], scopes=[EScope.S1S2])
        frame = pd.DataFrame({"company_name": [PRIVATE_NAME, PRIVATE_NAME],
                              "company_id": [PRIVATE_ID, PRIVATE_ID],
                              "time_frame": [ETimeFrames.MID] * 2,
                              "scope": [EScope.S1S2] * 2, "investment_value": [100.0, 100.0],
                              "temperature_score": [2.0, 3.0], "temperature_results": [0.0, 0.0]})
        with self.assertLogs(level="WARNING") as logs:
            result = score._get_score_aggregation(frame, ETimeFrames.MID, EScope.S1S2)
        self.assertAlmostEqual(result.all.score, 2.5)
        self.assert_private_values_absent(" ".join(logs.output))

    def test_score_type_error_uses_same_fallback_without_echoing_inputs(self):
        score = TemperatureScore(time_frames=[ETimeFrames.MID], scopes=[EScope.S1S2])
        target = pd.Series(dict(to_calculate=True, reduction_ambition=0.5,
                                target_type="absolute", param=PRIVATE_VALUE,
                                annual_reduction_rate=0.1, intercept=1.0, sbti_validated=False))
        with self.assertLogs("ITR.temperature_score", level="WARNING") as logs:
            result, _ = score.get_score(target)
        self.assertEqual(result, score.default_score)
        self.assert_private_values_absent(" ".join(logs.output))

    def test_zero_denominator_warning_does_not_echo_target_id(self):
        score = TemperatureScore(time_frames=[ETimeFrames.MID], scopes=[EScope.S1S2])
        rows = [dict(company_id=PRIVATE_ID, time_frame=ETimeFrames.MID, scope=scope,
                     to_calculate=True, temperature_score=2.0, temperature_results=0.0,
                     ghg_s1=100.0, ghg_s2=-100.0, target_type="absolute",
                     target_ids=[PRIVATE_TARGET]) for scope in (EScope.S1, EScope.S2, EScope.S1S2)]
        with self.assertLogs("ITR.temperature_score", level="WARNING") as logs:
            score._calculate_s1s2_score(pd.DataFrame(rows))
        self.assert_private_values_absent(" ".join(logs.output))

    def test_cta_errors_do_not_echo_unknown_uploaded_status(self):
        reader = SBTi.__new__(SBTi)
        from ITR.configs import PortfolioCoverageTVPConfig
        reader.c = PortfolioCoverageTVPConfig
        frame = pd.DataFrame({"company_name": [PRIVATE_NAME], "isin": [PRIVATE_ID],
                              "lei": [PRIVATE_ID], "near_term_status": [PRIVATE_VALUE]})
        with self.assertRaisesRegex(ValueError, "1 unknown near-term status") as caught:
            reader._normalize_cta_file(frame)
        self.assert_private_values_absent(str(caught.exception))

    def test_cta_download_errors_and_paths_are_not_printed(self):
        reader = SBTi.__new__(SBTi)
        reader.c = SimpleNamespace(FILE_TARGETS=PRIVATE_NAME, SKIP_CTA_FILE_IF_EXISTS=False)
        output = io.StringIO()
        with patch.object(reader, "_check_if_cta_file_exists", side_effect=[False, True]), \
                patch.object(reader, "_fetch_and_save_cta_file", side_effect=requests.RequestException(PRIVATE_VALUE)), \
                contextlib.redirect_stdout(output):
            reader._download_cta_file()
        self.assertIn("Network error", output.getvalue())
        self.assert_private_values_absent(output.getvalue())

    def test_excel_category_conversion_does_not_print_invalid_value(self):
        target = self.target_frame(scope="S3", s3_category=PRIVATE_VALUE)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            target.to_excel(writer, sheet_name="target_data", index=False)
            pd.DataFrame({"company_id": [PRIVATE_ID], "company_name": [PRIVATE_NAME]}).to_excel(
                writer, sheet_name="fundamental_data", index=False)
        buffer.seek(0)
        output = io.StringIO()
        with self.assertLogs("ITR.data.excel", level="WARNING") as logs, contextlib.redirect_stdout(output):
            ExcelProvider(buffer)
        self.assert_private_values_absent(" ".join(logs.output) + output.getvalue())


if __name__ == "__main__":
    unittest.main()
