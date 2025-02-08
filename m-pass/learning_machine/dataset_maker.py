"""
データセットを作成する (データチェック機能強化版 - レポート形式ログ出力版 - 数値変換失敗時文字列保持版)。
"""

import os
import pandas as pd
from lxml import etree as ET
import numpy as np
import re
from sklearn.model_selection import (
    train_test_split,
)  # train_test_split のインポートは削除 (無作為分割は行わないため)


class DatasetMaker:
    def __init__(self, target_dir: str):
        self.list_of_data = []
        self.test_data = None
        self.train_data = None
        self.dataset = None
        self.target_dir = target_dir

    def make_dataset(self):
        """
        データセットを作成し、レポート形式でログを出力する (数値変換失敗時文字列保持版)。
        データセットを学習用とテスト用に時系列分割する。
        """
        if self.target_dir is None:
            raise ValueError("target_dir is None")

        dataset_list = []
        total_invalid_value_count = 0
        total_nan_value_count = 0
        total_inf_value_count = 0
        total_file_read_error_count = 0
        total_xml_parse_error_count = 0
        total_skipped_record_count = 0

        file_process_reports = []  # ファイルごとの処理レポートを格納するリスト

        # target_dir 内の XML ファイルを処理 (変更なし)
        for file in os.listdir(self.target_dir):
            if file.endswith(".xml"):
                file_path = os.path.join(self.target_dir, file)
                file_report = {
                    "file_path": file_path,
                    "status": "success",
                }  # ファイルレポート初期化
                file_process_reports.append(
                    file_report
                )  # ファイルレポートをリストに追加

                try:
                    (
                        df,
                        file_invalid_value_count,
                        file_nan_value_count,
                        file_inf_value_count,
                        file_skipped_record_count,
                        file_xml_parse_error_count,
                    ) = self.xml_to_dataframe_v2(file_path)
                    file_report["dataframe_shape"] = str(
                        df.shape
                    )  # DataFrame形状をレポートに追加
                    if not df.empty:
                        dataset_list.append(
                            df
                        )  # 作成した DataFrame を dataset_list に追加
                    else:
                        file_report["status"] = (
                            "warning"  # 空DataFrameの場合は警告ステータス
                        )
                        file_report["message"] = (
                            "DataFrame is empty after processing."  # 警告メッセージをレポートに追加
                        )

                    file_report["invalid_value_count"] = str(
                        file_invalid_value_count
                    )  # ファイルごとのデータチェック結果をレポートに追加
                    file_report["nan_value_count"] = str(file_nan_value_count)
                    file_report["inf_value_count"] = str(file_inf_value_count)
                    file_report["skipped_record_count"] = str(file_skipped_record_count)
                    file_report["xml_parse_error_count"] = str(
                        file_xml_parse_error_count
                    )

                    total_invalid_value_count += (
                        file_invalid_value_count  # 全体の集計に加算
                    )
                    total_nan_value_count += file_nan_value_count
                    total_inf_value_count += file_inf_value_count
                    total_skipped_record_count += file_skipped_record_count
                    total_xml_parse_error_count += file_xml_parse_error_count

                except Exception as e:
                    file_report["status"] = "error"  # エラー発生時はエラーステータス
                    file_report["message"] = str(e)  # エラーメッセージをレポートに追加
                    total_file_read_error_count += 1
                    continue

        if not dataset_list:
            raise ValueError(
                "No valid XML files found in target_dir. Dataset will be empty."
            )

        dataset_original = pd.concat(
            dataset_list, ignore_index=True
        )  # データセットを結合

        # データセットを endDate でソート (時系列順に並び替え)
        dataset_original = dataset_original.sort_values(
            by="endDate"
        )  # ★ 追記: endDate でソート

        # データセットを学習用とテスト用に時系列分割 (先頭から80%を学習、残り20%をテスト)
        train_size = 0.8  # 学習データの割合
        split_index = int(len(dataset_original) * train_size)  # 分割インデックスを計算
        train_data_original = dataset_original[
            :split_index
        ]  # 先頭から分割インデックスまでを学習データ
        test_data_original = dataset_original[
            split_index:
        ]  # 分割インデックス以降をテストデータ

        print("Dataset creation completed.")
        print("Data Check Report:")  # データチェックレポートを出力 (変更なし)
        print(f"  - Files with read errors: {total_file_read_error_count}")
        print(f"  - Files with XML parse errors: {total_xml_parse_error_count}")
        print(
            f"  - Records with invalid value format (not converted to number): {total_invalid_value_count}"
        )  # ★ 変更: "数値に変換できなかった値の数" と明記
        print(f"  - Records with NaN values: {total_nan_value_count}")
        print(f"  - Records with Inf values: {total_inf_value_count}")
        print(
            f"  - Records skipped due to invalid value (NaN or Inf): {total_skipped_record_count}"
        )

        print("\nFile Processing Report:")  # ファイル処理レポートを出力 (変更なし)
        for report in file_process_reports:
            status_str = f"[{report['status'].upper()}]"  # ステータスを大文字で表示
            message_str = (
                f"- Message: {report['message']}" if "message" in report else ""
            )  # メッセージがあれば表示
            dataframe_shape_str = (
                f"- DataFrame shape: {report['dataframe_shape']}"
                if "dataframe_shape" in report
                else ""
            )  # DataFrame形状があれば表示
            invalid_value_str = (
                f"- Invalid values: {report['invalid_value_count']}"
                if "invalid_value_count" in report
                else ""
            )  # 不正値数があれば表示
            nan_value_str = (
                f"- NaN values: {report['nan_value_count']}"
                if "nan_value_count" in report
                else ""
            )  # NaN値数があれば表示
            inf_value_str = (
                f"- Inf values: {report['inf_value_count']}"
                if "inf_value_str" in report
                else ""
            )  # Inf値数があれば表示
            skipped_record_str = (
                f"- Skipped records: {report['skipped_record_count']}"
                if "skipped_record_count" in report
                else ""
            )  # スキップレコード数があれば表示
            xml_parse_error_str = (
                f"- XML parse errors: {report['xml_parse_error_count']}"
                if "xml_parse_error_count" in report
                else ""
            )  # XMLパースエラー数があれば表示

            print(
                f"  - File: {report['file_path']} {status_str} {dataframe_shape_str} {message_str} {invalid_value_str} {nan_value_str} {inf_value_str} {skipped_record_str} {xml_parse_error_str}"
            )

        self.dataset = (
            dataset_original  # データセット全体を self.dataset に格納 (変更なし)
        )
        self.test_data = (
            test_data_original  # テストデータセットを self.test_data に格納 (変更なし)
        )
        self.train_data = (
            train_data_original  # 学習データセットを self.train_data に格納 (変更なし)
        )
        return (
            test_data_original,
            train_data_original,
        )  # 分割されたデータセットを返す (変更なし)

    def xml_to_dataframe_v2(self, xml_file_path):
        """
        XMLファイルをpandas DataFrameに変換し、データチェックを行う関数 (レポート形式ログ出力版 - 数値変換失敗時文字列保持版)。
        """
        invalid_value_count = 0
        nan_value_count = 0
        inf_value_count = 0
        skipped_record_count = 0
        xml_parse_error_count_file = 0

        parser = ET.XMLParser(huge_tree=True)
        try:
            tree = ET.parse(xml_file_path, parser)
            root = tree.getroot()
        except ET.ParseError as e:
            # 詳細ログ削除: print(f"DEBUG: Error parsing XML file: {xml_file_path}, error: {e}")
            return pd.DataFrame(), 0, 0, 0, 0, 1

        data = []
        for index, record in enumerate(root.findall("Record")):
            # 詳細ログ削除: print(f"DEBUG: Processing Record index: {index} in file: {xml_file_path}")
            record_type = record.get("type")
            value_str = record.get("value")
            unit = record.get("unit")
            creation_date = record.get("creationDate")
            start_date = record.get("startDate")
            end_date = record.get("endDate")
            source_name = record.get("sourceName")
            device_info = record.get("device")

            # 詳細ログ削除: print(f"DEBUG: Record attributes - type: {record_type}, value: {value_str}, unit: {unit}, creationDate: {creation_date}, startDate: {start_date}, endDate: {endDate}, sourceName: {source_name}, device: {device_info}")

            value_numeric = np.nan
            try:
                unit_pattern = r"([\d\.]+)\s*([a-zA-Z/·]+)$"
                match = re.match(unit_pattern, value_str)
                if match:
                    value_numeric = float(match.group(1))
                    unit = match.group(2)
                else:
                    value_numeric = float(value_str)

                if np.isnan(value_numeric):
                    nan_value_count += 1
                    value_numeric = np.nan
                elif np.isinf(value_numeric):
                    inf_value_count += 1
                    value_numeric = np.inf

            except (ValueError, TypeError):
                # 詳細ログ削除: print(f"DEBUG: Warning: Invalid value '{value_str}' found in file: {xml_file_path}, record type: {record_type}, record index: {index}. Value will be kept as string.")
                invalid_value_count += 1
                value_numeric = value_str  # ★ 変更: 数値変換失敗時は value_str をそのまま value_numeric に代入 (文字列として保持)

            data.append(
                [
                    record_type,
                    value_numeric,
                    unit,
                    creation_date,
                    start_date,
                    end_date,
                    source_name,
                    device_info,
                ]
            )

        df = pd.DataFrame(
            data,
            columns=[
                "type",
                "value",
                "unit",
                "creationDate",
                "startDate",
                "endDate",
                "sourceName",
                "device",
            ],
        )
        # 詳細ログ削除: print(f"DEBUG: DataFrame created from {xml_file_path}, shape: {df.shape}")
        return (
            df,
            invalid_value_count,
            nan_value_count,
            inf_value_count,
            skipped_record_count,
            xml_parse_error_count_file,
        )


# デバッグ用にしか使わない (変更なし)
def export_dataframe_to_xml_stream(df: pd.DataFrame, output_file_path: str):
    """
    DataFrame を XML 形式でストリーム出力する関数 (lxml.etree 使用)
    """
    root = ET.Element("root")
    tree = ET.ElementTree(root)

    for index, row in df.iterrows():
        record_element = ET.SubElement(root, "record")
        for column, value in row.items():
            element = ET.SubElement(record_element, str(column))
            if column == "xml_data":
                try:
                    xml_element = ET.fromstring(value.encode("utf-8"))
                    element.append(xml_element)
                except ET.ParseError as e:
                    print(f"Error parsing XML string: {e}")
                    element.text = str(value)
            else:
                element.text = str(value)

    with open(output_file_path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True, pretty_print=True)
