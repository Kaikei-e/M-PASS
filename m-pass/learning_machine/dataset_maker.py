"""
データセットを作成する (XMLファイルのストリーミングパース版)。
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
        self.target_dir = target_dir

    def stream_xml_file(self, xml_file_path):
        """
        大きなXMLファイルを iterparse により逐次パースして、
        各レコードを辞書としてyieldします。
        iOS HealthKit のXMLの場合、タグは "Record" で、情報は属性として与えられます。
        """
        context = ET.iterparse(xml_file_path, events=("end",), tag="Record")
        for event, elem in context:
            record = {}
            # HealthKit XMLで一般的な属性（必要に応じて追加・調整してください）
            for key in ["type", "value", "unit", "sourceName", "device", "creationDate", "startDate", "endDate"]:
                record[key] = elem.attrib.get(key)
            yield record
            elem.clear()  # メモリ解放

    def make_dataset(self):
        """
        複数のXMLファイルをストリーミング処理し、DataFrameに変換します。
        以下の前処理を実施：
          - XMLから抽出した値でDataFrame作成
          - endDate を日時型に変換し、時系列でソート
          - value カラムを数値型に変換
        その上で、学習用（train）とテスト用（test）に分割します。
        """
        all_records = []
        for file_name in os.listdir(self.target_dir):
            if not file_name.lower().endswith(".xml"):
                continue
            xml_file_path = os.path.join(self.target_dir, file_name)
            print(f"[INFO] Processing file: {xml_file_path} ...")
            count = 0
            for record in self.stream_xml_file(xml_file_path):
                all_records.append(record)
                count += 1
            print(f"[INFO] Finished {xml_file_path}. Total records so far: {len(all_records)} (this file: {count})")
        
        df = pd.DataFrame(all_records)
        if df.empty:
            raise ValueError("生成されたDataFrameが空です。XMLファイルの内容を再確認してください。")
        
        # 前処理：endDate を日時型に変換し、時系列でソート
        df["endDate"] = pd.to_datetime(df["endDate"], errors="coerce")
        df.sort_values("endDate", inplace=True)
        
        # 前処理：value カラムを数値に変換
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        
        # 分割（例：80%を学習、残りをテスト）
        split_index = int(len(df) * 0.8)
        train_data = df.iloc[:split_index].copy()
        test_data = df.iloc[split_index:].copy()
        return test_data, train_data

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
