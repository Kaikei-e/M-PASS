#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET


def parse_and_split_xml(input_file, output_dir, output_prefix, split_func):
    """
    入力XMLファイルをパースし、Record要素をtype属性でグループ化して、
    各グループごとに出力ファイルを作成（既存の場合は追記）する。
    出力ファイルは以下の形式となる。

    <?xml version="1.0" encoding="utf-8"?>
    <HealthData>
       <{type}>
         <SourceName>...</SourceName>
         <SourceVersion>...</SourceVersion>
         <Device>...</Device>
         <Unit>...</Unit>
         <CreationDate>...</CreationDate>
         <StartDate>...</StartDate>
         <EndDate>...</EndDate>
         <Value>...</Value>
       </{type}>
       ...
    </HealthData>
    """
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing file: {input_file}")

    # XML全体を読み込む
    with open(input_file, "r", encoding="utf-8") as f:
        xml_data = f.read()
    print("XML file read successfully")

    # XMLをパース（Strictモードは False とする）
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        print("Error parsing XML file:", e)
        exit(1)

    print("XML file parsed successfully")

    # <Record> 要素のみ抽出し、type属性でグループ化
    records_by_type = {}
    for rec in root.findall("Record"):
        type_attr = rec.get("type")
        if not type_attr:
            continue
        # 分割判定関数(split_func)を利用（必ずTrue, type属性文字列を返す）
        should_split, new_filename = split_func(rec)
        if should_split:
            key = new_filename if new_filename else type_attr
            records_by_type.setdefault(key, []).append(rec)

    print("Records grouped by type successfully")

    # 各グループごとに出力ファイルを作成または追記
    for type_attr, recs in records_by_type.items():
        # 出力ファイル名は「{output_dir}/{type_attr}.xml」とする
        file_path = os.path.join(output_dir, f"{type_attr}.xml")

        # レコードを出力するフォーマットを作成するヘルパー関数
        def build_record_str(rec, tag):
            # main.go の書式に合わせて、各属性をタグとして出力
            src_name = rec.get("sourceName", "")
            src_version = rec.get("sourceVersion", "")
            device = rec.get("device", "")
            unit = rec.get("unit", "")
            creation = rec.get("creationDate", "")
            start_date = rec.get("startDate", "")
            end_date = rec.get("endDate", "")
            value = rec.get("value", "")
            return (
                f"<{tag}>"
                f"<SourceName>{src_name}</SourceName>"
                f"<SourceVersion>{src_version}</SourceVersion>"
                f"<Device>{device}</Device>"
                f"<Unit>{unit}</Unit>"
                f"<CreationDate>{creation}</CreationDate>"
                f"<StartDate>{start_date}</StartDate>"
                f"<EndDate>{end_date}</EndDate>"
                f"<Value>{value}</Value>"
                f"</{tag}>\n"
            )

        if not os.path.exists(file_path):
            # ファイルが存在しない場合は、新規作成
            with open(file_path, "w", encoding="utf-8") as f:
                # XMLヘッダーとルート開始タグ
                f.write('<?xml version="1.0" encoding="utf-8"?>\n')
                f.write("<HealthData>\n")
                # レコード出力（タグはtype属性の値を使用）
                for rec in recs:
                    f.write(build_record_str(rec, type_attr))
                # ルート終了タグ
                f.write("</HealthData>\n")
            print("New output file created and records written:", file_path)
        else:
            # 既存のファイルへ追記（まずファイル内の </HealthData> タグを除去）
            with open(file_path, "r+", encoding="utf-8") as f:
                content = f.read()
                closing_tag = "</HealthData>\n"
                idx = content.rfind(closing_tag)
                if idx == -1:
                    print("Error: closing tag not found in file:", file_path)
                    continue
                # closing tag までの内容を保持
                new_content = content[:idx]
                f.seek(0)
                f.write(new_content)
                # 追記するレコードを出力
                for rec in recs:
                    f.write(build_record_str(rec, type_attr))
                # ルート終了タグを再度書き込む
                f.write(closing_tag)
                f.truncate()
            print("Appended records to existing file successfully:", file_path)

    print("Done!")


def split_by_record_element(rec):
    """
    Record要素の分割判定関数。
    type属性をそのままグループ名として返す。
    """
    t = rec.get("type", "")
    return (True, t)
