package main

import (
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// HealthData は XML ファイルのルート要素を表す構造体
type HealthData struct {
	XMLName xml.Name `xml:"HealthData"` // ルート要素のタグ名を指定
	Records []Record `xml:"Record"`
}

// Record は XML ファイル内の Record 要素を表す構造体
type Record struct {
	Type          string `xml:"type,attr"`
	Value         string `xml:"value,attr"`
	SourceName    string `xml:"sourceName,attr"`
	SourceVersion string `xml:"sourceVersion,attr"`
	Device        string `xml:"device,attr,omitempty"`
	Unit          string `xml:"unit,attr,omitempty"`
	CreationDate  string `xml:"creationDate,attr,omitempty"`
	StartDate     string `xml:"startDate,attr,omitempty"`
	EndDate       string `xml:"endDate,attr,omitempty"`
	// MetadataEntry や InstantaneousBeatsPerMinute などの子要素も必要に応じて追加
	MetadataEntries []MetadataEntry `xml:"MetadataEntry,omitempty"`
}

type MetadataEntry struct {
	Key   string `xml:"key,attr"`
	Value string `xml:"value,attr"`
}

func main() {
	if len(os.Args) < 3 { // 引数は2つ必要 (XMLファイルパスと出力ディレクトリ)
		fmt.Println("Usage: go run main.go <xml_file_path> <output_directory>")
		os.Exit(1)
	}

	xmlPath := os.Args[1]
	outputDir := os.Args[2]

	xmlFile, err := os.Open(xmlPath)
	if err != nil {
		fmt.Println("Error opening XML file:", err)
		os.Exit(1)
	}
	defer xmlFile.Close()

	xmlData, err := io.ReadAll(xmlFile)
	if err != nil {
		fmt.Println("Error reading XML file:", err)
		os.Exit(1)
	}

	var healthData HealthData
	// XMLをHealthData構造体にデコード
	decoder := xml.NewDecoder(bytes.NewReader(xmlData))
	decoder.Strict = false // 厳密なXML構文チェックを無効にする
	decoder.CharsetReader = func(charset string, input io.Reader) (io.Reader, error) {
		return input, nil // 文字コード変換は行わない
	}

	err = decoder.Decode(&healthData)
	if err != nil {
		fmt.Println("Error decoding XML:", err) // XMLデコード時のエラー
		os.Exit(1)
	}

	// type 属性ごとに Record をグループ化
	recordsByType := make(map[string][]Record)
	for _, record := range healthData.Records {
		recordsByType[record.Type] = append(recordsByType[record.Type], record)
	}

	// 出力ディレクトリを作成 (存在しない場合)
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		fmt.Println("Error creating output directory:", err)
		os.Exit(1)
	}

	// type ごとにファイルを作成・追記
	for typeAttr, records := range recordsByType {
		// ファイルパスを作成
		filePath := filepath.Join(outputDir, typeAttr+".xml")

		// ファイルが存在するか確認
		_, err := os.Stat(filePath)
		fileExists := !os.IsNotExist(err)

		var file *os.File
		if fileExists {
			// 既存ファイルの場合は追記モードで開く
			file, err = os.OpenFile(filePath, os.O_RDWR, 0644)
			if err != nil {
				fmt.Println("Error opening existing file:", err)
				os.Exit(1)
			}

			// </HealthData> タグを削除するためにファイルサイズを調整
			fileInfo, _ := file.Stat()
			fileSize := fileInfo.Size()
			if fileSize > int64(len("</HealthData>")) {
				file.Truncate(fileSize - int64(len("</HealthData>")))
				file.Seek(-int64(len("</HealthData>")), io.SeekEnd) //SeekEndからマイナス方向に移動
			}

		} else {
			// 新規ファイルの場合は作成
			file, err = os.Create(filePath)
			if err != nil {
				fmt.Println("Error creating file:", err)
				os.Exit(1)
			}
			// XML ヘッダーとルート要素の開始タグを書き込み
			file.WriteString(`<?xml version="1.0" encoding="utf-8"?>` + "\n")
			file.WriteString("<HealthData>\n")
		}

		// Record を XML 形式で書き込み
		for _, record := range records {
			recordXML, err := xml.MarshalIndent(record, "  ", "    ") // インデント付きでエンコード
			if err != nil {
				fmt.Println("Error marshalling record to XML:", err)
				continue // エラーが発生した場合はスキップ
			}
			_, err = file.Write(append(recordXML, '\n')) // XML を書き込み (改行を追加)
			if err != nil {
				fmt.Println("Error writing record to file:", err)
				file.Close() // エラーが発生した場合はファイルを閉じる
				os.Exit(1)
			}
		}

		// ルート要素の終了タグを書き込み、ファイルを閉じる
		file.WriteString("</HealthData>\n")
		file.Close()
		fmt.Printf("Processed records of type '%s' and saved to '%s'\n", typeAttr, filePath)
	}

	fmt.Println("Done!")
}
