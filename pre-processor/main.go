package main

import (
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// parse the XML file
type HealthData struct {
	Records []Record `xml:"Record"`
}
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
}

func main() {
	// wait args
	if len(os.Args) < 2 {
		fmt.Println("Please provide the path to the XML file.")
		os.Exit(1)
	}

	// first arg is the path to the XML file, second arg is the path to the output directory
	xmlPath := os.Args[1]
	outputDir := os.Args[2]

	// read the XML file
	xmlFile, err := os.Open(xmlPath)
	if err != nil {
		fmt.Println("Error opening XML file:", err)
		os.Exit(1)
	}
	defer xmlFile.Close()

	fmt.Println("Reading XML file:", xmlPath)

	// read the XML file
	xmlData, err := io.ReadAll(xmlFile)
	if err != nil {
		fmt.Println("Error reading XML file:", err)
		os.Exit(1)
	}

	fmt.Println("XML file read successfully")

	var healthData HealthData
	d := xml.NewDecoder(strings.NewReader(string(xmlData)))
	d.Strict = false
	err = d.Decode(&healthData)
	if err != nil {
		fmt.Println("Error parsing XML file:", err)
		os.Exit(1)
	}

	fmt.Println("XML file parsed successfully")

	// Group the records by type attribute
	recordsByType := make(map[string][]Record)
	for _, record := range healthData.Records {
		recordsByType[record.Type] = append(recordsByType[record.Type], record)
	}

	fmt.Println("Records grouped by type successfully")

	// create the output directory
	os.MkdirAll(outputDir, 0755)

	fmt.Println("Output directory created successfully")

	// create the output files (append if file exists, else create new)
	for typeAttr, recs := range recordsByType {
		filePath := filepath.Join(outputDir, fmt.Sprintf("%s.xml", typeAttr))
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			// File does not exist: create new file with root element.
			file, err := os.Create(filePath)
			if err != nil {
				fmt.Println("Error creating output file:", err)
				os.Exit(1)
			}
			// Write XML header and opening root tag.
			file.WriteString(`<?xml version="1.0" encoding="utf-8"?>` + "\n")
			file.WriteString("<HealthData>\n")
			// Write only the values.
			for _, rec := range recs {
				file.WriteString(fmt.Sprintf("<%s><SourceName>%s</SourceName><SourceVersion>%s</SourceVersion><Device>%s</Device><Unit>%s</Unit><CreationDate>%s</CreationDate><StartDate>%s</StartDate><EndDate>%s</EndDate><Value>%s</Value></%s>\n", typeAttr, rec.SourceName, rec.SourceVersion, rec.Device, rec.Unit, rec.CreationDate, rec.StartDate, rec.EndDate, rec.Value, typeAttr))
			}
			// Write closing root tag.
			file.WriteString("</HealthData>\n")
			file.Close()
			fmt.Println("New output file created and records written:", filePath)
		} else {
			// File exists: open file in read-write mode so we can append new records.
			file, err := os.OpenFile(filePath, os.O_RDWR, 0644)
			if err != nil {
				fmt.Println("Error opening existing output file:", err)
				os.Exit(1)
			}
			// Read the file content.
			content, err := io.ReadAll(file)
			if err != nil {
				fmt.Println("Error reading existing output file:", err)
				os.Exit(1)
			}
			closingTag := []byte("</HealthData>\n")
			idx := bytes.LastIndex(content, closingTag)
			if idx == -1 {
				fmt.Println("Error: closing tag not found in file:", filePath)
				os.Exit(1)
			}
			// Truncate the file to remove the closing tag.
			err = file.Truncate(int64(idx))
			if err != nil {
				fmt.Println("Error truncating file:", err)
				os.Exit(1)
			}
			// Seek to the new end.
			_, err = file.Seek(0, io.SeekEnd)
			if err != nil {
				fmt.Println("Error seeking to file end:", err)
				os.Exit(1)
			}
			// Append only the values.
			for _, rec := range recs {
				file.WriteString(fmt.Sprintf("<SourceName>%s</SourceName><SourceVersion>%s</SourceVersion><Device>%s</Device><Unit>%s</Unit><CreationDate>%s</CreationDate><StartDate>%s</StartDate><EndDate>%s</EndDate><Value>%s</Value>\n", rec.SourceName, rec.SourceVersion, rec.Device, rec.Unit, rec.CreationDate, rec.StartDate, rec.EndDate, rec.Value))
			}
			// Append the closing tag again.
			file.Write(closingTag)
			file.Close()
			fmt.Println("Appended records to existing file successfully:", filePath)
		}
	}

	fmt.Println("Done!")
}
