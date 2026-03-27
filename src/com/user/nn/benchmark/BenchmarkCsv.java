package com.user.nn.benchmark;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.Map;

public final class BenchmarkCsv {
    private BenchmarkCsv() {
    }

    public static void appendRow(Path file, LinkedHashMap<String, String> row) throws IOException {
        Files.createDirectories(file.getParent());
        boolean exists = Files.exists(file);

        try (BufferedWriter writer = Files.newBufferedWriter(
                file,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND)) {
            if (!exists) {
                writer.write(String.join(",", row.keySet()));
                writer.newLine();
            }

            boolean first = true;
            for (Map.Entry<String, String> entry : row.entrySet()) {
                if (!first) {
                    writer.write(',');
                }
                writer.write(escape(entry.getValue()));
                first = false;
            }
            writer.newLine();
        }
    }

    private static String escape(String value) {
        if (value == null) {
            return "";
        }
        boolean needsQuotes = value.indexOf(',') >= 0 || value.indexOf('"') >= 0 || value.indexOf('\n') >= 0;
        if (!needsQuotes) {
            return value;
        }
        return '"' + value.replace("\"", "\"\"") + '"';
    }
}
