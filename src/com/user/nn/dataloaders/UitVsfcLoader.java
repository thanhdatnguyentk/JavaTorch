package com.user.nn.dataloaders;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class UitVsfcLoader {
    public static final String DEFAULT_DATA_DIR = "examples/data/uit-vsfc";

    public static class Entry {
        public final String text;
        public final String sentimentLabel;
        public final String topicLabel;
        public final int sentimentId;
        public final int topicId;

        public Entry(String text, String sentimentLabel, String topicLabel, int sentimentId, int topicId) {
            this.text = text;
            this.sentimentLabel = sentimentLabel;
            this.topicLabel = topicLabel;
            this.sentimentId = sentimentId;
            this.topicId = topicId;
        }
    }

    public static class LabelEncoder {
        private final Map<String, Integer> labelToId = new LinkedHashMap<>();
        private final List<String> idToLabel = new ArrayList<>();

        public int fitOrGet(String label) {
            String key = normalizeLabel(label);
            Integer id = labelToId.get(key);
            if (id != null) {
                return id;
            }
            int newId = idToLabel.size();
            labelToId.put(key, newId);
            idToLabel.add(key);
            return newId;
        }

        public int getId(String label) {
            String key = normalizeLabel(label);
            Integer id = labelToId.get(key);
            if (id == null) {
                throw new IllegalArgumentException("Unknown label: " + label);
            }
            return id;
        }

        public int size() {
            return idToLabel.size();
        }

        public List<String> labels() {
            return Collections.unmodifiableList(idToLabel);
        }

        private static String normalizeLabel(String s) {
            return s == null ? "" : s.trim().toLowerCase(Locale.ROOT);
        }
    }

    public static class DatasetSplits {
        public final List<Entry> train;
        public final List<Entry> dev;
        public final List<Entry> test;
        public final LabelEncoder sentimentEncoder;
        public final LabelEncoder topicEncoder;

        public DatasetSplits(
                List<Entry> train,
                List<Entry> dev,
                List<Entry> test,
                LabelEncoder sentimentEncoder,
                LabelEncoder topicEncoder) {
            this.train = train;
            this.dev = dev;
            this.test = test;
            this.sentimentEncoder = sentimentEncoder;
            this.topicEncoder = topicEncoder;
        }
    }

    public static DatasetSplits load() throws IOException {
        return load(DEFAULT_DATA_DIR);
    }

    public static DatasetSplits load(String rootDir) throws IOException {
        File root = resolveRootDirectory(rootDir);

        RawSplit trainRaw = readSplit(root, "train");
        RawSplit devRaw = readSplit(root, "dev");
        RawSplit testRaw = readSplit(root, "test");

        LabelEncoder sentimentEncoder = new LabelEncoder();
        LabelEncoder topicEncoder = new LabelEncoder();

        List<Entry> train = encode(trainRaw, sentimentEncoder, topicEncoder, true);
        List<Entry> dev = encode(devRaw, sentimentEncoder, topicEncoder, false);
        List<Entry> test = encode(testRaw, sentimentEncoder, topicEncoder, false);

        return new DatasetSplits(train, dev, test, sentimentEncoder, topicEncoder);
    }

    private static File resolveRootDirectory(String rootDir) throws IOException {
        List<String> candidates = new ArrayList<>();
        if (rootDir != null && !rootDir.trim().isEmpty()) {
            candidates.add(rootDir.trim());
        }
        if (!DEFAULT_DATA_DIR.equals(rootDir)) {
            candidates.add(DEFAULT_DATA_DIR);
        }
        candidates.add("data/uit-vsfc");

        for (String candidate : candidates) {
            File root = new File(candidate);
            if (root.exists() && root.isDirectory()) {
                return root;
            }
        }

        throw new IOException("UIT-VSFC directory not found. Checked: " + candidates);
    }

    private static class RawSplit {
        final String name;
        final List<String> texts;
        final List<String> sentimentLabels;
        final List<String> topicLabels;

        RawSplit(String name, List<String> texts, List<String> sentimentLabels, List<String> topicLabels) {
            this.name = name;
            this.texts = texts;
            this.sentimentLabels = sentimentLabels;
            this.topicLabels = topicLabels;
        }
    }

    private static RawSplit readSplit(File root, String splitName) throws IOException {
        File splitDir = new File(root, splitName);
        if (!splitDir.exists() || !splitDir.isDirectory()) {
            throw new IOException("Missing split directory: " + splitDir.getAbsolutePath());
        }

        File textFile = new File(splitDir, "sents.txt");
        File sentimentFile = new File(splitDir, "sentiments.txt");
        File topicFile = new File(splitDir, "topics.txt");

        if (!textFile.exists() || !sentimentFile.exists() || !topicFile.exists()) {
            throw new IOException("Split " + splitName + " must contain sents.txt, sentiments.txt, topics.txt");
        }

        List<String> texts = readLinesUtf8(textFile, false);
        List<String> sentiments = readLinesUtf8(sentimentFile, true);
        List<String> topics = readLinesUtf8(topicFile, true);

        if (texts.size() != sentiments.size() || texts.size() != topics.size()) {
            throw new IOException("Mismatched line counts in split " + splitName
                    + " (sents=" + texts.size()
                    + ", sentiments=" + sentiments.size()
                    + ", topics=" + topics.size() + ")");
        }
        if (texts.isEmpty()) {
            throw new IOException("Split " + splitName + " is empty");
        }

        return new RawSplit(splitName, texts, sentiments, topics);
    }

    private static List<String> readLinesUtf8(File file, boolean trim) throws IOException {
        List<String> out = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                out.add(trim ? line.trim() : line);
            }
        }
        return out;
    }

    private static List<Entry> encode(RawSplit raw, LabelEncoder sentimentEncoder, LabelEncoder topicEncoder, boolean fit) {
        List<Entry> out = new ArrayList<>(raw.texts.size());
        for (int i = 0; i < raw.texts.size(); i++) {
            String text = raw.texts.get(i);
            String sentiment = raw.sentimentLabels.get(i);
            String topic = raw.topicLabels.get(i);

            if (text == null || text.trim().isEmpty() || sentiment.isEmpty() || topic.isEmpty()) {
                continue;
            }

            int sentimentId;
            int topicId;
            if (fit) {
                sentimentId = sentimentEncoder.fitOrGet(sentiment);
                topicId = topicEncoder.fitOrGet(topic);
            } else {
                try {
                    sentimentId = sentimentEncoder.getId(sentiment);
                    topicId = topicEncoder.getId(topic);
                } catch (IllegalArgumentException ex) {
                    throw new IllegalArgumentException("Unknown label in split " + raw.name + " at line " + (i + 1) + ": " + ex.getMessage());
                }
            }

            out.add(new Entry(text, sentiment, topic, sentimentId, topicId));
        }
        return out;
    }

    public static class VietnameseTokenizer {
        private static final Map<String, String> REPLACEMENTS = createReplacements();

        private static Map<String, String> createReplacements() {
            Map<String, String> m = new HashMap<>();
            m.put("ko", "khong");
            m.put("k", "khong");
            m.put("kh", "khong");
            m.put("hok", "khong");
            m.put("dc", "duoc");
            m.put("đc", "duoc");
            m.put("cx", "cung");
            m.put("vs", "voi");
            m.put("mik", "minh");
            m.put("mk", "minh");
            m.put("bt", "binhthuong");
            m.put("ntn", "nhu_the_nao");
            m.put("j", "gi");
            return m;
        }

        public List<String> tokenize(String text) {
            String normalized = normalize(text);
            if (normalized.isEmpty()) {
                return Collections.emptyList();
            }
            String[] parts = normalized.split("\\s+");
            List<String> out = new ArrayList<>(parts.length);
            for (String p : parts) {
                if (p.isEmpty()) {
                    continue;
                }
                out.add(REPLACEMENTS.getOrDefault(p, p));
            }
            return out;
        }

        public String normalize(String text) {
            if (text == null) {
                return "";
            }
            String s = text.trim().toLowerCase(Locale.ROOT);
            s = Normalizer.normalize(s, Normalizer.Form.NFKC);
            s = s.replaceAll("https?://\\S+", " url ");
            s = s.replaceAll("[@#][\\p{L}\\p{Nd}_]+", " mention ");
            s = s.replaceAll("[^\\p{L}\\p{Nd}\\s_]", " ");
            s = s.replaceAll("\\s+", " ").trim();
            return s;
        }
    }
}
