package com.user.nn.dataloaders;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.*;

public class MovieCommentLoader {
    public static final String DATA_DIR = "data/rt-polarity/";
    
    // Direct raw text files from a GitHub repository
    public static final String POS_URL = "https://raw.githubusercontent.com/arpit7123/DeepLearningQA/master/rt-polarity.pos";
    public static final String NEG_URL = "https://raw.githubusercontent.com/arpit7123/DeepLearningQA/master/rt-polarity.neg";

    public static class Entry {
        public String text;
        public int label;

        public Entry(String text, int label) {
            this.text = text;
            this.label = label;
        }
    }

    public static List<Entry> load() throws Exception {
        prepareData();
        List<Entry> entries = new ArrayList<>();
        
        readLines(new File(DATA_DIR + "pos.txt"), 1, entries);
        readLines(new File(DATA_DIR + "neg.txt"), 0, entries);

        return entries;
    }

    private static void readLines(File file, int label, List<Entry> out) throws IOException {
        if (!file.exists()) throw new FileNotFoundException("Missing data file: " + file.getAbsolutePath());
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) out.add(new Entry(line, label));
            }
        }
    }

    public static void prepareData() throws Exception {
        File dir = new File(DATA_DIR);
        if (!dir.exists()) dir.mkdirs();

        download(POS_URL, new File(DATA_DIR + "pos.txt"));
        download(NEG_URL, new File(DATA_DIR + "neg.txt"));
    }

    private static void download(String urlStr, File target) throws Exception {
        if (target.exists() && target.length() > 0) return;
        
        System.out.println("Downloading " + urlStr + " ...");
        URL url = new URL(urlStr);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestProperty("User-Agent", "Mozilla/5.0");
        
        try (InputStream in = conn.getInputStream();
             FileOutputStream out = new FileOutputStream(target)) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
        }
        System.out.println("Downloaded " + target.getName() + " size: " + target.length());
    }
}
