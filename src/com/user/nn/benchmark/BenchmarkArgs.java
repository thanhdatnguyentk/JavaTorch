package com.user.nn.benchmark;

import java.util.LinkedHashMap;
import java.util.Map;

public final class BenchmarkArgs {
    private BenchmarkArgs() {
    }

    public static Map<String, String> parse(String[] args) {
        Map<String, String> parsed = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i++) {
            String token = args[i];
            if (!token.startsWith("--")) {
                continue;
            }

            String body = token.substring(2);
            int eq = body.indexOf('=');
            if (eq >= 0) {
                parsed.put(body.substring(0, eq), body.substring(eq + 1));
                continue;
            }

            if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                parsed.put(body, args[i + 1]);
                i++;
            } else {
                parsed.put(body, "true");
            }
        }
        return parsed;
    }

    public static String getString(Map<String, String> args, String key, String defaultValue) {
        return args.getOrDefault(key, defaultValue);
    }

    public static int getInt(Map<String, String> args, String key, int defaultValue) {
        String value = args.get(key);
        if (value == null) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException ex) {
            throw new IllegalArgumentException("Invalid integer for --" + key + ": " + value);
        }
    }

    public static long getLong(Map<String, String> args, String key, long defaultValue) {
        String value = args.get(key);
        if (value == null) {
            return defaultValue;
        }
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException ex) {
            throw new IllegalArgumentException("Invalid long for --" + key + ": " + value);
        }
    }

    public static boolean getBoolean(Map<String, String> args, String key, boolean defaultValue) {
        String value = args.get(key);
        if (value == null) {
            return defaultValue;
        }
        if ("true".equalsIgnoreCase(value) || "1".equals(value) || "yes".equalsIgnoreCase(value)) {
            return true;
        }
        if ("false".equalsIgnoreCase(value) || "0".equals(value) || "no".equalsIgnoreCase(value)) {
            return false;
        }
        throw new IllegalArgumentException("Invalid boolean for --" + key + ": " + value);
    }
}