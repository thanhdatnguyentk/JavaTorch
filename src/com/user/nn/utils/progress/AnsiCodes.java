package com.user.nn.utils.progress;

/**
 * ANSI escape codes for terminal control.
 * Provides utilities for cursor movement, line clearing, and text coloring.
 */
public class AnsiCodes {
    
    // ANSI escape sequences
    private static final String ESC = "\u001B[";
    
    // Cursor control
    public static final String CURSOR_UP = ESC + "1A";
    public static final String CURSOR_DOWN = ESC + "1B";
    public static final String CURSOR_FORWARD = ESC + "1C";
    public static final String CURSOR_BACK = ESC + "1D";
    public static final String SAVE_CURSOR = ESC + "s";
    public static final String RESTORE_CURSOR = ESC + "u";
    public static final String HIDE_CURSOR = ESC + "?25l";
    public static final String SHOW_CURSOR = ESC + "?25h";
    
    // Line control
    public static final String CLEAR_LINE = ESC + "2K";
    public static final String CLEAR_LINE_TO_END = ESC + "0K";
    public static final String CLEAR_LINE_TO_START = ESC + "1K";
    public static final String CLEAR_SCREEN = ESC + "2J";
    public static final String MOVE_TO_LINE_START = "\r";
    
    // Colors (foreground)
    public static final String BLACK = ESC + "30m";
    public static final String RED = ESC + "31m";
    public static final String GREEN = ESC + "32m";
    public static final String YELLOW = ESC + "33m";
    public static final String BLUE = ESC + "34m";
    public static final String MAGENTA = ESC + "35m";
    public static final String CYAN = ESC + "36m";
    public static final String WHITE = ESC + "37m";
    
    // Bright colors
    public static final String BRIGHT_BLACK = ESC + "90m";
    public static final String BRIGHT_RED = ESC + "91m";
    public static final String BRIGHT_GREEN = ESC + "92m";
    public static final String BRIGHT_YELLOW = ESC + "93m";
    public static final String BRIGHT_BLUE = ESC + "94m";
    public static final String BRIGHT_MAGENTA = ESC + "95m";
    public static final String BRIGHT_CYAN = ESC + "96m";
    public static final String BRIGHT_WHITE = ESC + "97m";
    
    // Text styles
    public static final String RESET = ESC + "0m";
    public static final String BOLD = ESC + "1m";
    public static final String DIM = ESC + "2m";
    public static final String ITALIC = ESC + "3m";
    public static final String UNDERLINE = ESC + "4m";
    
    // Platform detection
    private static final boolean IS_WINDOWS = System.getProperty("os.name").toLowerCase().contains("win");
    private static final boolean ANSI_SUPPORTED = checkAnsiSupport();
    
    /**
     * Check if ANSI codes are supported on this terminal.
     */
    private static boolean checkAnsiSupport() {
        // Check common environment variables
        String term = System.getenv("TERM");
        if (term != null && !term.equals("dumb")) {
            return true;
        }
        
        // Windows Terminal and ConEmu support ANSI
        String wtSession = System.getenv("WT_SESSION");
        String conEmu = System.getenv("ConEmuANSI");
        if (wtSession != null || "ON".equals(conEmu)) {
            return true;
        }
        
        // Check if running in IDE (IntelliJ, Eclipse, VS Code)
        String ideaInitial = System.getenv("IDEA_INITIAL_DIRECTORY");
        String vscode = System.getenv("VSCODE_PID");
        if (ideaInitial != null || vscode != null) {
            return true;
        }
        
        // On Windows without explicit support, disable ANSI
        if (IS_WINDOWS) {
            return false;
        }
        
        // Unix-like systems generally support ANSI
        return true;
    }
    
    /**
     * Enable Windows console virtual terminal processing (for ANSI support).
     * This is a best-effort attempt - may not work on all Windows versions.
     */
    public static void enableWindowsAnsi() {
        if (IS_WINDOWS) {
            try {
                // Try to enable virtual terminal processing via Windows API
                // This requires Windows 10 or later
                ProcessBuilder pb = new ProcessBuilder("cmd", "/c", "echo");
                pb.inheritIO();
                Process process = pb.start();
                process.waitFor();
            } catch (Exception e) {
                // Silently fail - not critical
            }
        }
    }
    
    /**
     * Check if ANSI codes are supported.
     */
    public static boolean isAnsiSupported() {
        return ANSI_SUPPORTED;
    }
    
    /**
     * Clear the current line and move cursor to start.
     */
    public static String clearLine() {
        return MOVE_TO_LINE_START + CLEAR_LINE;
    }
    
    /**
     * Move cursor up n lines.
     */
    public static String cursorUp(int n) {
        return ESC + n + "A";
    }
    
    /**
     * Move cursor down n lines.
     */
    public static String cursorDown(int n) {
        return ESC + n + "B";
    }
    
    /**
     * Move cursor to specific position (1-indexed).
     */
    public static String moveCursor(int row, int col) {
        return ESC + row + ";" + col + "H";
    }
    
    /**
     * Apply color to text with optional styles.
     */
    public static String colorText(String text, String color) {
        if (!ANSI_SUPPORTED) {
            return text;
        }
        return color + text + RESET;
    }
    
    /**
     * Apply color and bold to text.
     */
    public static String colorTextBold(String text, String color) {
        if (!ANSI_SUPPORTED) {
            return text;
        }
        return BOLD + color + text + RESET;
    }
    
    /**
     * Strip ANSI codes from text (for length calculation).
     */
    public static String stripAnsi(String text) {
        return text.replaceAll("\u001B\\[[;\\d]*m", "");
    }
    
    /**
     * Get the visible length of text (ignoring ANSI codes).
     */
    public static int visibleLength(String text) {
        return stripAnsi(text).length();
    }
    
    /**
     * Disable ANSI codes (return empty strings or plain text).
     */
    private static boolean forceDisable = false;
    
    public static void setForceDisable(boolean disable) {
        forceDisable = disable;
    }
    
    public static boolean isEnabled() {
        return ANSI_SUPPORTED && !forceDisable;
    }
}
