package com.user.nn.predict;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.dataloaders.Data;

import java.util.*;
import java.util.function.Function;

/**
 * Predictor chuyên biệt cho bài toán phân tích văn bản / NLP.
 * Hỗ trợ:
 *   - Tokenize + padding text
 *   - Dự đoán sentiment từ text raw
 *   - Vocabulary mapping
 */
public class TextPredictor extends Predictor {

    private Data.Vocabulary vocab;
    private Data.BasicTokenizer tokenizer;
    private Function<String, List<String>> customTokenizer;
    private int maxLen;

    /**
     * Tạo TextPredictor.
     *
     * @param model  Model NLP đã train
     * @param vocab  Vocabulary dùng khi train
     * @param maxLen Chiều dài sequence tối đa
     */
    public TextPredictor(Module model, Data.Vocabulary vocab, int maxLen) {
        super(model);
        this.vocab = vocab;
        this.tokenizer = new Data.BasicTokenizer();
        this.maxLen = maxLen;
    }

    /**
     * Tạo TextPredictor với label mapping.
     */
    public TextPredictor(Module model, Data.Vocabulary vocab, int maxLen, String[] labels) {
        super(model, labels);
        this.vocab = vocab;
        this.tokenizer = new Data.BasicTokenizer();
        this.maxLen = maxLen;
    }

    /**
     * Đặt custom tokenizer để thay thế BasicTokenizer mặc định.
     * Sử dụng khi cần tokenizer chuyên biệt (VD: VietnameseTokenizer).
     *
     * @param tokenizer Function nhận text trả về List<String> tokens
     * @return this (builder pattern)
     */
    public TextPredictor setTokenizer(Function<String, List<String>> tokenizer) {
        this.customTokenizer = tokenizer;
        return this;
    }

    // ======================== PREDICT FROM TEXT ========================

    /**
     * Dự đoán từ một câu text.
     *
     * @param text Câu raw text
     * @return PredictionResult
     */
    public PredictionResult predictText(String text) {
        Tensor input = preprocessText(text);
        return predict(input);
    }

    /**
     * Dự đoán batch từ nhiều câu text.
     *
     * @param texts Mảng text
     * @return Mảng PredictionResult
     */
    public PredictionResult[] predictTexts(String[] texts) {
        int n = texts.length;
        float[] batchData = new float[n * maxLen];

        for (int i = 0; i < n; i++) {
            float[] tokenIds = tokenizeAndPad(texts[i]);
            System.arraycopy(tokenIds, 0, batchData, i * maxLen, maxLen);
        }

        Tensor input = Torch.tensor(batchData, n, maxLen);
        return predictBatch(input);
    }

    /**
     * Dự đoán sentiment (tích cực/tiêu cực) từ text.
     * Giả định model output 2 classes: [negative, positive]
     *
     * @param text Raw text
     * @return "positive" hoặc "negative" kèm confidence
     */
    public String predictSentiment(String text) {
        PredictionResult result = predictText(text);
        String sentiment = result.getPredictedClass() == 1 ? "POSITIVE" : "NEGATIVE";
        return String.format("%s (confidence: %.4f)", sentiment, result.getConfidence());
    }

    // ======================== PREPROCESSING ========================

    /**
     * Tokenize và pad text thành Tensor [1, maxLen].
     */
    private Tensor preprocessText(String text) {
        float[] tokenIds = tokenizeAndPad(text);
        return Torch.tensor(tokenIds, 1, maxLen);
    }

    /**
     * Tokenize text, map qua vocab, pad hoặc truncate đến maxLen.
     */
    private float[] tokenizeAndPad(String text) {
        List<String> tokens = (customTokenizer != null)
                ? customTokenizer.apply(text)
                : tokenizer.tokenize(text);

        float[] ids = new float[maxLen];
        // Pad token = 0 (default)
        int len = Math.min(tokens.size(), maxLen);
        for (int i = 0; i < len; i++) {
            ids[i] = vocab.getId(tokens.get(i));
        }
        return ids;
    }

    // ======================== FACTORY ========================

    /**
     * Tạo TextPredictor cho Sentiment Analysis (2 classes mặc định).
     */
    public static TextPredictor forSentiment(Module model, Data.Vocabulary vocab, int maxLen) {
        return new TextPredictor(model, vocab, maxLen, 
            new String[] { "Negative", "Positive" });
    }

    /**
     * Tạo TextPredictor cho Sentiment Analysis với labels tuỳ chỉnh.
     *
     * @param labels Danh sách tên label (VD: ["negative", "neutral", "positive"])
     */
    public static TextPredictor forSentiment(Module model, Data.Vocabulary vocab, int maxLen, String[] labels) {
        return new TextPredictor(model, vocab, maxLen, labels);
    }

    // ======================== GETTERS ========================

    public Data.Vocabulary getVocab() { return vocab; }
    public int getMaxLen() { return maxLen; }
}
