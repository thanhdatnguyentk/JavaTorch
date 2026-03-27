package com.user.nn.utils.dashboard;

import java.io.InputStream;
import java.util.Map;

/**
 * Interface cho việc đăng ký xử lý tác vụ Inference (Dự đoán ảnh, text,...)
 * Từ file upload hoặc text raw, tiền xử lý và gọi Model, sau đó trả về dữ liệu kết quả tuỳ chỉnh.
 */
public interface PredictHandler {
    
    /**
     * @param fileName Tên file upload (nếu có)
     * @param fileStream Luồng dữ liệu file (nếu có)
     * @param text Dữ liệu text truyền lên (nếu có)
     * @return Bất cứ Object nào, sẽ được Javalin tự serialize qua JSON gửi về Frontend.
     */
    Object predict(String fileName, InputStream fileStream, String text) throws Exception;
}
