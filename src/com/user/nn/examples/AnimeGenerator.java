package com.user.nn.examples;

import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import java.io.File;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.util.Locale;

public class AnimeGenerator {

    private TrainGANAnime.Generator generator;
    private int latentDim;
    private boolean useGpu;

    public AnimeGenerator(String modelPath, int latentDim) throws Exception {
        this.latentDim = latentDim;
        this.generator = new TrainGANAnime.Generator(latentDim);
        this.generator.load(modelPath);
        this.useGpu = tryToGpu();
        this.generator.eval();
    }

    private boolean tryToGpu() {
        try {
            this.generator.toGPU();
            return true;
        } catch (Throwable t) {
            this.generator.toCPU();
            return false;
        }
    }

    public BufferedImage generateImage() {
        Tensor z = Torch.randn(new int[]{1, latentDim});
        if (useGpu) {
            z.toGPU();
        }
        Tensor out = this.generator.forward(z);
        out.toCPU();
        out = out.reshape(3, 64, 64);
        
        BufferedImage img = new BufferedImage(64, 64, BufferedImage.TYPE_INT_RGB);
        float[] rgbData = out.data;
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                int r = (int) (((rgbData[0 * 64 * 64 + y * 64 + x] + 1) / 2) * 255);
                int g = (int) (((rgbData[1 * 64 * 64 + y * 64 + x] + 1) / 2) * 255);
                int b = (int) (((rgbData[2 * 64 * 64 + y * 64 + x] + 1) / 2) * 255);
                r = Math.max(0, Math.min(255, r));
                g = Math.max(0, Math.min(255, g));
                b = Math.max(0, Math.min(255, b));
                img.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }
        return img;
    }

    public void saveImage(String path) throws Exception {
        File f = new File(path);
        BufferedImage img = generateImage();
        ImageIO.write(img, "png", f);
    }

    public static void main(String[] args) throws Exception {
        String modelPath = args.length > 0 ? args[0] : "generated/anime_gan/gan_anime_generator.bin";
        int latent = args.length > 1 ? Integer.parseInt(args[1]) : 100;
        int count = args.length > 2 ? Integer.parseInt(args[2]) : 8;
        String outDir = args.length > 3 ? args[3] : "generated/anime_gan/samples";

        File out = new File(outDir);
        if (!out.exists()) {
            out.mkdirs();
        }

        AnimeGenerator generator = new AnimeGenerator(modelPath, latent);
        for (int i = 0; i < count; i++) {
            String path = new File(out, String.format(Locale.US, "anime_%03d.png", i)).getPath();
            generator.saveImage(path);
            System.out.println("Saved: " + path);
        }
    }
}
