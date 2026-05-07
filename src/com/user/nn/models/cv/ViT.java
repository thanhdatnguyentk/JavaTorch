package com.user.nn.models.cv;

import com.user.nn.core.*;
import com.user.nn.core.Module;
import com.user.nn.layers.Linear;
import com.user.nn.layers.Conv2d;
import com.user.nn.norm.LayerNorm;
import com.user.nn.attention.TransformerEncoderLayer;
import java.util.List;

/**
 * Vision Transformer (ViT) implementation for image classification.
 */
public class ViT extends Module {
    public int imgSize;
    public int patchSize;
    public int numPatches;
    public int embedDim;
    
    public Conv2d patchEmbed;
    public Parameter clsToken;
    public Parameter posEmbed;
    public Module[] blocks;
    public LayerNorm norm;
    public Linear head;

    public ViT(int imgSize, int patchSize, int inChannels, int numClasses, 
               int embedDim, int depth, int numHeads, int mlpDim, float dropout) {
        this.imgSize = imgSize;
        this.patchSize = patchSize;
        this.embedDim = embedDim;
        this.numPatches = (imgSize / patchSize) * (imgSize / patchSize);

        // 1. Patch Embedding: uses a Conv2d trick
        this.patchEmbed = new Conv2d(inChannels, embedDim, patchSize, patchSize, patchSize, patchSize, 0, 0, true);
        addModule("patchEmbed", patchEmbed);

        // 2. Learnable Class Token and Positional Embedding
        this.clsToken = new Parameter(Torch.randn(new int[]{1, 1, embedDim}).mul(0.02f));
        this.posEmbed = new Parameter(Torch.randn(new int[]{1, numPatches + 1, embedDim}).mul(0.02f));
        addParameter("clsToken", clsToken);
        addParameter("posEmbed", posEmbed);

        // 3. Transformer Encoder Blocks
        this.blocks = new Module[depth];
        for (int i = 0; i < depth; i++) {
            blocks[i] = new TransformerEncoderLayer(embedDim, numHeads, mlpDim, dropout);
            addModule("block" + i, blocks[i]);
        }

        // 4. Global Norm and Classification Head
        this.norm = new LayerNorm(embedDim);
        this.head = new Linear(embedDim, numClasses, true);
        addModule("norm", norm);
        addModule("head", head);
    }

    @Override
    public Tensor forward(Tensor x) {
        int n = x.shape[0];
        
        // Input: [N, C, H, W]
        // 1. Patchify: [N, E, H/P, W/P]
        x = patchEmbed.forward(x);
        
        // 2. Reshape: [N, E, numPatches] -> [N, numPatches, E]
        x = x.reshape(n, embedDim, numPatches);
        x = Torch.transpose(x, 1, 2);
        
        // 3. Prep CLS Token: [N, 1, E]
        Tensor cls = Torch.expand(clsToken.getTensor(), n, 1, embedDim);
        
        // 4. Concatenate CLS token with patch features
        x = Torch.cat(List.of(cls, x), 1);
        
        // 5. Add Position Embedding
        Tensor pos = Torch.expand(posEmbed.getTensor(), n, numPatches + 1, embedDim);
        x = Torch.add(x, pos);
        
        // 6. Transformer Blocks
        for (Module block : blocks) {
            x = block.forward(x);
        }
        
        // 7. Extract CLS token features and pass to head
        // Output from blocks: [N, L+1, E]
        // Take the 0th element in the sequence dimension
        x = Torch.narrow(x, 1, 0, 1).reshape(n, embedDim);
        x = norm.forward(x);
        return head.forward(x);
    }
}
