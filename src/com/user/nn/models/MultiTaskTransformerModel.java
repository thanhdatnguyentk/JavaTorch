package com.user.nn.models;

import com.user.nn.attention.TransformerEncoderLayer;
import com.user.nn.core.Module;
import com.user.nn.core.Parameter;
import com.user.nn.core.Tensor;
import com.user.nn.core.Torch;
import com.user.nn.layers.Dropout;
import com.user.nn.layers.Embedding;
import com.user.nn.layers.Linear;
import com.user.nn.norm.LayerNorm;

public class MultiTaskTransformerModel extends Module {
    public Embedding embedding;
    public Parameter posEmbed;
    public Module[] blocks;
    public Dropout dropout;
    public LayerNorm norm;
    public Linear sentimentHead;
    public Linear topicHead;
    public int maxSeqLen;

    public MultiTaskTransformerModel(
            int vocabSize,
            int embedDim,
            int maxSeqLen,
            int depth,
            int numHeads,
            int ffDim,
            int sentimentClasses,
            int topicClasses,
            float dropoutP) {
        this.embedding = new Embedding(vocabSize, embedDim);
        this.maxSeqLen = maxSeqLen;
        this.posEmbed = new Parameter(Torch.randn(new int[]{1, maxSeqLen, embedDim}).mul(0.02f));
        this.blocks = new Module[depth];
        for (int i = 0; i < depth; i++) {
            blocks[i] = new TransformerEncoderLayer(embedDim, numHeads, ffDim, dropoutP);
            addModule("block" + i, blocks[i]);
        }
        this.dropout = new Dropout(dropoutP);
        this.norm = new LayerNorm(embedDim);
        this.sentimentHead = new Linear(embedDim, sentimentClasses, true);
        this.topicHead = new Linear(embedDim, topicClasses, true);

        addModule("embedding", embedding);
        addParameter("posEmbed", posEmbed);
        addModule("dropout", dropout);
        addModule("norm", norm);
        addModule("sentimentHead", sentimentHead);
        addModule("topicHead", topicHead);
    }

    private Tensor pooledFeatures(Tensor x) {
        int seqLen = x.shape[1];
        if (seqLen > maxSeqLen) {
            throw new IllegalArgumentException("Sequence length " + seqLen + " exceeds maxSeqLen=" + maxSeqLen);
        }

        Tensor h = embedding.forward(x);
        Tensor pos = Torch.narrow(posEmbed.getTensor(), 1, 0, seqLen);
        pos = Torch.expand(pos, x.shape[0], seqLen, h.shape[2]);
        h = Torch.add(h, pos);
        h = dropout.forward(h);
        for (Module block : blocks) {
            h = block.forward(h);
        }

        int batch = x.shape[0];
        int embedDim = h.shape[2];
        Tensor clsLike = Torch.narrow(h, 1, 0, 1).reshape(batch, embedDim);
        return norm.forward(clsLike);
    }

    public Tensor[] forwardBoth(Tensor x) {
        Tensor shared = pooledFeatures(x);
        Tensor sentimentLogits = sentimentHead.forward(shared);
        Tensor topicLogits = topicHead.forward(shared);
        return new Tensor[] { sentimentLogits, topicLogits };
    }

    @Override
    public Tensor forward(Tensor x) {
        return forwardBoth(x)[0];
    }

    public java.util.Map<String, Float> getEmbeddingNorms(Tensor x, com.user.nn.dataloaders.UitVsfcLoader.VietnameseTokenizer tokenizer, com.user.nn.dataloaders.Data.Vocabulary vocab) {
        try (com.user.nn.core.MemoryScope scope = new com.user.nn.core.MemoryScope()) {
            Tensor h = embedding.forward(x);
            // In Transformer, we might want to see the combined embedding (token + pos)
            int seqLen = x.shape[1];
            Tensor pos = Torch.narrow(posEmbed.getTensor(), 1, 0, seqLen);
            pos = Torch.expand(pos, x.shape[0], seqLen, h.shape[2]);
            h = Torch.add(h, pos);

            Tensor squared = Torch.mul(h, h);
            Tensor sum = Torch.sum(squared, 2);
            Tensor norms = Torch.sqrt(sum);

            if (norms.isGPU()) norms.toCPU();

            java.util.Map<String, Float> weights = new java.util.HashMap<>();
            for (int j = 0; j < seqLen; j++) {
                float tokenId = x.data[j];
                if (tokenId == 0) continue;
                String word = vocab.getWord((int) tokenId);
                if (word != null) {
                    weights.put(word, norms.data[j]);
                }
            }
            return weights;
        }
    }
}
