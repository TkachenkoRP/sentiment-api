package ru.my.sentimentapi;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Инференс готовой модели .onnx через ONNX Runtime (CPU).
 */
@Service
public class OnnxSentimentService {
    private static final Logger log = LoggerFactory.getLogger(OnnxSentimentService.class);

    public record Result(String label, double score, Map<String, Double> scores, double latencyMs) {
    }

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final MeterRegistry registry;
    private final Timer inferenceTimer;

    private final Path modelPath;
    private final Path vocabPath;
    private final List<String> labels;
    private final int maxLength;
    private final int threads;
    private final String modelName;

    private OrtSession session;
    private WordPieceTokenizer tokenizer;

    public OnnxSentimentService(MeterRegistry registry,
                                @Value("${sentiment.model-path}") String modelPath,
                                @Value("${sentiment.vocab-path}") String vocabPath,
                                @Value("${sentiment.labels}") List<String> labels,
                                @Value("${sentiment.max-length}") int maxLength,
                                @Value("${sentiment.threads}") int threads,
                                @Value("${sentiment.model-name}") String modelName) {
        this.registry = registry;
        this.modelPath = Path.of(modelPath);
        this.vocabPath = Path.of(vocabPath);
        this.labels = labels;
        this.maxLength = maxLength;
        this.threads = threads;
        this.modelName = modelName;
        this.inferenceTimer = Timer.builder("sentiment.inference")
                .description("Время инференса ONNX-модели")
                .publishPercentileHistogram()
                .register(registry);
    }

    @PostConstruct
    void load() throws Exception {
        tokenizer = new WordPieceTokenizer(vocabPath);
        try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {
            opts.setIntraOpNumThreads(threads);
            session = env.createSession(modelPath.toString(), opts);
        }
        log.info("ONNX model loaded: {} inputs={} outputs={}", modelPath, session.getInputNames(), session.getOutputNames());
    }

    public Result analyze(String text) {
        long t0 = System.nanoTime();
        try {
            WordPieceTokenizer.Encoding enc = tokenizer.encode(text, maxLength);
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", OnnxTensor.createTensor(env, new long[][]{enc.inputIds()}));
            inputs.put("attention_mask", OnnxTensor.createTensor(env, new long[][]{enc.attentionMask()}));
            if (session.getInputNames().contains("token_type_ids")) {
                inputs.put("token_type_ids", OnnxTensor.createTensor(env, new long[][]{enc.tokenTypeIds()}));
            }
            float[] logits;
            try (OrtSession.Result out = session.run(inputs)) {
                logits = ((float[][]) out.get(0).getValue())[0];
            } finally {
                inputs.values().forEach(OnnxTensor::close);
            }

            double[] p = softmax(logits);
            int best = 0;
            Map<String, Double> scores = new LinkedHashMap<>();
            for (int i = 0; i < p.length; i++) {
                scores.put(labels.get(i), round(p[i]));
                if (p[i] > p[best]) best = i;
            }
            long dt = System.nanoTime() - t0;
            inferenceTimer.record(dt, TimeUnit.NANOSECONDS);
            registry.counter("sentiment.requests", "label", labels.get(best)).increment();
            return new Result(labels.get(best), round(p[best]), scores, dt / 1_000_000.0);
        } catch (OrtException e) {
            throw new IllegalStateException("ONNX inference failed", e);
        }
    }

    public String modelName() {
        return modelName;
    }

    public List<String> labels() {
        return labels;
    }

    private static double[] softmax(float[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) max = Math.max(max, v);
        double sum = 0;
        double[] e = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            e[i] = Math.exp(logits[i] - max);
            sum += e[i];
        }
        for (int i = 0; i < e.length; i++) e[i] /= sum;
        return e;
    }

    private static double round(double v) {
        return Math.round(v * 10000.0) / 10000.0;
    }

    @PreDestroy
    void close() throws OrtException {
        if (session != null) session.close();
    }
}
