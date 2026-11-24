package ru.my.sentimentapi;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import java.util.Map;

@RestController
@RequestMapping("/api")
public class SentimentController {
    private static final int MAX_TEXT = 2000;

    public record TextRequest(String text) {
    }

    public record SentimentResponse(String sentiment, double score, Map<String, Double> scores,
                                    String model, double latencyMs, String pod) {
    }

    private final OnnxSentimentService service;

    public SentimentController(OnnxSentimentService service) {
        this.service = service;
    }

    // GET /api/sentiment?text=hello  ->  {"sentiment":"neutral", ...}
    @GetMapping("/sentiment")
    public SentimentResponse get(@RequestParam String text) {
        return analyze(text);
    }

    // POST /api/sentiment  {"text": "..."}
    @PostMapping("/sentiment")
    public SentimentResponse post(@RequestBody TextRequest request) {
        return analyze(request.text());
    }

    @GetMapping("/model")
    public Map<String, Object> model() {
        return Map.of("model", service.modelName(), "labels", service.labels(), "runtime", "ONNX Runtime (CPU)");
    }

    private SentimentResponse analyze(String text) {
        if (text == null || text.isBlank() || text.length() > MAX_TEXT) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "text must be 1.." + MAX_TEXT + " characters");
        }
        OnnxSentimentService.Result r = service.analyze(text);
        // имя пода = HOSTNAME: видно, какая реплика ответила (демонстрация балансировки)
        String pod = System.getenv().getOrDefault("HOSTNAME", "local");
        return new SentimentResponse(r.label(), r.score(), r.scores(), service.modelName(), r.latencyMs(), pod);
    }
}
