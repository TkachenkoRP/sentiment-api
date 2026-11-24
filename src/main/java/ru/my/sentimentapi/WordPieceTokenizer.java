package ru.my.sentimentapi;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Токенизатор BERT WordPiece (cased) — ПРЕДОБРАБОТКА входа для ONNX-модели.
 * Он превращает текст в числовые id по словарю vocab.txt и НЕ содержит никакой
 * логики определения тональности: решение принимает только нейросеть.
 * Повторяет поведение BertNormalizer + BertPreTokenizer + WordPiece из HF tokenizers.
 */
public class WordPieceTokenizer {
    public record Encoding(long[] inputIds, long[] attentionMask, long[] tokenTypeIds) {
    }

    private static final int MAX_CHARS_PER_WORD = 100;

    private final Map<String, Integer> vocab = new HashMap<>();
    private final int clsId;
    private final int sepId;
    private final int unkId;

    public WordPieceTokenizer(Path vocabFile) throws IOException {
        List<String> lines = Files.readAllLines(vocabFile, StandardCharsets.UTF_8);
        for (int i = 0; i < lines.size(); i++) {
            vocab.putIfAbsent(lines.get(i), i);
        }
        clsId = vocab.get("[CLS]");
        sepId = vocab.get("[SEP]");
        unkId = vocab.get("[UNK]");
    }

    public Encoding encode(String text, int maxLength) {
        List<Integer> ids = new ArrayList<>();
        ids.add(clsId);
        outer:
        for (String word : preTokenize(clean(text))) {
            for (int id : wordPiece(word)) {
                if (ids.size() >= maxLength - 1) {
                    break outer;                       // усечение длинного текста
                }
                ids.add(id);
            }
        }
        ids.add(sepId);

        long[] inputIds = new long[ids.size()];
        long[] mask = new long[ids.size()];
        for (int i = 0; i < inputIds.length; i++) {
            inputIds[i] = ids.get(i);
            mask[i] = 1L;
        }
        return new Encoding(inputIds, mask, new long[ids.size()]);
    }

    /**
     * BertNormalizer: убрать управляющие символы, нормализовать пробелы, обособить CJK.
     */
    private static String clean(String text) {
        StringBuilder sb = new StringBuilder(text.length() + 8);
        text.codePoints().forEach(cp -> {
            if (cp == 0 || cp == 0xFFFD || isControl(cp)) return;
            if (Character.isWhitespace(cp) || Character.isSpaceChar(cp)) {
                sb.append(' ');
            } else if (isCjk(cp)) {
                sb.append(' ').appendCodePoint(cp).append(' ');
            } else {
                sb.appendCodePoint(cp);
            }
        });
        return sb.toString();
    }

    /**
     * BertPreTokenizer: разбиение по пробелам и отдельным знакам пунктуации.
     */
    private static List<String> preTokenize(String text) {
        List<String> words = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        for (int cp : text.codePoints().toArray()) {
            if (cp == ' ') {
                flush(cur, words);
            } else if (isPunctuation(cp)) {
                flush(cur, words);
                words.add(new String(Character.toChars(cp)));
            } else {
                cur.appendCodePoint(cp);
            }
        }
        flush(cur, words);
        return words;
    }

    /**
     * WordPiece: жадный поиск самого длинного префикса из словаря (## — продолжение слова).
     */
    private List<Integer> wordPiece(String word) {
        int[] cps = word.codePoints().toArray();
        if (cps.length > MAX_CHARS_PER_WORD) return List.of(unkId);
        List<Integer> out = new ArrayList<>();
        int start = 0;
        while (start < cps.length) {
            int end = cps.length;
            Integer found = null;
            while (start < end) {
                String piece = new String(cps, start, end - start);
                if (start > 0) piece = "##" + piece;
                found = vocab.get(piece);
                if (found != null) break;
                end--;
            }
            if (found == null) return List.of(unkId);
            out.add(found);
            start = end;
        }
        return out;
    }

    private static void flush(StringBuilder cur, List<String> words) {
        if (cur.length() > 0) {
            words.add(cur.toString());
            cur.setLength(0);
        }
    }

    private static boolean isControl(int cp) {
        if (cp == '\t' || cp == '\n' || cp == '\r') return false;
        return switch (Character.getType(cp)) {
            case Character.CONTROL, Character.FORMAT, Character.UNASSIGNED, Character.PRIVATE_USE -> true;
            default -> false;
        };
    }

    private static boolean isPunctuation(int cp) {
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
            return true;
        }
        return switch (Character.getType(cp)) {
            case Character.CONNECTOR_PUNCTUATION, Character.DASH_PUNCTUATION, Character.START_PUNCTUATION,
                 Character.END_PUNCTUATION, Character.INITIAL_QUOTE_PUNCTUATION,
                 Character.FINAL_QUOTE_PUNCTUATION, Character.OTHER_PUNCTUATION -> true;
            default -> false;
        };
    }

    private static boolean isCjk(int cp) {
        return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF)
               || (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2EBEF)
               || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
    }
}
