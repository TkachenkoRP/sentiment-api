package ru.my.sentimentapi;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.actuate.observability.AutoConfigureObservability;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
@AutoConfigureObservability
class SentimentApiApplicationTests {

    @Autowired
    MockMvc mvc;

    @Test
    void positiveRussian() throws Exception {
        mvc.perform(get("/api/sentiment").param("text", "Отличный сервис, всем рекомендую!"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sentiment").value("positive"));
    }

    @Test
    void negativeEnglish() throws Exception {
        mvc.perform(get("/api/sentiment").param("text", "This is terrible and awful"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sentiment").value("negative"));
    }

    @Test
    void blankTextIsRejected() throws Exception {
        mvc.perform(get("/api/sentiment").param("text", " "))
                .andExpect(status().isBadRequest());
    }

    @Test
    void prometheusEndpointIsExposed() throws Exception {
        mvc.perform(get("/actuator/prometheus")).andExpect(status().isOk());
    }

}
