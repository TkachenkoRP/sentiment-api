FROM maven:3.9-eclipse-temurin-17 AS build
WORKDIR /src

COPY pom.xml .
RUN mvn -q -B dependency:go-offline

COPY src ./src
RUN mvn -q -B package -DskipTests

RUN java -Djarmode=tools -jar target/sentiment-api-1.0.0.jar extract --destination /out --application-filename app.jar

ARG ORT_ARCH=linux-x64
RUN set -eux; \
    ORT="$(ls /out/lib/onnxruntime-*.jar)"; \
    mkdir /tmp/ort && cd /tmp/ort && jar xf "$ORT"; \
    for d in ai/onnxruntime/native/*; do [ "$(basename "$d")" = "$ORT_ARCH" ] || rm -rf "$d"; done; \
    rm "$ORT" && jar cf "$ORT" -C /tmp/ort .

RUN set -eux; \
    MODS="$(jdeps --ignore-missing-deps -q --recursive --multi-release 17 --print-module-deps --class-path '/out/lib/*' /out/app.jar)"; \
    jlink --add-modules "$MODS,java.management,java.xml,jdk.crypto.ec,jdk.zipfs" \
          --strip-debug --no-man-pages --no-header-files --compress=2 --output /jre

FROM gcr.io/distroless/cc-debian12:nonroot
WORKDIR /app

COPY --from=build /jre /opt/jre
COPY --from=build /out /app

COPY model /app/model

ENV JAVA_TOOL_OPTIONS="-XX:+UseSerialGC -XX:MaxRAMPercentage=65 -Xss512k -XX:+ExitOnOutOfMemoryError" \
    SENTIMENT_THREADS=2

EXPOSE 8080
ENTRYPOINT ["/opt/jre/bin/java", "-jar", "/app/app.jar"]

