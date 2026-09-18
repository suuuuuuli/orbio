# Arena — debata inwestycyjna wieloagentowa

## Czym to jest
Aplikacja, w której zespół agentów LLM prowadzi ustrukturyzowaną debatę o aktywie
(ticker giełdowy albo token). Wyjściem NIE jest rekomendacja "kupuj/sprzedawaj",
tylko teza z jawnym poziomem pewności i listą falsyfikatorów (co musiałoby się
wydarzyć, żeby tezę porzucić). To framing produktu — trzymaj się go w tekstach UI.

## Agenci
- **bull** — buduje tezę pozytywną
- **bear** — buduje tezę negatywną
- **quant** — liczy na realnych danych, wykonując kod w sandboxie; nie spekuluje
- **judge** — ocenia twierdzenia, wymusza źródła, rozlicza budżet wiarygodności

Cztery role, cztery modele od trzech dostawców — to celowe, chodzi o realną
różnicę zdań, nie o kosmetykę. Wszystkie idą przez JEDEN endpoint zgodny z API
OpenAI (`ORBIO_OPENAI_BASE_URL`), więc różni dostawcy nie oznaczają różnych
klientów w kodzie: `llm.py` normalizuje odpowiedź do wspólnego kształtu.

Nazwy modeli WYŁĄCZNIE ze stałych w config.py, a każdy używany model musi być
zarejestrowany w `MODEL_PROVIDERS` — nieznany model kończy się błędem, nie
domysłem (`provider_for`).

## Mechanika
Debata trwa ROUNDS rund. Każdy agent ma pulę CREDIBILITY_BUDGET punktów i przy
każdym twierdzeniu obstawia część puli (pole `stake`). Twierdzenie obalone lub
bez źródła zabiera stawkę. To wymusza kalibrację zamiast pewności siebie.
Na koniec: teza + poziom pewności + falsyfikatory.

## Stack
- Backend: Python 3.10+, FastAPI, streaming odpowiedzi do frontendu (SSE)
- LLM: SDK Anthropic wskazujące na bramkę Orbio (base_url z .env)
- Frontend: statyczny HTML/JS bez frameworka, renderuje karty twierdzeń na żywo
- Sandbox: wykonywanie kodu kwanta

## Struktura
backend/    config.py, main.py, models.py, agents.py, judge.py, debate.py
frontend/   index.html
data/       dane historyczne (tryb z zamrożoną datą)

## Zasady — przestrzegaj bezwzględnie
- NIE dotykaj .env i nie wypisuj jego zawartości w kodzie ani w logach
- NIGDY nie wypisuj wartości zmiennych z .env — nawet częściowo, nawet przy
  diagnostyce. Zmienna o nazwie sugerującej URL może zawierać klucz, więc
  „to tylko fragment adresu" nie jest wymówką. Diagnozuj po długości, po
  prefiksie schematu (`http://`/`https://`) albo po samym fakcie ustawienia
- NIE wykonuj komend git (commit, push, add) — robię to sam
- Klucz i base_url czytaj wyłącznie przez config.py (load_dotenv)
- Nazwy modeli tylko ze stałych BULL_MODEL / BEAR_MODEL / QUANT_MODEL / JUDGE_MODEL
- Jeden moduł na zadanie. Nie buduj "całej aplikacji" w jednym podejściu
- Nie dodawaj zależności bez dopisania ich do requirements.txt
- Kod ma być czytelny dla jednej osoby pod presją czasu — bez nadmiarowych abstrakcji

## Schemat twierdzenia (źródło prawdy: backend/models.py)
Claim: id, agent, round, text, claim_type (factual|interpretive|quantitative),
source_url, source_quote, stake, confidence, targets (id atakowanego twierdzenia),
status (pending|verified|unsourced|refuted|computed), judge_note,
code, result (tylko dla quant).

`targets` zamienia płaską listę w graf — arena rysuje, co kogo obala.
`stake` jest oddzielone od `confidence` celowo: sędzia karze rozjazd między
deklarowaną pewnością a gotowością postawienia punktów.

## Kontekst
Projekt na 6-dniowy hackathon. Priorytet: działająca pętla debaty + wizualizacja.
Wszystko inne jest opcjonalne.