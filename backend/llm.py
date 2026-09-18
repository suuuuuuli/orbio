"""Cienka warstwa nad bramka Orbio - dwa SDK, jedno wejscie.

Model wybiera dostawce przez MODEL_PROVIDERS z config.py: "anthropic" idzie
SDK Anthropic, "openai" przez chat.completions. Odpowiedzi obu sa normalizowane
do _Reply, wiec parsowanie, retry i logowanie nie wiedza, kto odpowiadal.

Trzy funkcje publiczne, wszystkie wymuszaja odpowiedz w postaci JSON zgodnego
ze schematem Pydantic:
- call_structured()             -> zwalidowany obiekt
- call_structured_with_search() -> (obiekt, URL-e z wyszukiwania)
- call_structured_with_tools()  -> (obiekt, URL-e, wyjscia z sandboxa)
"""

import json
import re
import time
from typing import NamedTuple

import anthropic
import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from config import (
    DEFAULT_TIERS,
    MAX_TOKENS,
    ORBIO_BASE_URL,
    ORBIO_KEY,
    ORBIO_OPENAI_BASE_URL,
    provider_for,
)
from sources import allowlist_for

# Bramka Orbio uwierzytelnia sie tokenem (Authorization: Bearer), nie x-api-key.
_anthropic_client = anthropic.Anthropic(
    base_url=ORBIO_BASE_URL,
    auth_token=ORBIO_KEY,
    api_key=None,
)

_openai_client: OpenAI | None = None


def _openai() -> OpenAI:
    """Klient OpenAI tworzony przy pierwszym uzyciu - z kontrola konfiguracji.

    ORBIO_OPENAI_BASE_URL musi byc adresem http(s). Jesli w .env siedzi tam
    cos innego (np. klucz), SDK zglasza mylacy 'Connection error' - lepiej
    powiedziec wprost, co jest nie tak.
    """
    global _openai_client
    if _openai_client is None:
        url = (ORBIO_OPENAI_BASE_URL or "").strip()
        if not url:
            raise RuntimeError(
                "ORBIO_OPENAI_BASE_URL nie jest ustawione w .env - "
                "sciezka OpenAI nie ma gdzie wolac"
            )
        if not url.startswith(("http://", "https://")):
            raise RuntimeError(
                "ORBIO_OPENAI_BASE_URL nie wyglada na adres http(s) - "
                "w .env jest tam wartosc innego rodzaju (klucz?). "
                "Wpisz tam adres bramki w formacie OpenAI."
            )
        _openai_client = OpenAI(base_url=url, api_key=ORBIO_KEY)
    return _openai_client


_JSON_INSTRUCTION = """\
Respond with EXACTLY ONE JSON object matching the schema below.
No preamble, no commentary after the JSON, no markdown fences (```).
The first character of your response is an opening brace, the last a closing brace.

JSON Schema:
"""


def _search_tool() -> dict:
    """Wyszukiwarka bramki. Jeden silnik (exa) dla WSZYSTKICH agentow.

    Rozne silniki znaczylyby rozne indeksy, a wtedy roznica zdan bull/bear
    bylaby artefaktem infrastruktury, nie mysli. allowed_domains bierzemy
    z SOURCE_TIERS - filtrowanie domen robi bramka, nie druga allowlista.
    """
    return {
        "type": "openrouter:web_search",
        "parameters": {
            "engine": "exa",
            "max_results": 5,
            "max_uses": 3,
            "allowed_domains": allowlist_for(DEFAULT_TIERS),
        },
    }


def _shell_tool(container_id: str) -> dict:
    """Sandbox dla kwanta. Kontener BEZ SIECI - quant liczy, a nie dociaga."""
    return {
        "type": "openrouter:shell",
        "parameters": {
            "environment": {
                "type": "container_reference",
                "container_id": container_id,
            }
        },
    }


# Limit wywolan narzedzi - pole na poziomie requestu, nie w definicji narzedzia.
_MAX_TOOL_CALLS = 5

# Ile razy wolno kontynuowac ture przerwana przez narzedzie (stop_reason=pause_turn).
_MAX_CONTINUATIONS = 3

# Sufit dla podnoszenia limitu przy ponowieniu po obcieciu odpowiedzi.
_MAX_TOKENS_CEILING = 12000

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_fences(raw: str) -> str:
    """Usuwa ewentualne ogrodzenie ```json ... ``` i biale znaki."""
    return _FENCE.sub("", raw.strip()).strip()


def _text_of(response) -> str:
    """Wyciaga blok tekstowy z JSON-em z odpowiedzi Anthropica.

    Przy wyszukiwaniu odpowiedz ma wiele blokow (narracja miedzy zapytaniami,
    wyniki narzedzia, na koniec JSON), wiec content[0] to nie to samo co wynik.
    Idziemy od konca i bierzemy pierwszy blok, ktory wyglada jak obiekt JSON.
    """
    texts = [b.text for b in response.content if b.type == "text"]
    for text in reversed(texts):
        if _strip_fences(text).startswith("{"):
            return text
    return "".join(texts)


def _blocks_of_type(payload: object, wanted: str) -> list[dict]:
    """Szuka w odpowiedzi blokow o danym polu 'type', na dowolnej glebokosci.

    Bramka pakuje wyniki narzedzi w rozne miejsca i kształt nie jest stabilny
    miedzy wersjami - zamiast zgadywac sciezke, szukamy po 'type' bloku.
    """
    found: list[dict] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == wanted:
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return found


def _collect_urls(payload: object) -> list[str]:
    """Zbiera URL-e z fragmentu odpowiedzi (bez duplikatow, z kolejnoscia)."""
    urls: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            url = node.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                if url not in urls:
                    urls.append(url)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return urls


def _shell_outputs(payload: object) -> list[str]:
    """Wyciaga tresc z blokow openrouter_shell_tool_result."""
    outputs: list[str] = []
    for block in _blocks_of_type(payload, "openrouter_shell_tool_result"):
        parts = [
            str(block[key]).strip()
            for key in ("output", "stdout", "stderr", "content", "result")
            if block.get(key)
        ]
        if parts:
            outputs.append("\n".join(parts))
    return outputs


def _search_urls_anthropic(response) -> list[str]:
    """URL-e z blokow web_search_tool_result (SDK Anthropic)."""
    urls: list[str] = []
    for block in response.content:
        if block.type != "web_search_tool_result":
            continue
        # Sukces: content to lista wynikow. Blad narzedzia: pojedynczy obiekt.
        results = block.content
        if not isinstance(results, list):
            print(f"[llm] wyszukiwanie zwrocilo blad: {results}")
            continue
        for result in results:
            url = getattr(result, "url", None)
            if url and url not in urls:
                urls.append(url)
    return urls


def _strict_json_schema(schema: type[BaseModel]) -> dict:
    """Schemat pod response_format ze strict: true.

    Tryb strict wymaga additionalProperties: false i KAZDEGO pola w 'required'.
    Pydantic tego nie generuje (pola z domyslna wartoscia sa opcjonalne), wiec
    domykamy schemat tutaj - pola opcjonalne i tak przyjmuja null.
    """
    root = schema.model_json_schema()

    def close(node: object) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("properties"), dict):
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                close(value)
        elif isinstance(node, list):
            for item in node:
                close(item)

    close(root)
    return {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "strict": True, "schema": root},
    }


def _parse(raw: str, schema: type[BaseModel]) -> BaseModel:
    """Parsuje JSON i waliduje schematem. Rzuca ValidationError/ValueError."""
    return schema.model_validate(json.loads(_strip_fences(raw)))


class _Reply(NamedTuple):
    """Wspolny kształt odpowiedzi - po nim jedzie cala reszta _call.

    Anthropic i OpenAI oddaja zupelnie inne obiekty; normalizujemy je TUTAJ,
    zeby parsowanie, retry, healing i logowanie nie wiedzialy o dostawcy.
    """

    text: str              # blok tekstowy z JSON-em
    urls: list[str]        # URL-e z wyszukiwania
    stop: str              # "end_turn" | "max_tokens" | "pause_turn" | ...
    in_tokens: int
    out_tokens: int
    echo: object           # co dolozyc jako content asystenta przy ponowieniu
    shell: list[str] = []  # wyjscia z sandboxa (openrouter_shell_tool_result)


def _send_anthropic(
    model: str,
    system_prompt: str,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int,
) -> _Reply:
    response = _anthropic_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
        **({"tools": tools} if tools else {}),
    )
    usage = response.usage
    dumped = response.model_dump() if hasattr(response, "model_dump") else {}
    return _Reply(
        text=_text_of(response),
        urls=_search_urls_anthropic(response),
        stop=response.stop_reason or "end_turn",
        in_tokens=usage.input_tokens,
        out_tokens=usage.output_tokens,
        # Pelne bloki - inaczej przy ponowieniu gubimy wyniki narzedzia.
        echo=response.content,
        shell=_shell_outputs(dumped),
    )


def _send_openai(
    model: str,
    system_prompt: str,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int,
    response_format: dict | None = None,
) -> _Reply:
    extra: dict = {}
    if tools:
        # Narzedzia bramki (openrouter:*) i max_tool_calls ida przez extra_body -
        # SDK OpenAI nie zna tych kształtow i nie przepuscilby ich typami.
        extra["extra_body"] = {"tools": tools, "max_tool_calls": _MAX_TOOL_CALLS}
    if response_format:
        extra["response_format"] = response_format

    completion = _openai().chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        # System prompt idzie jako pierwsza wiadomosc, nie osobny parametr.
        messages=[{"role": "system", "content": system_prompt}, *messages],
        **extra,
    )
    choice = completion.choices[0]
    usage = completion.usage
    dumped = choice.model_dump() if hasattr(choice, "model_dump") else {}

    # Zrodla wracaja jako cytowania (annotations) albo jako bloki wynikow -
    # zaleznie od tego, co bramka odda. Bierzemy jedno i drugie.
    urls = _collect_urls(dumped.get("message", {}).get("annotations"))
    for url in _collect_urls(_blocks_of_type(dumped, "openrouter_web_search_tool_result")):
        if url not in urls:
            urls.append(url)

    return _Reply(
        text=choice.message.content or "",
        urls=urls,
        # finish_reason "length" to odpowiednik anthropicowego "max_tokens".
        stop="max_tokens" if choice.finish_reason == "length" else "end_turn",
        in_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
        out_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        echo=choice.message.content or "",
        shell=_shell_outputs(dumped),
    )


def _send(
    provider: str,
    model: str,
    system_prompt: str,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int,
    response_format: dict | None = None,
) -> _Reply:
    """Rozgalezienie po dostawcy - jedyne miejsce, ktore zna oba SDK."""
    if provider == "anthropic":
        return _send_anthropic(model, system_prompt, messages, tools, max_tokens)
    if provider == "openai":
        return _send_openai(
            model, system_prompt, messages, tools, max_tokens, response_format
        )
    raise ValueError(f"nieznany dostawca: {provider!r} (mam: anthropic, openai)")


def _mentions(err: Exception, words: tuple[str, ...]) -> bool:
    text = str(err).lower()
    return any(word in text for word in words)


def _call(
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    tools: list[dict] | None,
    max_tokens: int,
) -> tuple[BaseModel, list[str], list[str]]:
    """Wspolna sciezka dla funkcji publicznych.

    Zwraca (zwalidowany obiekt, URL-e z wyszukiwania, wyjscia z sandboxa).
    Format odpowiedzi jest wymuszany przez response_format (json_schema,
    strict), ale healing zostaje jako zabezpieczenie: przy bledzie parsowania
    idzie JEDNA ponowna proba z trescia bledu w rozmowie.

    Jesli odpowiedz zostala ucieta na limicie, ponowienie idzie z limitem
    PODWOJONYM (do _MAX_TOKENS_CEILING) i z prosba o zwiezlosc - powtarzanie
    tego samego limitu daje dokladnie to samo obciecie.
    """
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    system_prompt = f"{system}\n\n{_JSON_INSTRUCTION}{schema_json}"
    provider = provider_for(model)
    messages: list[dict] = [{"role": "user", "content": user}]
    response_format: dict | None = _strict_json_schema(schema)
    raw = ""
    limit = max_tokens
    truncated = False

    def send(limit: int) -> _Reply:
        """Wysyla zapytanie; przy odrzuceniu response_format ponawia bez niego."""
        nonlocal response_format
        try:
            return _send(
                provider, model, system_prompt, messages, tools, limit, response_format
            )
        except (anthropic.BadRequestError, openai.BadRequestError) as err:
            if response_format is None or not _mentions(
                err, ("response_format", "json_schema", "schema")
            ):
                raise
            print(
                "[llm] bramka nie wpuszcza response_format/json_schema - "
                "lece bez niego, zostaje healing"
            )
            response_format = None
            return _send(provider, model, system_prompt, messages, tools, limit, None)

    for attempt in (1, 2):
        urls: list[str] = []
        shell: list[str] = []

        # Narzedzie moze przerwac ture (pause_turn) - wtedy dosylamy ja dalej.
        for _ in range(_MAX_CONTINUATIONS + 1):
            started = time.monotonic()
            reply = send(limit)
            elapsed = time.monotonic() - started
            urls += [u for u in reply.urls if u not in urls]
            shell += reply.shell
            print(
                f"[llm] model={model} ({provider}) proba={attempt} "
                f"czas={elapsed:.2f}s in={reply.in_tokens} out={reply.out_tokens} "
                f"stop={reply.stop} urls={len(urls)} shell={len(shell)}"
            )
            if reply.stop != "pause_turn":
                break
            messages.append({"role": "assistant", "content": reply.echo})

        # Uciecie na max_tokens objawia sie jako blad parsowania w polowie
        # JSON-a - bez tego logu szuka sie go dlugo i w zlym miejscu.
        truncated = reply.stop == "max_tokens"
        if truncated:
            print(
                f"[llm] UWAGA: odpowiedz ucieta na max_tokens={limit} "
                "- JSON jest niepelny"
            )

        raw = reply.text
        try:
            return _parse(raw, schema), urls, shell
        except (ValidationError, ValueError) as err:
            if attempt == 2:
                break
            print(f"[llm] walidacja nieudana, ponawiam: {err}")

            if truncated:
                nowy_limit = min(limit * 2, _MAX_TOKENS_CEILING)
                if nowy_limit > limit:
                    print(f"[llm] podnosze limit {limit} -> {nowy_limit} na ponowienie")
                    limit = nowy_limit
                else:
                    print(f"[llm] limit {limit} to sufit - ponawiam bez podnoszenia")

            messages += [
                # U Anthropica pelne bloki, nie sam tekst - inaczej gubimy
                # wyniki narzedzia. U OpenAI to zwykly string (patrz _send).
                {"role": "assistant", "content": reply.echo},
                {
                    "role": "user",
                    "content": (
    "Your response failed validation:\n"
    f"{err}\n\n"
    "Fix it and return ONLY a valid JSON object matching the schema."
    + (
        " Your response was TOO LONG and was cut off mid-way. "
        "Answer CONCISELY: short text fields, no elaboration or "
        "repetition, only essential code. Fewer claims beats "
        "incomplete JSON."
        if truncated
        else ""
    )
),
                },
            ]

    if truncated:
        # Bez tego wyjatek pokazuje "Unterminated string", co wyglada na problem
        # ze schematem, a problemem jest dlugosc odpowiedzi.
        raise ValueError(
            f"call_structured: odpowiedz modelu {model} zostala ucieta na "
            f"max_tokens={limit} (nie blad schematu) - {schema.__name__} jest "
            f"niepelny po dwoch probach. Podnies limit dla tego wywolania albo "
            f"kaz modelowi odpowiadac krocej. Surowa odpowiedz:\n{raw}"
        )

    raise ValueError(
        f"call_structured: model {model} nie zwrocil poprawnego "
        f"{schema.__name__} po dwoch probach. Surowa odpowiedz:\n{raw}"
    )


def _call_with_tool_fallback(
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    tools: list[dict],
    max_tokens: int,
) -> tuple[BaseModel, list[str], list[str]]:
    """Wola z narzedziami; gdy bramka je odbije - wola bez nich.

    Fallback jest jawny, nie cichy: wolajacy dostaje pusta liste URL-i, wiec
    walidacja zrodel u niego wyzeruje kazdy source_url - brak zrodla zamiast
    zrodla niesprawdzonego.
    """
    try:
        return _call(model, system, user, schema, tools=tools, max_tokens=max_tokens)
    except (anthropic.BadRequestError, openai.BadRequestError) as err:
        if not _mentions(err, ("tool", "narzedzi")):
            raise
        print(
            f"[llm] UWAGA: bramka odbila narzedzia ({str(err)[:160]}). "
            "Lece bez nich - zadne source_url nie przejdzie walidacji."
        )
        parsed, _, _ = _call(
            model, system, user, schema, tools=None, max_tokens=max_tokens
        )
        return parsed, [], []


def call_structured(
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    max_tokens: int = MAX_TOKENS,
) -> BaseModel:
    """Wola model przez bramke Orbio i zwraca obiekt `schema`."""
    parsed, _, _ = _call(model, system, user, schema, tools=None, max_tokens=max_tokens)
    return parsed


def call_structured_with_search(
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    max_tokens: int = MAX_TOKENS,
) -> tuple[BaseModel, list[str]]:
    """Jak call_structured, ale model ma do dyspozycji wyszukiwarke.

    Zwraca (obiekt, search_urls). Wolajacy MUSI sprawdzic, czy URL-e podane
    przez model naleza do search_urls - inaczej model przemyci link z wag.
    """
    parsed, urls, _ = _call_with_tool_fallback(
        model, system, user, schema, [_search_tool()], max_tokens
    )
    return parsed, urls


def call_structured_with_tools(
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    max_tokens: int = MAX_TOKENS,
    container_id: str | None = None,
) -> tuple[BaseModel, list[str], list[str]]:
    """Wyszukiwarka + (opcjonalnie) sandbox. Zwraca (obiekt, urls, shell).

    container_id wskazuje kontener debaty - jeden na debate, zeby kolejne
    wywolania kwanta trafialy do tego samego srodowiska.
    """
    tools = [_search_tool()]
    if container_id:
        tools.append(_shell_tool(container_id))
    return _call_with_tool_fallback(model, system, user, schema, tools, max_tokens)


if __name__ == "__main__":
    from config import BEAR_MODEL, BULL_MODEL
    from models import Claim

    SYSTEM = (
    "You are the bull agent in an investment debate. Make one claim "
    "supporting the positive thesis. agent='bull', round=1. stake is an "
    "integer 1-20, confidence a number 0-1. With no source, set "
    "source_url to null."
    )
    USER = "Make one claim about NVDA stock."

    # Dokladnie te modele beda sie spierac w debacie.
    for label, model in (("BULL", BULL_MODEL), ("BEAR", BEAR_MODEL)):
        print(f"\n===== {label}: {model} =====")
        try:
            claim = call_structured(model=model, system=SYSTEM, user=USER, schema=Claim)
            print(f"  stake={claim.stake} confidence={claim.confidence} typ={claim.claim_type}")
            print(f"  {claim.text}")
        except Exception as err:
            print(f"  NIE UDALO SIE: {type(err).__name__}: {str(err)[:200]}")
