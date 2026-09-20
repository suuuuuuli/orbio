# Financial Court

Four models argue an investment case, one rules on it, nobody gets to make things up.

**Live:** https://financialcourt.xyz

Built during Orbio Build Week.

## What it does

You pick an asset — a stock ticker or a token. A bull and a bear each build a case
over two or three rounds. An expert witness runs calculations. A judge rules on every
claim and settles a credibility budget. The verdict is a thesis with an explicit
confidence level and a list of falsifiers — never a buy or sell recommendation.

You can bet on a side, top up between rounds at live odds, interrupt with an objection
mid-case, and question the judge after the verdict.

## Why it isn't just four chatbots

Every factual claim carries a source URL and a quote. After the model answers, the code
checks that the quote appears **verbatim** in the document it points at — if not, the
claim loses its source and its stake. The expert witness writes Python that actually
executes, and its numbers must trace back to the source pack. Even the judge is checked:
a `verified` ruling without a real quote gets overridden to `unsourced` in code.

Sources come from SEC EDGAR filings for equities and DeFiLlama for tokens, assembled
server-side. The agents can't search, so they can't cite from memory.

## Models

| Role | Model | Provider |
|---|---|---|
| Delusional Bull | gemini-3.1-pro | Google |
| Permabear | gpt-5.5 | OpenAI |
| Expert Witness | gpt-5.3-codex | OpenAI |
| Market Wizard (judge) | claude-fable-5 | Anthropic |

Four models, three providers, one Orbio key. The pixel art was generated through the
same key.


