from anthropic import Anthropic
from config import ORBIO_BASE_URL, ORBIO_KEY

client = Anthropic(base_url=ORBIO_BASE_URL, auth_token=ORBIO_KEY, api_key=None)

resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=100,
    messages=[{"role": "user", "content": "Odpowiedz jednym zdaniem: dziala?"}],
)
print(resp.content[0].text)