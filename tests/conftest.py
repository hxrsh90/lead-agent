import os

# Set required env vars before any module imports config.py
os.environ.setdefault("AGENT_MODE", "custom")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("CLAY_API_KEY", "test-key")
os.environ.setdefault("VIBE_API_KEY", "test-key")
os.environ.setdefault("DB_PATH", ":memory:")
