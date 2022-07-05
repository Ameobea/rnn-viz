set dotenv-load := true

run:
  bun run dev --port 3040

build:
  bun run build

preview:
  bun run preview --port 3040 --host 0.0.0.0
