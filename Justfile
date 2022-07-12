set dotenv-load := true

run:
  bun run dev --port 3040

build-wasm:
  cd src/engine && just build

build:
  just build-wasm
  bun run build

preview:
  bun run preview --port 3040 --host 0.0.0.0
