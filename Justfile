set dotenv-load := true

run:
  yarn dev --port 3040

build:
  yarn build

preview:
  yarn preview --port 3040 --host 0.0.0.0
