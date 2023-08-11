test:
  cargo test --verbose --release --all-features

fmt:
  cargo fmt --verbose --all -- --check

clippy:
  cargo clippy --verbose --all-targets --all-features -- -D warnings