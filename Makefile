fmt:
	cargo fmt

lint:
	cargo clippy --all-targets --all-features -- -D warnings

test:
	cargo test --all-features

check:
	cargo check

audit:
	cargo audit

ci: fmt lint check test