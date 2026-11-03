# Source and release provenance

This release is a clean source snapshot prepared and published at the project maintainer's direction. It does not publish the historical private repository or its Git history.

Three core modules originate from the archived Hui Phase-D source snapshot dated 2026-08-08. `provenance/sources.json` records original and packaged SHA-256 values. The two documented changes convert imports to package-relative imports; source mathematics is unchanged. Verify with `python tools/verify_sources.py`.

The adapter, synthetic demo, tests and packaging surround that frozen core. WKK Lab is the project/report attribution. Apache-2.0 is the project license; separately identified third-party assets retain their own terms. Source hashes establish byte provenance, not a comprehensive independent ownership or transitive-license audit.

The report distinguishes historical archived experiments from installed-package CPU checks, and distinguishes per-block N2 from later segmented C4/C8 variants. No weights, corpus, credentials, private host configuration or old Git history are included. Historical raw checkpoints are not redistributed or newly rehashed as part of this release.

The absence of a redistributable GPU reproduction bundle is a limitation: no independent N2 50M-token reproduction or general GPU memory/throughput guarantee is claimed.
