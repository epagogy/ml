# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.x     | ✓ Active  |

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email: security@mlwork.org

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

You will receive a response within 7 days. We follow responsible disclosure — we ask for 90 days to patch before public disclosure.

## Scope

Security issues we care about:
- Model deserialization vulnerabilities (`ml.load()` uses skops with a trusted-types whitelist)
- Dependency vulnerabilities in the dependency chain
- Data leakage bugs that could cause incorrect security-sensitive ML predictions

Out of scope: issues in user-supplied data or models (we're a library, not a service).

## Known security considerations

**Model loading:** `ml.load()` uses skops format with a trusted-types whitelist. Only known-safe sklearn, XGBoost, LightGBM, and CatBoost types are allowed. Do not load model files from untrusted sources — this applies to any ML serialization format.
