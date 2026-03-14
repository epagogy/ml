# Security Policy

## Supported versions

| Version | Status |
|---------|--------|
| 1.x | Active |

## Reporting a vulnerability

**Do not open a public GitHub issue.**

Email **security@epagogy.ai** with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

Response within 7 days. We follow responsible disclosure: 90 days to patch before public disclosure.

## Scope

**In scope:**
- Model deserialization vulnerabilities (`ml.load()` uses skops with a trusted-types whitelist)
- Dependency chain vulnerabilities
- Data leakage bugs affecting security-sensitive predictions

**Out of scope:** issues in user-supplied data or models (ml is a library, not a service).

## Model loading

`ml.load()` uses skops format with a trusted-types whitelist. Only known-safe sklearn, XGBoost, LightGBM, and CatBoost types are allowed. Do not load model files from untrusted sources.
