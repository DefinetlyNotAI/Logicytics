# Security policy

## Supported versions

| Version | Security support | Release date |
| --- | --- | --- |
| 4.0.x | Supported | 2026-08-31 |
| 3.x and earlier | Unsupported | — |

Only the current v4 release line receives security fixes. Upgrade through the
documented [migration boundary](MIGRATION.md); do not run an unsupported checkout
against sensitive evidence.

## Reporting a vulnerability

Do not open a public issue containing an exploit, credential, private key,
collected artifact, host identifier, or unredacted log. Email
[Nirt_12023@outlook.com](mailto:Nirt_12023@outlook.com) with the subject
`Logicytics security vulnerability`.

Include:

- the affected Logicytics version and commit;
- Windows edition/build and Python version;
- the affected collector, command, capability, or package contract;
- reproducible steps and expected security boundary;
- impact and required authorization/elevation state; and
- a minimal redacted proof of concept.

Reports should target behavior owned by this repository and be reproducible.
Maintainers will acknowledge receipt, investigate the report, coordinate a fix
and disclosure window when accepted, and credit the reporter if requested.

## Security boundaries

Logicytics requires authorization acknowledgement and explicit capability
approval. A vulnerability includes bypassing those checks, escaping a collector
workspace, publishing undeclared evidence, leaking secrets to logs/metadata,
executing an undeclared process or network action, corrupting package/hash
verification, or allowing one collector to affect unrelated runs.

Do not weaken isolation, redaction, path validation, output limits, cancellation,
or package verification while developing a fix. Use synthetic fixtures only.
