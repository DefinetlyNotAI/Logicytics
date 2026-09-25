# Security policy

## Supported versions

| Version | Security support | Release date       |
| ------- | ---------------- | ------------------ |
| 4.0.x   | Supported        | September 25, 2026 |
| 3.6.x   | Partial Support  | July 26, 2025      |
| 3.5.x   | Partial Support  | July 26, 2025      |
| 3.4.x   | Partial Support  | January 3, 2025    |
| 3.3.x   | Unsupported      | January 3, 2025    |
| 3.2.x   | Unsupported      | December 19, 2024  |
| 3.1.x   | Unsupported      | December 11, 2024  |
| 3.0.x   | Unsupported      | December 6, 2024   |
| 2.5.x   | Unsupported      | November 25, 2024  |
| 2.4.x   | Unsupported      | November 12, 2024  |
| 2.3.x   | Unsupported      | September 21, 2024 |
| 2.2.x   | Unsupported      | September 9, 2024  |
| 2.1.x   | Unsupported      | August 29, 2024    |
| 2.0.x   | Unsupported      | August 25, 2024    |
| 1.6.x   | Unsupported      | June 18, 2024      |
| 1.5.x   | Unsupported      | June 10, 2024      |
| 1.4.x   | Unsupported      | May 30, 2024       |
| 1.3.x   | Unsupported      | May 21, 2024       |
| 1.2.x   | Unsupported      | May 16, 2024       |
| 1.1.x   | Unsupported      | May 10, 2024       |
| 1.0.x   | Unsupported      | May 4, 2024        |

Only the current v4 release line receives security fixes. Upgrade through the
documented [migration boundary](docs/MIGRATION.md); do not run an unsupported checkout
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
