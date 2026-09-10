# Upstream data and reproducibility

The public GitHub ophthalmology project is an upstream reference and is not copied into this repository. This keeps the release small and preserves the upstream license boundary.

Upstream repository:

`https://github.com/qqnnhhdmpc666/low-resource-fundus-qa`

For a local integration checkout:

```powershell
git clone https://github.com/qqnnhhdmpc666/low-resource-fundus-qa.git vendor/low-resource-fundus-qa
```

The release includes `data/smoke_qa.jsonl` so the Agent can be tested immediately after cloning without downloading the large public corpus or a model. Its records are synthetic smoke fixtures, not clinical ground truth.

The upstream QA `answer/output` fields must remain separate from retrieval evidence. They may be used in a separately audited training view, but they are not automatically injected into the Evidence Verifier.

