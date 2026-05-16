# Releasing

## PyPI

Releases are published to PyPI automatically when a pull request is merged into `main` with a version bump tag in the PR title:

| Tag in PR title | Version bump | Example |
|---|---|---|
| `[patch]` | Bug fixes, docs (0.1.0 → 0.1.1) | `Fix conformer bug [patch]` |
| `[minor]` | New features, backwards-compatible (0.1.0 → 0.2.0) | `Add ProtonationBlock [minor]` |
| `[major]` | Breaking changes (0.1.0 → 1.0.0) | `Redesign block API [major]` |

PRs without a tag merge normally without triggering a release.

## MCP Registry

cmxflow is also published to the [MCP Registry](https://registry.modelcontextprotocol.io) so LLM clients can discover the `cmxflow-mcp` server.

### One-time setup

Install the publisher CLI:

```bash
brew install mcp-publisher  # macOS
# or see https://github.com/modelcontextprotocol/registry/releases for other platforms
```

### Per-release flow

After a new version lands on PyPI:

1. Update `server.json`:
   - `version` (top level)
   - `packages[0].version`
   Both must match the PyPI version exactly.

2. Verify ownership marker still exists in `README.md`:
   ```html
   <!-- mcp-name: io.github.b-shields/cmxflow -->
   ```
   The MCP Registry reads the PyPI long description and verifies this string matches the `name` in `server.json`. Do not remove it.

3. Authenticate and publish:
   ```bash
   mcp-publisher login github
   mcp-publisher publish
   ```

4. Verify:
   ```bash
   curl "https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.b-shields/cmxflow"
   ```
