# Market Brief

Template for the daily Market Brief page, published as a Claude Artifact and linked from the Home page of the Hybrid Build app.

- Live page: https://claude.ai/code/artifact/061f66cf-9e85-440f-bd51-5a4a40565d14
- Refreshed daily at 7:00am Singapore time (23:00 UTC) by a scheduled Claude Code cloud agent (routine), which:
  1. Reads `market-brief.html` from this folder for the current design/layout
  2. Searches the web for that day's real market news
  3. Edits only the content (date, index snapshot, news, stocks to watch, risks, sources) — never the CSS or fonts
  4. Commits the updated file back here
  5. Republishes the same Artifact URL above

Not financial advice — informational summary only.
