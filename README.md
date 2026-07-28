# sush.one

Personal blog and portfolio. Jekyll + the [Duet](https://jekyllthemes.io/theme/duet)
theme, deployed to GitHub Pages.

## Quick start

Docker is the only prerequisite — you don't need Ruby installed.

```bash
make serve     # → http://localhost:4100
```

The first run builds the image (a minute or two); after that it's cached, and
edits to your content appear live without a rebuild.

Run `make` on its own to list every command.

## Writing

Content is Markdown with YAML front matter, filed by category:

| Where | What |
| --- | --- |
| `_posts/<category>/` | Blog posts — filename must be `YYYY-MM-DD-title.md` |
| `_pages/<category>/` | Standalone pages |
| `images/` | Image assets |

Post categories: `art`, `food`, `projects`, `blog`. Pages add `work`.
Copy an existing file in the same folder as your template — the front matter it
needs is already there.

Posts under `_posts/blog/` get the `blog-post` layout
automatically; everything else gets `post`. See `defaults:` in `_config.yml`.

A `projects` collection is declared in `_config.yml` but has no `_projects/`
directory — portfolio entries live in `_pages/projects/` instead.

## Drafts

To keep a post out of the published site while still previewing it locally, add
`published: false` to its front matter:

```yaml
---
title: Work in progress
published: false
---
```

`make serve` and `make staging` pass `--unpublished`, so drafts render locally.
`make prod` and CI omit it, so they show exactly what goes live. Delete the line
to publish.

Currently drafted: `_posts/blog/concepts/2024-01-05-nerf.md`.

Note that this hides the *page*, not its images — anything under `images/` is
still copied to the site and reachable by direct URL, just unlinked. Move the
folder out of `images/` if the assets themselves need to be private.

## Configuration

| File | Controls |
| --- | --- |
| `_config.yml` | Jekyll build: collections, permalinks, Markdown, plugins |
| `config/staging.yml`, `config/production.yml` | Per-environment overrides, layered on top of `_config.yml` |
| `_data/settings.yml` | Theme appearance: colors, fonts, nav menu, social links, contact form |

Most day-to-day changes (menu, colors, links) live in `_data/settings.yml`, not
`_config.yml`.

## Build files

These exist for tooling and are excluded from the published site:

| Path | Purpose |
| --- | --- |
| `Makefile` | Entrypoint for every command — start here |
| `config/` | Per-environment Jekyll config overrides |
| `docker/` | The image and compose file behind every `make` target |
| `docs/duet-theme/` | The Duet theme's own docs and license, kept for reference |
| `Gemfile`, `Gemfile.lock` | Ruby dependencies, pinned — installed into the image |
| `.ruby-version` | Ruby version (3.1), read by CI and the Docker build |
| `.github/workflows/pages.yml` | CI build on every push (native Ruby, not Docker) |

Everything else at the repo root has to be there: `CNAME` (GitHub Pages),
`_config.yml` (Jekyll), `index.md` and `404.html` (site routes), `README.md`,
`Makefile`, and the `Gemfile` pair.

## Deploying

Push to `main`. GitHub Pages builds and publishes; there's no manual deploy step.
